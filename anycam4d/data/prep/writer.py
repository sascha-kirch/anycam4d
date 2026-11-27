"""Native incremental dataset writer: ``CalibratedView`` -> AnyCam4D dataset directory."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from anycam4d.camera.registry import param_count_range
from anycam4d.data.prep.pose import rotation_matrix_to_quat
from anycam4d.data.schema import DatasetSchema
from anycam4d.point_ops.ply import write_ply


if TYPE_CHECKING:
    from collections.abc import Iterable

    from jaxtyping import Float

    from anycam4d.data.prep.sample import CalibratedView
    from anycam4d.enums import CameraModelType


_VALID_IMAGE_FORMATS = frozenset({"webp", "png", "jpg"})
# WEBP for alpha views is *lossy* (q95), never lossless: libwebp always compresses the alpha plane
# losslessly regardless of quality, so a frame's binary validity mask stays exactly 0/255
# while its RGB is lossily compressed like every opaque view. Lossless WEBP was ~7x slower
# (~2.9 s vs ~0.4 s per 2560x1920 frame) and ballooned encoder memory by ~1.5 GB per call — enough
# to OOM a multi-camera parallel run — for a *larger* file than lossy, so it is not used at all.
_WEBP_QUALITY = 95
_PNG_COMPRESSION = 3  # only used if a caller explicitly selects png; lossless but ~10x larger than lossy webp
_JPEG_QUALITY = 95  # cv2 IMWRITE_JPEG_QUALITY for opaque views


def _encode_params(fmt: str) -> list[int]:
    """cv2 encode params for ``fmt``: lossy quality for webp/jpg, compression level for png."""
    return {
        "png": [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION],
        "webp": [cv2.IMWRITE_WEBP_QUALITY, _WEBP_QUALITY],
        "jpg": [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
    }.get(fmt, [])


class PrepWriter:
    """Streams encoded views into an AnyCam4D dataset; JSON, point cloud and manifest are written in ``finalize``.

    Calibration is passed through verbatim: rescaling or model rewriting is a ``Stage``'s job.
    """

    def __init__(
        self,
        out_dir: Path,
        eval_cams: Iterable[int] = (),
        image_format: str = "webp",
    ) -> None:
        """Create the writer and its output directory.

        Args:
            out_dir: Dataset root directory (created if absent).
            eval_cams: Camera ids routed to ``eval_dataset.json`` instead of the train split.
            image_format: ``"webp"`` (lossless) or ``"png"``.
        """
        if image_format not in _VALID_IMAGE_FORMATS:
            raise ValueError(f"Got unsupported image_format '{image_format}'; expected one of {_VALID_IMAGE_FORMATS}")

        self._out_dir = Path(out_dir)
        self._images_dir = self._out_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._eval_cams = set(eval_cams)
        self._image_format = image_format

        self._cameras: dict[int, dict] = {}
        self._train_captures: list[dict] = []
        self._eval_captures: list[dict] = []

    def register_camera(
        self,
        cam_id: int,
        model: CameraModelType,
        params: list[float],
        width: int,
        height: int,
        reproject_source: dict | None = None,
        max_incidence_deg: float | None = None,
    ) -> None:
        """Record one camera's intrinsics (validated), without needing a decoded image.

        ``reproject_source`` is emitted only when the view
        actually was resampled off another lens; a genuine native camera leaves it absent, so it is
        written when present and never required. ``max_incidence_deg`` is the lens's recorded field
        limit, which no distortion polynomial encodes; it is written only for a camera whose own
        lens it describes, so a reprojected camera (whose footprint is replayed through ``reproject_source``) leave it absent.
        """
        lo, hi = param_count_range(model)
        n = len(params)
        if not (lo <= n <= hi):
            raise ValueError(f"Camera {cam_id}: model '{model.value}' expects {lo}..{hi} params, got {n}")
        camera = {"camera_id": cam_id, "model": model.value, "width": width, "height": height, "params": params}
        if reproject_source is not None:
            camera["reproject_source"] = reproject_source
        if max_incidence_deg is not None:
            camera["max_incidence_deg"] = max_incidence_deg
        self._cameras[cam_id] = camera

    def add_capture(
        self, cam_id: int, frame_id: int, timestamp: float, c2w: Float[np.ndarray, "4 4"], image_path: str
    ) -> None:
        """Record one capture (frame + pose + timestamp + image path), routed by the eval split."""
        capture = {
            "image_path": image_path,
            "camera_id": cam_id,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "rotation": rotation_matrix_to_quat(c2w[:3, :3]),
            "translation": c2w[:3, 3].tolist(),
        }
        captures = self._eval_captures if cam_id in self._eval_cams else self._train_captures
        captures.append(capture)

    def finalize(
        self,
        points: Float[np.ndarray, "N 3"] | None = None,
        colors: Float[np.ndarray, "N 3"] | None = None,
        manifest: dict | None = None,
    ) -> None:
        """Write train/eval JSON, ``points3d.ply``, and the provenance manifest sidecar."""
        _check_frame_aligned(self._train_captures + self._eval_captures)
        self._write_dataset_json(self._train_captures, "train_dataset.json", include_pointcloud=points is not None)
        if self._eval_captures:
            self._write_dataset_json(self._eval_captures, "eval_dataset.json", include_pointcloud=False)
        if points is not None:
            self._write_ply(points, colors)
        if manifest is not None:
            with (self._out_dir / "prep_manifest.json").open("w") as f:
                json.dump(manifest, f, indent=2)

    def encode_view(self, view: CalibratedView) -> str:
        """Encode one capture (RGB->BGR, mask as alpha) and write it atomically; skip if present.

        Thread-safe: reads only immutable config (``_images_dir``, ``_image_format``) and writes a
        per-capture file, so the parallel renderer can call it concurrently across cameras without a
        lock. Returns the dataset-relative image path (for ``add_capture``).

        The output format follows the view's *actual* alpha: a fully-opaque view (no mask, or a
        mask that is all-valid) is written as fast jpg regardless of the configured format — the
        alpha-capable ``image_format`` (webp/png) is used only when the mask genuinely carries
        transparency (e.g. a matting mask). A
        file already on disk is kept (resume); the encode goes to a ``.part`` sibling and is
        atomically renamed into place, so an interrupt never leaves a truncated image a resume trusts.
        """
        has_alpha = view.mask is not None and not bool((view.mask == 255).all())
        fmt = self._pick_format(view.cam_id, has_alpha)
        filename = f"cam{view.cam_id:02d}_{view.frame_id:04d}.{fmt}"
        final = self._images_dir / filename
        rel = f"images/{filename}"
        if final.exists():
            return rel

        bgr = view.image[..., ::-1]
        image = np.concatenate([bgr, view.mask[..., None]], axis=-1) if has_alpha else bgr
        ok, buf = cv2.imencode(f".{fmt}", image, _encode_params(fmt))
        if not ok:
            raise RuntimeError(f"Camera {view.cam_id}: failed to encode {filename}")

        part = final.with_name(final.name + ".part")
        part.write_bytes(buf.tobytes())
        part.replace(final)
        return rel

    def _pick_format(self, cam_id: int, has_alpha: bool) -> str:
        """The output format for a view: jpg when opaque, else the alpha-capable configured format."""
        if not has_alpha:
            return "jpg"  # opaque -> fast jpg, whatever alpha format was configured
        if self._image_format == "jpg":
            raise ValueError(f"Camera {cam_id}: view has a transparent mask; jpg cannot store it (use webp or png)")
        return self._image_format

    def _write_dataset_json(self, captures: list[dict], filename: str, include_pointcloud: bool) -> None:
        """Assemble, self-validate, and write one ``*_dataset.json`` split file."""
        cam_ids = sorted({c["camera_id"] for c in captures})
        doc: dict = {
            "schema_version": 1,
            "metadata": {
                "world_coordinate_system": "right_handed",
                "camera_coordinate_system": "opencv",
                "transform_convention": "camera_to_world",
            },
            "cameras": [self._cameras[cid] for cid in cam_ids],
            "captures": captures,
        }
        if include_pointcloud:
            doc["pointcloud_path"] = "points3d.ply"

        DatasetSchema.model_validate(doc)  # fail fast: never emit an unloadable dataset
        with (self._out_dir / filename).open("w") as f:
            json.dump(doc, f, indent=2)

    def _write_ply(self, points: Float[np.ndarray, "N 3"], colors: Float[np.ndarray, "N 3"] | None) -> None:
        """Write ``points3d.ply``: xyz + rgb (default gray when uncolored)."""
        write_ply(self._out_dir / "points3d.ply", points, colors)


def _check_frame_aligned(captures: list[dict]) -> None:
    """Fail on a dataset the loader cannot group into frames: every frame must be observed by the same cameras.

    The cameras of a frame may carry different timestamps; only ``sampler=frame_aligned`` refuses that.
    """
    if not captures:
        return

    cameras_by_frame: dict[int, set[int]] = defaultdict(set)
    for cap in captures:
        cameras_by_frame[cap["frame_id"]].add(cap["camera_id"])

    sizes = {len(cams) for cams in cameras_by_frame.values()}
    if len(sizes) > 1:
        ragged = sorted(fid for fid, cams in cameras_by_frame.items() if len(cams) != max(sizes))[:5]
        raise ValueError(
            f"frames observed by differing camera counts {sorted(sizes)} (e.g. frames {ragged}); "
            f"every frame must be observed by the same cameras."
        )
