"""Anycam4DSource: reads a canonical AnyCam4D dataset directory back into pivot views.

The read-side counterpart to ``PrepWriter``: enables post-hoc variant generation (undistort,
resize, ...) from an already-imported canonical dataset without re-running
the (expensive) importer. Merges the train and eval splits into one frame-aligned stream — the
split is a *writer* concern (``PrepWriter(eval_cams=...)``), not a property of the pivot.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import distortion_width
from anycam4d.data.prep.calib import capture_calibration
from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.stream import SequenceSourceProto
from anycam4d.data.schema import DatasetSchema
from anycam4d.math.lens_domain import max_observed_incidence
from anycam4d.point_ops.ply import read_ply


if TYPE_CHECKING:
    from collections.abc import Iterator

    from jaxtyping import Float
    from jaxtyping import UInt8
    from pydantic import BaseModel

    from anycam4d.data.schema import CameraSchema
    from anycam4d.data.schema import CaptureSchema
    from anycam4d.enums import CameraModelType


@dataclass(frozen=True)
class CaptureRef:
    """A pixel-free handle to one capture: everything a worker needs to decode + transform it.

    Calibration is resolved up front (cheap, no pixels); the image itself is decoded lazily inside
    the worker via ``load_view`` so decoding parallelizes with the transform/encode instead of
    running serially in the main thread.
    """

    cam_id: int
    frame_id: int
    timestamp: float
    image_path: Path  # absolute path to the source image on disk
    model: CameraModelType
    K: Float[np.ndarray, "3 3"]
    params: list[float]
    c2w: Float[np.ndarray, "4 4"]
    reproject_source: dict | None
    max_incidence_deg: float | None = None


class Anycam4DSource(SequenceSourceProto):
    """Reads ``train_dataset.json`` (+ optional ``eval_dataset.json``) into ``CalibratedView``s.

    Both splits are merged into a single frame-aligned stream; ``eval_camera_ids`` records
    which camera ids came from the eval split so ``build_variant`` can reproduce the split
    faithfully on write. Frames are grouped on the capture's own ``frame_id``, so a variant
    round-trips the base dataset's frame numbering unchanged.
    """

    def __init__(self, data_dir: Path) -> None:
        """Parse the train (required) and eval (optional) splits of a canonical dataset."""
        self._data_dir = Path(data_dir)
        train_schema = _load_schema(self._data_dir / "train_dataset.json")
        eval_path = self._data_dir / "eval_dataset.json"
        eval_schema = _load_schema(eval_path) if eval_path.exists() else None

        self._cameras_by_id: dict[int, CameraSchema] = {cam.camera_id: cam for cam in train_schema.cameras}
        self._eval_camera_ids: set[int] = set()
        all_captures: list[CaptureSchema] = list(train_schema.captures)
        if eval_schema is not None:
            for cam in eval_schema.cameras:
                self._cameras_by_id[cam.camera_id] = cam
            self._eval_camera_ids = {cam.camera_id for cam in eval_schema.cameras}
            all_captures.extend(eval_schema.captures)

        self._captures_by_frame: dict[int, list[CaptureSchema]] = defaultdict(list)
        for cap in all_captures:
            self._captures_by_frame[cap.frame_id].append(cap)
        self._frame_ids: list[int] = sorted(self._captures_by_frame)

        self._max_incidence: dict[int, float | None] = {
            cam_id: _resolve_max_incidence(self._data_dir, cam) for cam_id, cam in self._cameras_by_id.items()
        }

        self._pointcloud_path: Path | None = (
            self._data_dir / train_schema.pointcloud_path if train_schema.pointcloud_path is not None else None
        )

    @property
    def eval_camera_ids(self) -> set[int]:
        """Camera ids that appear in ``eval_dataset.json``."""
        return set(self._eval_camera_ids)

    def frame_ids(self) -> list[int]:
        """The dataset's frame ids, sorted (stream order); not necessarily dense or 0-based."""
        return list(self._frame_ids)

    def camera_ids(self) -> list[int]:
        """Every camera id in the dataset (train + eval), sorted."""
        return sorted(self._cameras_by_id)

    def stream(self, batch_frames: int = 1) -> Iterator[list[CalibratedView]]:
        """Yield frame-aligned batches, loading each batch's images lazily (bounded RAM)."""
        for start in range(0, len(self._frame_ids), batch_frames):
            batch: list[CalibratedView] = []
            for frame_id in self._frame_ids[start : start + batch_frames]:
                batch.extend(self._build_view(frame_id, cap) for cap in self._captures_by_frame[frame_id])
            yield batch

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "N 3"] | None] | None:
        """Read the train split's point cloud, or ``None`` if it has none."""
        if self._pointcloud_path is None or not self._pointcloud_path.exists():
            return None
        return read_ply(self._pointcloud_path)

    def camera_refs(self) -> dict[int, list[CaptureRef]]:
        """Pixel-free capture refs grouped by camera, each camera's list in frame order.

        Grouping by camera is what lets the parallel renderer run one worker per camera: a stage's
        per-camera resample plan (e.g. the undistort remap grid) is then built once and reused across
        that camera's whole sequence, and the cameras render concurrently.
        """
        refs: dict[int, list[CaptureRef]] = defaultdict(list)
        for frame_id in self._frame_ids:
            for cap in self._captures_by_frame[frame_id]:
                cam = self._cameras_by_id[cap.camera_id]
                model, k, params, c2w = capture_calibration(cap, cam)
                refs[cap.camera_id].append(
                    CaptureRef(
                        cam_id=cap.camera_id,
                        frame_id=frame_id,
                        timestamp=cap.timestamp,
                        image_path=self._data_dir / cap.image_path,
                        model=model,
                        K=k,
                        params=params,
                        c2w=c2w,
                        reproject_source=_dump(cam.reproject_source),
                        max_incidence_deg=self._max_incidence[cap.camera_id],
                    )
                )
        return refs

    def _build_view(self, frame_id: int, cap: CaptureSchema) -> CalibratedView:
        """Reconstruct one ``CalibratedView`` from a capture record and its camera."""
        cam = self._cameras_by_id[cap.camera_id]
        image, mask = _load_image(self._data_dir / cap.image_path)
        model, k, params, c2w = capture_calibration(cap, cam)
        return CalibratedView(
            cam_id=cap.camera_id,
            frame_id=frame_id,
            timestamp=cap.timestamp,
            image=image,
            mask=mask,
            model=model,
            K=k,
            params=params,
            c2w=c2w,
            reproject_source=_dump(cam.reproject_source),
            max_incidence_deg=self._max_incidence[cap.camera_id],
        )


def _dump(block: BaseModel | None) -> dict | None:
    """Provenance block back to the plain dict the prep pivot carries (``None`` passes through)."""
    return block.model_dump() if block is not None else None


def _resolve_max_incidence(data_dir: Path, camera: CameraSchema) -> float | None:
    """One camera's field limit in degrees: recorded if the manifest has it, measured if not.

    A lens's field ends where its optics and its housing end, which its distortion coefficients do
    not describe — a circular fisheye stays monotone well past its own image circle. So the number
    is never inferred from the polynomial. It is read from the manifest, and failing that measured
    off the ``valid/<camera_id>`` mask the capture shipped, which is the only other record of where
    the lens actually stopped. A camera with neither gets ``None``, which asserts the polynomial's
    own domain is the whole story — exact for a lens that fills its frame.

    Measuring is the weaker of the two: it reads the outermost observed pixel *centre*, so a disc
    rasterised on a pixel grid comes back a fraction of a pixel narrow, and it cannot see field the
    capture cropped at the frame border. It is a floor on the true limit, never an overstatement.

    Args:
        data_dir: The dataset directory, whose ``valid/`` subdirectory is consulted.
        camera: The camera's manifest entry.

    Returns:
        The maximum observed incidence angle in degrees, or ``None``.
    """
    if camera.max_incidence_deg is not None:
        return camera.max_incidence_deg
    matches = sorted((data_dir / "valid").glob(f"{camera.camera_id}.*"))
    if len(matches) != 1:
        return None
    with Image.open(matches[0]) as pil_mask:
        arr = np.asarray(pil_mask)
    observed = (arr[..., 0] if arr.ndim == 3 else arr) > 0
    if observed.shape != (camera.height, camera.width):
        return None
    fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Canonical slot order equals the schema's positional order, so a prefix copy places the subset.
    dist = np.zeros(distortion_width(camera.model), dtype=np.float64)
    dist[: len(camera.params) - 4] = camera.params[4:]
    measured = max_observed_incidence(observed, k, dist, CAMERA_REGISTRY[camera.model].family)
    return None if measured is None else math.degrees(measured)


def _load_schema(path: Path) -> DatasetSchema:
    """Parse and validate one ``*_dataset.json`` file."""
    with path.open() as f:
        raw = json.load(f)
    return DatasetSchema.model_validate(raw)


def load_view(ref: CaptureRef) -> CalibratedView:
    """Decode a ``CaptureRef``'s image into a full ``CalibratedView`` (the deferred pixel load)."""
    image, mask = _load_image(ref.image_path)
    return CalibratedView(
        cam_id=ref.cam_id,
        frame_id=ref.frame_id,
        timestamp=ref.timestamp,
        image=image,
        mask=mask,
        model=ref.model,
        K=ref.K,
        params=ref.params,
        c2w=ref.c2w,
        reproject_source=ref.reproject_source,
        max_incidence_deg=ref.max_incidence_deg,
    )


def _load_image(path: Path) -> tuple[UInt8[np.ndarray, "H W 3"], UInt8[np.ndarray, "H W"] | None]:
    """Load one capture image: RGBA -> (rgb, alpha-as-mask), RGB -> (rgb, None). Uint8, no I/O caching."""
    with Image.open(path) as pil_img:
        converted = pil_img.convert("RGBA") if "A" in pil_img.getbands() else pil_img.convert("RGB")
        arr = np.array(converted)
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3]
    return arr, None
