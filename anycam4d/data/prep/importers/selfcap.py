"""SelfCap (EasyVolcap) release -> canonical ``opencv`` AnyCam4D dataset, without COLMAP.

- ``intri.yml``: ``K_<name>`` and ``D_<name>`` (``[k1, k2, p1, p2, k3]``); ``extri.yml``: world-to-camera
  ``Rot_<name>``, ``T_<name>``. The camera frame is OpenCV: ``R_c2w = Rotᵀ``, ``C = -Rotᵀ T``.
- The model is always ``opencv``, zero distortion included; pinhole comes from the ``ToPinhole`` variant.
- ``timestamp = output_index / effective_fps``, shared by all cameras and starting at ``start_frame``.
- ``optimized/sync.json`` sub-frame offsets are not applied, so builds match the published datasets.
- ``start_frame``/``end_frame`` are source frames applied before decimation; ``max_frames`` caps output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from anycam4d.data.prep.ffmpeg import decimation_factor
from anycam4d.data.prep.ffmpeg import output_count
from anycam4d.data.prep.ffmpeg import probe_dimensions
from anycam4d.data.prep.ffmpeg import probe_fps
from anycam4d.data.prep.ffmpeg import probe_frame_count
from anycam4d.data.prep.importers.exclusion import drop_excluded
from anycam4d.data.prep.stream import CameraSpec
from anycam4d.data.prep.stream import CaptureSpec
from anycam4d.data.prep.stream import VideoExtractJob
from anycam4d.data.prep.stream import VideoImporterProto
from anycam4d.enums import CameraModelType
from anycam4d.point_ops.ply import read_ply


if TYPE_CHECKING:
    from collections.abc import Iterable

    from jaxtyping import Float
    from jaxtyping import UInt8


class _CameraCalib:
    """Parsed per-camera calibration: name, OpenCV params (9), and c2w."""

    def __init__(self, name: str, params: list[float], c2w: Float[np.ndarray, "4 4"]) -> None:
        self.name = name
        self.cam_id = int(name)
        self.params = params
        self.c2w = c2w


class SelfCapImporter(VideoImporterProto):
    """Turns a SelfCap release (``optimized/*.yml`` + ``videos/*.mp4`` + ``dense_pcds/``) into a dataset."""

    def __init__(
        self,
        source_dir: Path,
        *,
        calib_dir: Path | None = None,
        fps_cap: float = 30.0,
        start_frame: int = 0,
        end_frame: int | None = None,
        exclude_cams: Iterable[int] = (),
        max_frames: int | None = None,
    ) -> None:
        """Configure the importer; calibration and frame planning run in ``prepare()``.

        Args:
            source_dir: Sequence directory with ``videos/``, ``dense_pcds/`` and usually ``optimized/``.
            calib_dir: Directory holding ``optimized/`` when shipped as a separate release.
            fps_cap: Maximum output frame rate.
            start_frame: First source frame to emit.
            end_frame: One past the last source frame; ``None`` for the video end.
            exclude_cams: Camera ids dropped at discovery.
            max_frames: Optional cap on output frames.
        """
        self._source_dir = Path(source_dir)
        self._calib_dir = Path(calib_dir) if calib_dir is not None else self._source_dir
        self._fps_cap = fps_cap
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._exclude_cams = tuple(exclude_cams)
        self._max_frames = max_frames

        self._prepared = False
        self._cam_ids: list[int] = []
        self._video_paths: dict[int, Path] = {}
        self._calib: dict[int, _CameraCalib] = {}
        self._dims: dict[int, tuple[int, int]] = {}
        self._decim_k = 1
        self._effective_fps = fps_cap
        self._count = 0  # number of output timesteps
        self._trim_end = 0  # source frame one-past-last, after clamping to shortest video

    def cameras(self) -> list[CameraSpec]:
        """Per-camera opencv intrinsics + resolution (resolution probed from each video)."""
        self.prepare()
        return [
            CameraSpec(
                cam_id=cid,
                model=CameraModelType.OPENCV,
                params=list(self._calib[cid].params),
                width=self._dims[cid][0],
                height=self._dims[cid][1],
            )
            for cid in self._cam_ids
        ]

    def captures(self) -> list[CaptureSpec]:
        """One capture per (camera, output timestep); every camera shares a timestep's timestamp."""
        self.prepare()
        return [
            CaptureSpec(
                cam_id=cid,
                frame_id=j,
                timestamp=j / self._effective_fps,
                c2w=self._calib[cid].c2w,
            )
            for j in range(self._count)
            for cid in self._cam_ids
        ]

    def extraction_jobs(self) -> list[VideoExtractJob]:
        """One ffmpeg job per camera: keep every ``decim_k``-th frame from ``start_frame``."""
        self.prepare()
        return [
            VideoExtractJob(
                cam_id=cid,
                video_path=self._video_paths[cid],
                decim_k=self._decim_k,
                start=self._start_frame,
                count=self._count,
            )
            for cid in self._cam_ids
        ]

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None] | None:
        """The ``dense_pcds`` cloud matching the trimmed span, or ``None``.

        The per-frame ``pcds/`` series is cropped to the working volume, so the whole-scene dense cloud is used.
        """
        self.prepare()
        pcd_path = self._nearest_dense_pcd()
        if pcd_path is None:
            return None
        return read_ply(pcd_path)

    def prepare(self) -> None:
        """Parse calibration, match videos, and compute the decimation plan."""
        if self._prepared:
            return
        self._parse_calibration()
        self._dims = {cid: probe_dimensions(self._video_paths[cid]) for cid in self._cam_ids}
        self._plan_frames()
        self._prepared = True

    def _plan_frames(self) -> None:
        """Probe fps + length, derive the decimation factor over the trimmed span, and output count."""
        src_fps = probe_fps(self._video_paths[self._cam_ids[0]])
        self._decim_k = decimation_factor(src_fps, self._fps_cap)
        self._effective_fps = (src_fps / self._decim_k) if src_fps > 0 else self._fps_cap
        min_len = min(probe_frame_count(self._video_paths[cid]) for cid in self._cam_ids)
        end = min_len if self._end_frame is None else min(self._end_frame, min_len)
        self._trim_end = end
        span = max(0, end - self._start_frame)
        self._count = output_count(span, self._decim_k, self._max_frames)

    def _parse_calibration(self) -> None:
        """Read ``intri.yml`` + ``extri.yml`` into per-camera ``opencv`` params + c2w; match videos."""
        opt = self._calib_dir / "optimized"
        intri = cv2.FileStorage(str(opt / "intri.yml"), cv2.FILE_STORAGE_READ)
        extri = cv2.FileStorage(str(opt / "extri.yml"), cv2.FILE_STORAGE_READ)
        try:
            # ``.keys()`` is the cv2 FileNode API (not a dict), so SIM118's rewrite doesn't apply.
            names = sorted(key[2:] for key in intri.root().keys() if key.startswith("K_"))  # noqa: SIM118
            if not names:
                raise ValueError(f"No K_<name> intrinsics found in {opt / 'intri.yml'}")
            # Excluded cameras are dropped before their video is even required to exist.
            kept = set(drop_excluded((int(n) for n in names), self._exclude_cams, source=str(self._source_dir)))
            for name in (n for n in names if int(n) in kept):
                video = self._source_dir / "videos" / f"{name}.mp4"
                if not video.exists():
                    raise FileNotFoundError(f"camera {name}: calibration present but {video} missing")
                calib = _build_calib(name, intri, extri)
                self._calib[calib.cam_id] = calib
                self._video_paths[calib.cam_id] = video
        finally:
            intri.release()
            extri.release()
        self._cam_ids = sorted(self._calib)

    def _nearest_dense_pcd(self) -> Path | None:
        """The dense cloud inside ``[start_frame, trim_end)``, else the one nearest the span (ties towards the start)."""
        pcd_dir = self._source_dir / "dense_pcds"
        candidates = sorted(pcd_dir.glob("*.ply")) if pcd_dir.is_dir() else []
        if not candidates:
            return None
        # ``trim_end`` is one-past-last, so the inclusive upper bound is ``trim_end - 1``.
        lo, hi = self._start_frame, max(self._start_frame, self._trim_end - 1)

        def rank(p: Path) -> tuple[int, int]:
            f = int(p.stem)
            dist_to_span = 0 if lo <= f <= hi else min(abs(f - lo), abs(f - hi))
            return (dist_to_span, abs(f - lo))

        return min(candidates, key=rank)


def _build_calib(name: str, intri: cv2.FileStorage, extri: cv2.FileStorage) -> _CameraCalib:
    """Assemble one camera's ``opencv`` params + OpenCV c2w from the two file storages."""
    k = intri.getNode(f"K_{name}").mat()
    dist = intri.getNode(f"D_{name}").mat().ravel()
    rot = extri.getNode(f"Rot_{name}").mat()  # world-to-camera rotation
    t = extri.getNode(f"T_{name}").mat().ravel()

    # OpenCV canonical opencv params: [fx, fy, cx, cy, k1, k2, p1, p2, k3]; EasyVolcap dist order
    # already equals this, so a straight copy (kept 5-wide even when all-zero).
    params = [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), *[float(x) for x in dist[:5]]]

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot.T
    c2w[:3, 3] = -rot.T @ t
    return _CameraCalib(name=name, params=params, c2w=c2w)
