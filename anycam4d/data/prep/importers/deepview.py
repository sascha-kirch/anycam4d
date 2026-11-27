"""DeepView (Immersive Light Field Video) importer: 46 Yi4k fisheye videos -> native ``fisheye_opencv`` dataset.

Uses the authors' calibration (``models.json``) rather than re-estimating fisheye intrinsics with COLMAP; the
initial cloud is triangulated into those fixed poses and densified by MVS, so fisheye SfM never has to register.

``models.json`` conventions:
- ``fx = fy = focal_length`` (``pixel_aspect_ratio`` is not applied); ``principal_point`` is in pixels.
- ``radial_distortion = [k1, k2, 0]`` is Kannala-Brandt with ``k3 = k4 = 0``: ``fisheye_opencv`` params
  ``[fx, fy, cx, cy, k1, k2, 0, 0]`` without conversion.
- World-to-camera is ``[R | -R t]`` with ``R = Rodrigues(orientation)`` and ``t = position``, so
  ``c2w = [R^T | position]``.
- A video without a calibration entry (``camera_0036`` on Welder) is skipped; videos match models by ``name``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from anycam4d.console import notice
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.colmap import KnownPoseView
from anycam4d.data.prep.colmap import run_dense_mvs
from anycam4d.data.prep.colmap import triangulate_with_known_poses
from anycam4d.data.prep.ffmpeg import decimation_factor
from anycam4d.data.prep.ffmpeg import extract_still
from anycam4d.data.prep.ffmpeg import output_count
from anycam4d.data.prep.ffmpeg import probe_fps
from anycam4d.data.prep.ffmpeg import probe_frame_count
from anycam4d.data.prep.importers.exclusion import drop_excluded
from anycam4d.data.prep.stream import CameraSpec
from anycam4d.data.prep.stream import CaptureSpec
from anycam4d.data.prep.stream import VideoExtractJob
from anycam4d.data.prep.stream import VideoImporterProto
from anycam4d.enums import CameraModelType


if TYPE_CHECKING:
    from collections.abc import Iterable

    from jaxtyping import Float
    from jaxtyping import UInt8


class _CameraCalib:
    """Parsed per-camera calibration: name, fisheye_opencv params (8), resolution, position, c2w."""

    def __init__(
        self,
        name: str,
        params: list[float],
        width: int,
        height: int,
        position: Float[np.ndarray, "3"],
        c2w: Float[np.ndarray, "4 4"],
    ) -> None:
        self.name = name
        self.cam_id = int(name.rsplit("_", 1)[-1])
        self.params = params
        self.width = width
        self.height = height
        self.position = position
        self.c2w = c2w


class DeepViewImporter(VideoImporterProto):
    """Turns a DeepView scene (``models.json`` + ``camera_NNNN.mp4``) into ``fisheye_opencv`` views."""

    def __init__(
        self,
        source_dir: Path,
        work_dir: Path,
        *,
        fps_cap: float = 30.0,
        exclude_cams: Iterable[int] = (),
        max_frames: int | None = None,
        colmap: ColmapOptions | None = None,
    ) -> None:
        """Configure the importer; calibration parse + plan run lazily on first ``prepare()``.

        Args:
            source_dir: Scene directory (holds ``models.json`` + ``camera_NNNN.mp4``).
            work_dir: Scratch dir for extracted frame-0 stills + the COLMAP model.
            fps_cap: Maximum output frame rate; the source is temporally decimated to at most this
                (a 60fps rig capped at 30 keeps every 2nd frame). Timestamps use the effective rate.
            exclude_cams: Camera ids dropped at discovery — excluded before triangulation/MVS see
                them, so they contribute neither points nor frames.
            max_frames: Optional cap on output frames (keeps tests / probes cheap).
            colmap: Triangulation + dense-MVS tunables (see :class:`ColmapOptions`); defaults if omitted.
        """
        self._source_dir = Path(source_dir)
        self._work_dir = Path(work_dir)
        self._fps_cap = fps_cap
        self._exclude_cams = tuple(exclude_cams)
        self._max_frames = max_frames
        self._colmap = colmap if colmap is not None else ColmapOptions()

        self._prepared = False
        self._cam_ids: list[int] = []
        self._video_paths: dict[int, Path] = {}
        self._calib: dict[int, _CameraCalib] = {}
        self._decim_k = 1
        self._effective_fps = fps_cap
        self._count = 0  # number of output timesteps

    def cameras(self) -> list[CameraSpec]:
        """Per-camera fisheye_opencv intrinsics + resolution (static rig, shared over all frames)."""
        self.prepare()
        return [
            CameraSpec(
                cam_id=cid,
                model=CameraModelType.FISHEYE_OPENCV,
                params=list(self._calib[cid].params),
                width=self._calib[cid].width,
                height=self._calib[cid].height,
            )
            for cid in self._cam_ids
        ]

    def captures(self) -> list[CaptureSpec]:
        """One capture per (camera, output timestep); the static calibration is shared per camera."""
        self.prepare()
        return [
            CaptureSpec(cam_id=cid, frame_id=j, timestamp=j / self._effective_fps, c2w=self._calib[cid].c2w)
            for j in range(self._count)
            for cid in self._cam_ids
        ]

    def extraction_jobs(self) -> list[VideoExtractJob]:
        """One ffmpeg job per camera: keep every ``decim_k``-th frame from the start."""
        self.prepare()
        return [
            VideoExtractJob(
                cam_id=cid, video_path=self._video_paths[cid], decim_k=self._decim_k, start=0, count=self._count
            )
            for cid in self._cam_ids
        ]

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None] | None:
        """Init cloud in the provided world frame: known-pose triangulation, densified when possible."""
        self.prepare()
        return self._build_cloud()

    def prepare(self) -> None:
        """Parse ``models.json``, match videos (skipping uncalibrated ones), compute the frame plan."""
        if self._prepared:
            return
        self._parse_models()
        self._plan_frames()
        self._prepared = True

    def _plan_frames(self) -> None:
        """Probe source fps + length, derive the decimation factor, effective fps, and output count."""
        src_fps = probe_fps(self._video_paths[self._cam_ids[0]])
        self._decim_k = decimation_factor(src_fps, self._fps_cap)
        self._effective_fps = (src_fps / self._decim_k) if src_fps > 0 else self._fps_cap
        n_src = min(probe_frame_count(self._video_paths[cid]) for cid in self._cam_ids)
        self._count = output_count(n_src, self._decim_k, self._max_frames)

    def _parse_models(self) -> None:
        """Read ``models.json`` into per-camera ``fisheye_opencv`` calibration, matched to videos."""
        with (self._source_dir / "models.json").open() as f:
            entries = json.load(f)
        for entry in entries:
            name = entry["name"]
            video = self._source_dir / f"{name}.mp4"
            if not video.exists():
                continue  # calibrated camera with no video — skip
            calib = _build_calib(entry)
            self._calib[calib.cam_id] = calib
            self._video_paths[calib.cam_id] = video
        if not self._calib:
            raise FileNotFoundError(f"No calibrated camera videos found under {self._source_dir}")
        self._cam_ids = drop_excluded(self._calib, self._exclude_cams, source=str(self._source_dir))

    def _build_cloud(self) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None]:
        """Triangulate a cloud into the provided poses, then densify with MVS (fallback to sparse)."""
        frame0_dir = self._work_dir / "frame0"
        frame0_dir.mkdir(parents=True, exist_ok=True)
        for cid in self._cam_ids:
            extract_still(self._video_paths[cid], 0, frame0_dir / f"{self._calib[cid].name}.png")

        views = [self._known_pose_view(cid) for cid in self._cam_ids]
        sparse_dir, xyz, rgb = triangulate_with_known_poses(
            frame0_dir,
            self._work_dir / "colmap",
            views,
            camera_model="OPENCV_FISHEYE",
            gpu=self._colmap.gpu,
            num_threads=self._colmap.threads,
            max_image_size=self._colmap.max_image_size,
        )
        if not self._colmap.dense:
            return xyz, rgb
        try:
            dxyz, drgb = run_dense_mvs(
                sparse_dir,
                frame0_dir,
                self._work_dir / "colmap",
                gpu=self._colmap.gpu,
                max_image_size=self._colmap.dense_max_image_size,
                num_threads=self._colmap.threads,
            )
        except RuntimeError as err:
            notice(
                f"[yellow]Dense MVS unavailable — keeping the sparse triangulated cloud ({xyz.shape[0]:,} points).[/]\n  {err}"
            )
            return xyz, rgb
        notice(f"[green]Dense MVS cloud: {dxyz.shape[0]:,} points (was {xyz.shape[0]:,} sparse).[/]")
        return dxyz, drgb

    def _known_pose_view(self, cam_id: int) -> KnownPoseView:
        """Build a fixed-pose triangulation view from a camera's provided fisheye calibration."""
        calib = self._calib[cam_id]
        r_w2c = calib.c2w[:3, :3].T  # c2w rotation is R_w2c^T, so transpose back
        t_w2c = -r_w2c @ calib.position
        cam_from_world = np.hstack([r_w2c, t_w2c.reshape(3, 1)]).astype(np.float64)
        return KnownPoseView(
            name=f"{calib.name}.png",
            model="OPENCV_FISHEYE",
            params=list(calib.params),
            width=calib.width,
            height=calib.height,
            cam_from_world=cam_from_world,
        )


def _build_calib(entry: dict) -> _CameraCalib:
    """Assemble one camera's ``fisheye_opencv`` params + OpenCV c2w from a ``models.json`` entry."""
    fx = fy = float(entry["focal_length"])  # pixel_aspect_ratio is informational, not applied
    cx, cy = float(entry["principal_point"][0]), float(entry["principal_point"][1])
    k1, k2 = float(entry["radial_distortion"][0]), float(entry["radial_distortion"][1])
    params = [fx, fy, cx, cy, k1, k2, 0.0, 0.0]  # Kannala-Brandt [k1, k2, k3=0, k4=0]

    position = np.array(entry["position"], dtype=np.float64)
    r_w2c, _ = cv2.Rodrigues(np.array(entry["orientation"], dtype=np.float64))
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r_w2c.T  # transpose: orientation is world-to-camera
    c2w[:3, 3] = position  # position is the camera center in world
    return _CameraCalib(
        name=entry["name"],
        params=params,
        width=int(entry["width"]),
        height=int(entry["height"]),
        position=position,
        c2w=c2w,
    )
