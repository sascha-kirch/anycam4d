"""Static rig of synchronized videos -> canonical AnyCam4D dataset, calibrated by COLMAP.

Each video is one fixed camera filming the same dynamic scene. The videos must already be synchronized (source frame
``i`` is the same instant in every video) and share one frame rate. Because the rig does not move, calibration is
solved once by COLMAP SfM on the multi-view still at ``calib_frame`` (intrinsics in the chosen lens model,
extrinsics and a sparse cloud, optionally densified by MVS) and reused for every frame.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from anycam4d.console import notice
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.colmap import run_dense_mvs
from anycam4d.data.prep.colmap import run_sfm
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

    import numpy as np
    from jaxtyping import Float
    from jaxtyping import UInt8

    from anycam4d.data.prep.colmap import ColmapCamera


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

#: Lens model of the dataset -> the COLMAP model SfM estimates.
COLMAP_MODEL = {
    CameraModelType.PINHOLE: "PINHOLE",
    CameraModelType.OPENCV: "OPENCV",
    CameraModelType.FISHEYE_OPENCV: "OPENCV_FISHEYE",
}


def discover_videos(video_dir: Path) -> dict[int, Path]:
    """Number the videos in ``video_dir`` 0..N-1 in file-name order.

    Raises:
        FileNotFoundError: If the directory holds no video.
    """
    paths = sorted(p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No videos ({', '.join(VIDEO_SUFFIXES)}) found in {video_dir}")
    return dict(enumerate(paths))


class VideoRigImporter(VideoImporterProto):
    """Turns a directory of synchronized videos into a canonical dataset."""

    def __init__(
        self,
        source_dir: Path,
        work_dir: Path,
        *,
        camera_model: CameraModelType = CameraModelType.OPENCV,
        shared_intrinsics: bool = False,
        calib_frame: int | None = None,
        start_frame: int = 0,
        fps_cap: float = 30.0,
        exclude_cams: Iterable[int] = (),
        max_frames: int | None = None,
        colmap: ColmapOptions | None = None,
    ) -> None:
        """Configure the importer; discovery and frame planning run on ``prepare()``, SfM after extraction.

        Args:
            source_dir: Directory holding one video per camera.
            work_dir: Scratch dir for the calibration stills and the COLMAP model.
            camera_model: Lens model COLMAP estimates and the dataset stores.
            shared_intrinsics: One set of intrinsics for all cameras (identical cameras and settings).
            calib_frame: Source frame COLMAP calibrates on; defaults to ``start_frame``. Pick one with texture
                and little motion blur.
            start_frame: First source frame of the dataset.
            fps_cap: Maximum output frame rate; the source is temporally decimated to at most this.
            exclude_cams: Camera ids dropped at discovery, before COLMAP or extraction sees them.
            max_frames: Optional cap on the number of output frames.
            colmap: Sparse-SfM + dense-MVS options; defaults if omitted.

        Raises:
            ValueError: If ``camera_model`` has no COLMAP counterpart.
        """
        if camera_model not in COLMAP_MODEL:
            raise ValueError(f"camera_model must be one of {[m.value for m in COLMAP_MODEL]}, got {camera_model}")
        self._source_dir = Path(source_dir)
        self._work_dir = Path(work_dir)
        self._camera_model = camera_model
        self._shared_intrinsics = shared_intrinsics
        self._start_frame = start_frame
        self._calib_frame = start_frame if calib_frame is None else calib_frame
        self._fps_cap = fps_cap
        self._exclude_cams = tuple(exclude_cams)
        self._max_frames = max_frames
        self._colmap = colmap if colmap is not None else ColmapOptions()

        self._prepared = False
        self._calibrated = False
        self._cam_ids: list[int] = []
        self._video_paths: dict[int, Path] = {}
        self._cameras: dict[int, ColmapCamera] = {}
        self._c2w: dict[int, Float[np.ndarray, "4 4"]] = {}
        self._points: tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"]] | None = None
        self._decim_k = 1
        self._effective_fps = fps_cap
        self._count = 0

    @property
    def video_paths(self) -> dict[int, Path]:
        """Camera id -> source video, after discovery and exclusion."""
        self.prepare()
        return {cid: self._video_paths[cid] for cid in self._cam_ids}

    def cameras(self) -> list[CameraSpec]:
        """Per-camera intrinsics + resolution recovered by COLMAP (static rig)."""
        self._ensure_calibrated()
        return [
            CameraSpec(
                cam_id=cid,
                model=self._cameras[cid].model,
                params=list(self._cameras[cid].params),
                width=self._cameras[cid].width,
                height=self._cameras[cid].height,
            )
            for cid in self._cam_ids
        ]

    def captures(self) -> list[CaptureSpec]:
        """One capture per (camera, output timestep); the static calibration is shared per camera."""
        self._ensure_calibrated()
        return [
            CaptureSpec(cam_id=cid, frame_id=j, timestamp=j / self._effective_fps, c2w=self._c2w[cid])
            for j in range(self._count)
            for cid in self._cam_ids
        ]

    def extraction_jobs(self) -> list[VideoExtractJob]:
        """One ffmpeg job per camera: every ``decim_k``-th frame from ``start_frame``."""
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

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"]] | None:
        """The COLMAP cloud ``(xyz, rgb)`` of the calibration frame."""
        self._ensure_calibrated()
        return self._points

    def prepare(self) -> None:
        """Discover cameras and plan frames; calibration is deferred so extraction runs first."""
        if self._prepared:
            return
        self._discover_videos()
        self._plan_frames()
        self._prepared = True

    def _discover_videos(self) -> None:
        """Bind ``_video_paths`` and the camera ids that survive exclusion."""
        self._video_paths = self._find_videos()
        self._cam_ids = drop_excluded(self._video_paths, self._exclude_cams, source=str(self._source_dir))

    def _find_videos(self) -> dict[int, Path]:
        """Camera id -> video; subclasses with a naming convention override this."""
        return discover_videos(self._source_dir)

    def _ensure_calibrated(self) -> None:
        """Run COLMAP once (after ``prepare``), binding the static calibration and cloud."""
        self.prepare()
        if self._calibrated:
            return
        self._run_calibration()
        self._calibrated = True

    def _plan_frames(self) -> None:
        """Check the rig shares one frame rate, then derive decimation, effective fps and output count.

        Raises:
            ValueError: If the videos disagree on frame rate or ``start_frame``/``calib_frame`` lie past the end.
        """
        rates = {cid: probe_fps(self._video_paths[cid]) for cid in self._cam_ids}
        src_fps = rates[self._cam_ids[0]]
        if any(not math.isclose(r, src_fps, rel_tol=1e-3) for r in rates.values()):
            raise ValueError(f"videos must share one frame rate (synchronized rig), got {rates}")
        n_src = min(probe_frame_count(self._video_paths[cid]) for cid in self._cam_ids)
        if not 0 <= self._start_frame < n_src or not 0 <= self._calib_frame < n_src:
            raise ValueError(
                f"start_frame={self._start_frame} and calib_frame={self._calib_frame} must lie in [0, {n_src})"
            )
        self._decim_k = decimation_factor(src_fps, self._fps_cap)
        self._effective_fps = (src_fps / self._decim_k) if src_fps > 0 else self._fps_cap
        self._count = output_count(n_src - self._start_frame, self._decim_k, self._max_frames)

    def _run_calibration(self) -> None:
        """Extract the calibration still of every camera, run COLMAP, and require every camera to register.

        Raises:
            RuntimeError: If COLMAP leaves a camera unregistered.
        """
        still_dir = self._work_dir / "calib_frame"
        shutil.rmtree(still_dir, ignore_errors=True)  # an excluded camera's old still must not reach COLMAP
        still_dir.mkdir(parents=True)
        for cid in self._cam_ids:
            extract_still(self._video_paths[cid], self._calib_frame, still_dir / _still_name(cid))

        colmap_dir = self._work_dir / "colmap"
        camera_mode = "SINGLE" if self._shared_intrinsics else "PER_IMAGE"
        inputs = {
            "camera_model": self._camera_model.value,
            "camera_mode": camera_mode,
            "calib_frame": self._calib_frame,
            "videos": {str(cid): self._video_paths[cid].name for cid in self._cam_ids},
        }
        _reset_if_inputs_changed(colmap_dir, inputs)
        result = run_sfm(
            still_dir,
            colmap_dir,
            camera_model=COLMAP_MODEL[self._camera_model],
            camera_mode=camera_mode,
            gpu=self._colmap.gpu,
            num_threads=self._colmap.threads,
            max_image_size=self._colmap.max_image_size,
        )

        missing = [cid for cid in self._cam_ids if _still_name(cid) not in result.camera_by_image]
        if missing:
            raise RuntimeError(
                f"COLMAP failed to register camera(s) {missing} on frame {self._calib_frame}; refusing to emit a "
                "partial dataset. Try another --calib-frame, or exclude those cameras."
            )
        for cid in self._cam_ids:
            name = _still_name(cid)
            self._cameras[cid] = result.camera_by_image[name]
            self._c2w[cid] = result.c2w_by_image[name]
        self._points = (result.points_xyz, result.points_rgb)
        if self._colmap.dense:
            self._points = self._densify(still_dir)

    def _densify(self, image_dir: Path) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None]:
        """Replace the sparse cloud with a dense MVS cloud, or keep sparse on failure (VRAM etc.)."""
        try:
            xyz, rgb = run_dense_mvs(
                self._work_dir / "colmap" / "sparse" / "0",
                image_dir,
                self._work_dir / "colmap",
                gpu=self._colmap.gpu,
                max_image_size=self._colmap.dense_max_image_size,
                num_threads=self._colmap.threads,
            )
        except RuntimeError as err:
            notice(f"[yellow]Dense MVS unavailable — keeping the sparse init cloud.[/]\n  {err}")
            return self._points
        notice(f"[green]Dense MVS cloud: {xyz.shape[0]:,} points (was {self._points[0].shape[0]:,} sparse).[/]")
        return xyz, rgb


def _reset_if_inputs_changed(colmap_dir: Path, inputs: dict) -> None:
    """Delete a COLMAP model made from other inputs, so a rerun with new settings recalibrates."""
    marker = colmap_dir / "inputs.json"
    if colmap_dir.exists() and (not marker.exists() or json.loads(marker.read_text()) != inputs):
        notice(f"[yellow]Calibration settings changed — discarding the COLMAP model in {colmap_dir}.[/]")
        shutil.rmtree(colmap_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(inputs, indent=2) + "\n")


def _still_name(cam_id: int) -> str:
    """File name of a camera's calibration still (also its COLMAP image name)."""
    return f"cam{cam_id:02d}.png"
