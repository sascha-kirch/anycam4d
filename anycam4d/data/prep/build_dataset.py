"""Build an AnyCam4D dataset from a raw multi-camera video source via ffmpeg.

This is the fast path for the raw video datasets (DeepView / DyNeRF / SelfCap). It does zero pixel
work in Python: ffmpeg decodes, temporally decimates, and encodes each camera's frames directly
into ``out_dir/images`` (in parallel), while this driver only forwards calibration and per-capture
poses to the writer. Post-hoc pixel variants (resize / undistort) are a separate path over
the emitted dataset (``build_variant``), which does stream pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from anycam4d.data.prep.ffmpeg import extract_all
from anycam4d.data.prep.writer import PrepWriter


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from jaxtyping import Float

    from anycam4d.data.prep.stream import CameraSpec
    from anycam4d.data.prep.stream import CaptureSpec
    from anycam4d.data.prep.stream import VideoImporterProto


def build_dataset(
    importer: VideoImporterProto,
    out_dir: Path,
    *,
    eval_cams: Iterable[int] = (),
    image_format: str = "jpg",
    quality: int = 2,
    workers: int | None = None,
    manifest: dict | None = None,
    recenter: bool = True,
) -> None:
    """Build an AnyCam4D dataset from a video source into ``out_dir``.

    Order of operations: ffmpeg frame extraction runs **first**, then calibration (any COLMAP SfM
    happens here, on the already-extracted footage), then the world is recentered so the first
    camera sits at the origin. Recentering is a pure translation of camera centers and init points,
    so relative geometry is unchanged — it only removes the arbitrary, far-from-origin frame that
    SfM / provided calibration can carry.

    Args:
        importer: The prepared-on-demand video source (DeepView / DyNeRF / SelfCap).
        out_dir: Dataset root to write (images + train/eval JSON + ply + manifest).
        eval_cams: Camera ids routed to the eval split.
        image_format: On-disk image format ffmpeg writes (``"jpg"`` default, or ``"png"``).
        quality: ffmpeg ``-q:v`` for jpg (1=best .. 31=worst); ignored for png.
        workers: Parallel ffmpeg processes (default: min(#cameras, cpu_count // 2) — each process
            holds its own decode buffers, so this is capped below cpu_count to bound peak memory,
            not just CPU oversubscription; lower it further on memory-constrained machines).
        manifest: Optional provenance manifest sidecar.
        recenter: Translate the world so the lowest-id camera is at the origin (default on).
    """
    importer.prepare()

    # Extract frames first; calibration (incl. any SfM) then runs on the extracted footage.
    extract_all(
        importer.extraction_jobs(), out_dir / "images", quality=quality, image_format=image_format, workers=workers
    )

    cameras = importer.cameras()
    captures = importer.captures()
    points = importer.point_cloud()
    xyz, rgb = points if points is not None else (None, None)

    origin = _first_camera_origin(cameras, captures) if recenter else np.zeros(3, dtype=np.float64)

    writer = PrepWriter(out_dir, eval_cams=eval_cams, image_format=image_format)
    for cam in cameras:
        writer.register_camera(cam.cam_id, cam.model, cam.params, cam.width, cam.height)
    for cap in captures:
        c2w = cap.c2w.copy()
        c2w[:3, 3] -= origin.astype(c2w.dtype)
        image_path = f"images/cam{cap.cam_id:02d}_{cap.frame_id:04d}.{image_format}"
        writer.add_capture(cap.cam_id, cap.frame_id, cap.timestamp, c2w, image_path)
    if xyz is not None:
        xyz = xyz - origin.astype(xyz.dtype)
    writer.finalize(xyz, rgb, manifest=manifest)


def _first_camera_origin(cameras: list[CameraSpec], captures: list[CaptureSpec]) -> Float[np.ndarray, "3"]:
    """World-space center of the lowest-id camera (its c2w translation); zeros if none found."""
    if not cameras:
        return np.zeros(3, dtype=np.float64)
    first_cam = min(cam.cam_id for cam in cameras)
    for cap in captures:
        if cap.cam_id == first_cam:
            return cap.c2w[:3, 3].astype(np.float64).copy()
    return np.zeros(3, dtype=np.float64)
