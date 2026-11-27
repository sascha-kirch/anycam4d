"""Protocols for the dataset-prep pipeline: sources, stages, writers, and video importers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    from jaxtyping import Float

    from anycam4d.data.prep.sample import CalibratedView
    from anycam4d.enums import CameraModelType


@dataclass(frozen=True)
class CameraSpec:
    """One camera's intrinsics for the pixel-free native-import path (no image needed).

    ``width``/``height`` come from the source calibration or an ffprobe of the video, since the
    native path never decodes a frame in Python to read its shape.
    """

    cam_id: int
    model: CameraModelType
    params: list[float]  # canonical OpenCV order [fx, fy, cx, cy, <distortion...>]
    width: int
    height: int


@dataclass(frozen=True)
class CaptureSpec:
    """One capture's metadata for the native-import path: pose + timestamp, no pixels.

    ``frame_id`` is the dense **output** timestep index (0-based, shared across cameras after
    temporal decimation), matching the ``camNN_{frame_id:04d}`` file ffmpeg emits.
    """

    cam_id: int
    frame_id: int
    timestamp: float  # seconds, pre-normalization
    c2w: Float[np.ndarray, "4 4"]  # OpenCV camera-to-world


@dataclass(frozen=True)
class VideoExtractJob:
    """One camera's ffmpeg extraction plan: keep every ``decim_k``-th source frame from ``start``.

    ``count`` is the number of **output** frames to emit (the caller pre-clamps it so every camera
    yields the same dense timestep count, preserving frame alignment).
    """

    cam_id: int
    video_path: Path
    decim_k: int
    start: int
    count: int


@runtime_checkable
class StageProto(Protocol):
    """A generic pivot -> pivot transform (resize, undistort, ...)."""

    def __call__(self, view: CalibratedView) -> CalibratedView:
        """Return a transformed copy of ``view``."""
        ...


@runtime_checkable
class SequenceSourceProto(Protocol):
    """Yields complete frame-batches of ``CalibratedView`` (an importer or ``Anycam4DSource``).

    ``stream()`` must yield **frame-aligned** batches: each batch is exactly ``batch_frames``
    complete frames, with every camera present for each frame it contains. Never split a single
    frame across two batches. Later importers and the shared ``build()`` driver depend on this
    invariant holding for any implementation.
    """

    def frame_ids(self) -> list[int]:
        """Return every frame id this source will yield, in stream order."""
        ...

    def camera_ids(self) -> list[int]:
        """Return every camera id present in each frame (the per-frame camera count)."""
        ...

    def stream(self, batch_frames: int = 1) -> Iterator[list[CalibratedView]]:
        """Yield frame-aligned batches of ``batch_frames`` complete frames each."""
        ...

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "N 3"] | None] | None:
        """Return an optional ``(xyz, colors)`` point cloud, or ``None`` if the source has none."""
        ...


@runtime_checkable
class VideoImporterProto(Protocol):
    """A raw multi-camera video source imported straight to disk via ffmpeg (no Python decode).

    Distinct from ``SequenceSourceProto`` (which streams decoded pixels for post-hoc pixel
    transforms): a video importer only supplies calibration + an ffmpeg extraction plan. The
    native-import driver runs the plan (ffmpeg writes the final images) and writes the manifests
    from ``cameras()`` + ``captures()`` — Python touches no pixels on this path.

    ``prepare()`` must run before the three accessors (it parses calibration, probes the videos,
    and computes the temporal-decimation plan); all three are consistent afterwards.
    """

    def prepare(self) -> None:
        """Parse calibration, probe videos, and compute the decimation plan (idempotent)."""
        ...

    def cameras(self) -> list[CameraSpec]:
        """Per-camera intrinsics + resolution for every camera that will be written."""
        ...

    def captures(self) -> list[CaptureSpec]:
        """Every output capture's pose + timestamp (frame-aligned across cameras)."""
        ...

    def extraction_jobs(self) -> list[VideoExtractJob]:
        """Per-camera ffmpeg extraction plans that produce the dataset images."""
        ...

    def point_cloud(self) -> tuple[Float[np.ndarray, "N 3"], Float[np.ndarray, "N 3"] | None] | None:
        """Return an optional ``(xyz, colors)`` init cloud, or ``None`` if the source has none."""
        ...
