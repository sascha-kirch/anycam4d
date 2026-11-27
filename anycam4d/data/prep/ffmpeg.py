"""ffprobe metadata and one-pass ffmpeg extraction of final dataset images, one process per camera.

Indices are source frames: output frame ``j`` is source frame ``start + j·decim_k``.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import TYPE_CHECKING

from anycam4d.console import notice
from anycam4d.console import progress_bar


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from anycam4d.data.prep.stream import VideoExtractJob


def probe_fps(video: Path) -> float:
    """Average frame rate of ``video`` in Hz, or 0.0 if ffprobe cannot tell."""
    out = _ffprobe(video, "avg_frame_rate")
    num, _, den = out.partition("/") if "/" in out else (out, "", "1")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return 0.0
    return n / d if d else 0.0


def probe_frame_count(video: Path) -> int:
    """Number of frames in ``video`` (``nb_frames``, or a duration*fps estimate as fallback)."""
    out = _ffprobe(video, "nb_frames")
    try:
        return int(out)
    except ValueError:
        pass
    # Some containers omit nb_frames; estimate from duration * fps.
    duration = _ffprobe(video, "duration")
    try:
        return round(float(duration) * probe_fps(video))
    except ValueError:
        return 0


def probe_dimensions(video: Path) -> tuple[int, int]:
    """``(width, height)`` of ``video`` in pixels."""
    w = _ffprobe(video, "width")
    h = _ffprobe(video, "height")
    return int(w), int(h)


def extract_still(video: Path, frame_index: int, out_path: Path) -> None:
    """Write source frame ``frame_index`` of ``video`` to ``out_path`` (lossless — for COLMAP)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    select = f"select='eq(n\\,{frame_index})'"
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vf", select, "-frames:v", "1", str(out_path),
    ]  # fmt: skip
    _run(cmd)


def extract_camera(job: VideoExtractJob, out_dir: Path, *, quality: int, image_format: str) -> None:
    """Decode, decimate and write one camera's frames as ``camNN_%04d.<ext>`` (JPEG ``-q:v`` 1 best to 31)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / f"cam{job.cam_id:02d}_%04d.{image_format}"
    # Unbounded (-threads 0 / auto) HEVC decode threading multiplies both CPU and per-process
    # decode-buffer memory by core count; with N cameras running in parallel that's N x cores
    # threads and can OOM a desktop machine on 4K sources. Cap threads per process instead.
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-threads", "2", "-i", str(job.video_path),
    ]  # fmt: skip
    if job.decim_k > 1 or job.start > 0:
        select = f"select='gte(n\\,{job.start})*not(mod(n-{job.start}\\,{job.decim_k}))'"
        cmd += ["-vf", select]
    cmd += ["-fps_mode", "passthrough", "-start_number", "0", "-frames:v", str(job.count)]
    if image_format in {"jpg", "jpeg"}:
        cmd += ["-q:v", str(quality)]
    cmd += [str(out_pattern)]
    _run(cmd)


def _camera_extracted(job: VideoExtractJob, out_dir: Path, image_format: str) -> bool:
    """Whether every expected frame of a camera is on disk; a partial prefix is re-extracted."""
    return all((out_dir / f"cam{job.cam_id:02d}_{j:04d}.{image_format}").exists() for j in range(job.count))


def extract_all(
    jobs: Iterable[VideoExtractJob],
    out_dir: Path,
    *,
    quality: int,
    image_format: str,
    workers: int | None = None,
) -> None:
    """Run ``extract_camera`` for every job in a small pool, skipping finished cameras.

    The pool stays well below the core count: each 4K HEVC decoder holds ~1 GB.
    """
    jobs = list(jobs)
    if not jobs:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not _camera_extracted(job, out_dir, image_format)]
    n_skipped = len(jobs) - len(todo)
    if n_skipped:
        notice(f"[yellow]ffmpeg: {n_skipped}/{len(jobs)} cameras already extracted — skipping.[/]")
    if not todo:
        return
    n_workers = workers if workers is not None else min(len(todo), max(1, (os.cpu_count() or 4) // 2))
    with (
        progress_bar("ffmpeg extract", len(todo), unit="cams") as advance,
        ThreadPoolExecutor(max_workers=n_workers) as pool,
    ):
        futures = [
            pool.submit(extract_camera, job, out_dir, quality=quality, image_format=image_format) for job in todo
        ]
        for future in as_completed(futures):
            future.result()  # re-raise any per-camera failure
            advance()


def decimation_factor(src_fps: float, fps_cap: float) -> int:
    """Keep-every-k factor ``max(1, round(src_fps / fps_cap))``; never upsamples."""
    if src_fps <= 0.0 or fps_cap <= 0.0:
        return 1
    return max(1, round(src_fps / fps_cap))


def output_count(span: int, decim_k: int, max_frames: int | None = None) -> int:
    """Number of output frames when keeping every ``decim_k``-th of ``span`` source frames."""
    n = (max(0, span) + decim_k - 1) // decim_k
    if max_frames is not None:
        n = min(n, max_frames)
    return n


def _ffprobe(video: Path, entry: str) -> str:
    """Return a single stream/format field of ``video`` via ffprobe (empty string on failure)."""
    stream_entry = entry in {"avg_frame_rate", "nb_frames", "width", "height"}
    key = f"stream={entry}" if stream_entry else f"format={entry}"
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", key, "-of", "default=noprint_wrappers=1:nokey=1", str(video),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    return result.stdout.strip()


def _run(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising ``RuntimeError`` with stderr on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}")
