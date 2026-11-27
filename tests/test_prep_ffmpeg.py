"""Tests for the ffmpeg extraction driver's resume skip (no real ffmpeg is run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycam4d.data.prep import ffmpeg
from anycam4d.data.prep.stream import VideoExtractJob


def _job(cam_id: int, count: int = 3) -> VideoExtractJob:
    return VideoExtractJob(cam_id=cam_id, video_path=Path("cam.mp4"), decim_k=1, start=0, count=count)


def test_camera_extracted_true_only_when_complete(tmp_path: Path) -> None:
    job = _job(1, count=3)
    assert not ffmpeg._camera_extracted(job, tmp_path, "jpg")
    for j in range(2):  # a partial prefix does not count as done
        (tmp_path / f"cam01_{j:04d}.jpg").write_text("x")
    assert not ffmpeg._camera_extracted(job, tmp_path, "jpg")
    (tmp_path / "cam01_0002.jpg").write_text("x")
    assert ffmpeg._camera_extracted(job, tmp_path, "jpg")


def test_extract_all_skips_finished_cameras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera already fully on disk is never handed to ffmpeg; the rest are."""
    ran: list[int] = []

    def fake_extract(job, out_dir, *, quality, image_format):  # noqa: ANN001, ANN202 - test spy
        ran.append(job.cam_id)
        for j in range(job.count):
            (out_dir / f"cam{job.cam_id:02d}_{j:04d}.{image_format}").write_text("x")

    monkeypatch.setattr(ffmpeg, "extract_camera", fake_extract)

    jobs = [_job(0), _job(1)]
    # Pre-populate cam 0 so it is considered done.
    for j in range(3):
        (tmp_path / f"cam00_{j:04d}.jpg").write_text("x")

    ffmpeg.extract_all(jobs, tmp_path, quality=2, image_format="jpg")
    assert ran == [1]  # only the unfinished camera decoded

    ran.clear()
    ffmpeg.extract_all(jobs, tmp_path, quality=2, image_format="jpg")
    assert ran == []  # second run: both cameras already extracted -> full skip
