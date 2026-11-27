"""``VideoRigImporter``: discovery, frame planning and what reaches COLMAP; SfM is stubbed."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.data.prep.colmap import ColmapCamera
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.colmap import ColmapResult
from anycam4d.data.prep.importers import video_rig as video_rig_mod
from anycam4d.data.prep.importers.video_rig import VideoRigImporter
from anycam4d.data.prep.importers.video_rig import discover_videos
from anycam4d.enums import CameraModelType


# Frame f is dominated by channel f % 3 (R, G, B), so frame order survives the lossy round trip.
_COLORS_RGB = [(230, 10, 10), (10, 230, 10), (10, 10, 230)]


def _write_video(path: Path, n_frames: int, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for f in range(n_frames):
        r, g, b = _COLORS_RGB[f % 3]
        writer.write(np.full((48, 64, 3), (b, g, r), dtype=np.uint8))
    writer.release()


def _dominant_channel(path: Path) -> int:
    return int(np.argmax(cv2.imread(str(path))[..., ::-1].reshape(-1, 3).mean(axis=0)))


def _stub_sfm(monkeypatch: pytest.MonkeyPatch, n_cams: int) -> dict:
    """Replace ``run_sfm`` with a fake rig; returns the dict the call's arguments land in."""
    seen: dict = {}

    def fake(image_dir: Path, _work_dir: Path, **kwargs: object) -> ColmapResult:
        seen.update(kwargs, image_dir=image_dir)
        names = [f"cam{cid:02d}.png" for cid in range(n_cams)]
        camera = ColmapCamera(
            model=CameraModelType.OPENCV, width=64, height=48, params=[50.0, 50.0, 32.0, 24.0] + [0.0] * 4
        )
        return ColmapResult(
            camera_by_image=dict.fromkeys(names, camera),
            c2w_by_image={n: np.eye(4, dtype=np.float32) for n in names},
            points_xyz=np.zeros((5, 3), dtype=np.float32),
            points_rgb=np.zeros((5, 3), dtype=np.uint8),
        )

    monkeypatch.setattr(video_rig_mod, "run_sfm", fake)
    return seen


def test_videos_are_numbered_in_file_name_order(tmp_path: Path) -> None:
    for name in ("b_side.MOV", "a_front.mp4", "notes.txt", "c_top.mkv"):
        (tmp_path / name).touch()
    assert {cid: p.name for cid, p in discover_videos(tmp_path).items()} == {
        0: "a_front.mp4",
        1: "b_side.MOV",
        2: "c_top.mkv",
    }


def test_a_directory_without_videos_fails(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").touch()
    with pytest.raises(FileNotFoundError, match="No videos"):
        discover_videos(tmp_path)


def test_start_frame_shifts_extraction_and_calibration_reads_its_own_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output frame j is source frame start + j with timestamps from 0; COLMAP sees ``calib_frame``."""
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("left.mp4", "right.mp4"):
        _write_video(videos / name, n_frames=5)
    seen = _stub_sfm(monkeypatch, n_cams=2)

    importer = VideoRigImporter(
        videos, tmp_path / "work", start_frame=1, calib_frame=2, colmap=ColmapOptions(dense=False)
    )
    out = tmp_path / "out"
    build_dataset(importer, out, eval_cams=(1,))

    train = json.loads((out / "train_dataset.json").read_text())
    assert sorted({c["frame_id"] for c in train["captures"]}) == [0, 1, 2, 3]
    assert [_dominant_channel(out / f"images/cam00_{j:04d}.jpg") for j in range(4)] == [1, 2, 0, 1]
    assert train["captures"][0]["timestamp"] == 0.0
    assert _dominant_channel(seen["image_dir"] / "cam00.png") == 2
    assert (out / "eval_dataset.json").exists()


@pytest.mark.parametrize(
    ("model", "shared", "colmap_model", "mode"),
    [
        (CameraModelType.OPENCV, False, "OPENCV", "PER_IMAGE"),
        (CameraModelType.FISHEYE_OPENCV, True, "OPENCV_FISHEYE", "SINGLE"),
        (CameraModelType.PINHOLE, False, "PINHOLE", "PER_IMAGE"),
    ],
)
def test_lens_model_and_shared_intrinsics_reach_colmap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: CameraModelType, shared: bool, colmap_model: str, mode: str
) -> None:
    _write_video(tmp_path / "cam.mp4", n_frames=2)
    seen = _stub_sfm(monkeypatch, n_cams=1)
    importer = VideoRigImporter(
        tmp_path, tmp_path / "work", camera_model=model, shared_intrinsics=shared, colmap=ColmapOptions(dense=False)
    )
    importer.cameras()
    assert (seen["camera_model"], seen["camera_mode"]) == (colmap_model, mode)


def test_videos_at_different_frame_rates_are_rejected(tmp_path: Path) -> None:
    _write_video(tmp_path / "a.mp4", n_frames=3, fps=30.0)
    _write_video(tmp_path / "b.mp4", n_frames=3, fps=25.0)
    with pytest.raises(ValueError, match="one frame rate"):
        VideoRigImporter(tmp_path, tmp_path / "work").prepare()


def test_a_calibration_frame_past_the_end_is_rejected(tmp_path: Path) -> None:
    _write_video(tmp_path / "a.mp4", n_frames=3)
    with pytest.raises(ValueError, match="calib_frame=7"):
        VideoRigImporter(tmp_path, tmp_path / "work", calib_frame=7).prepare()


def test_a_lens_model_colmap_cannot_estimate_is_rejected(tmp_path: Path) -> None:
    unsupported = next(m for m in CameraModelType if m not in video_rig_mod.COLMAP_MODEL)
    with pytest.raises(ValueError, match="camera_model"):
        VideoRigImporter(tmp_path, tmp_path / "work", camera_model=unsupported)


@pytest.mark.parametrize("change", [{"camera_model": CameraModelType.FISHEYE_OPENCV}, {"calib_frame": 1}, {}])
def test_a_rerun_with_other_calibration_settings_discards_the_old_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict
) -> None:
    """``run_sfm`` reuses ``sparse/0``; a model made from other settings must be gone before it looks."""
    _write_video(tmp_path / "cam.mp4", n_frames=3)
    _stub_sfm(monkeypatch, n_cams=1)
    stub = video_rig_mod.run_sfm
    found_old_model = []

    def fake(image_dir: Path, work_dir: Path, **kwargs: object) -> ColmapResult:
        found_old_model.append((work_dir / "sparse" / "0").exists())
        (work_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        return stub(image_dir, work_dir, **kwargs)

    monkeypatch.setattr(video_rig_mod, "run_sfm", fake)
    for settings in ({}, change):
        VideoRigImporter(tmp_path, tmp_path / "work", colmap=ColmapOptions(dense=False), **settings).cameras()
    assert found_old_model == [False, not change]
