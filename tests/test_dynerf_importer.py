"""Tests for ``DyNeRFImporter``: ffmpeg extraction + sparse-id parsing (fast), plus gated end-to-end SfM."""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest

import cv2

from anycam4d.data.prep.colmap import ColmapCamera
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.colmap import ColmapResult
from anycam4d.data.prep.importers import video_rig as video_rig_mod
from anycam4d.data.prep.importers.dynerf import DyNeRFImporter
from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.enums import CameraModelType


# Solid RGB colors per frame so BGR<->RGB order + frame ordering are both checkable after
# the (lossy) mp4 -> jpg round-trip: frame f is dominated by channel f (0=R, 1=G, 2=B).
_FRAME_COLORS_RGB = [(230, 10, 10), (10, 230, 10), (10, 10, 230), (230, 230, 10)]


def _write_video(path: Path, n_frames: int, size: tuple[int, int] = (64, 48), fps: float = 30.0) -> None:
    """Write a solid-color-per-frame mp4 (frame f uses ``_FRAME_COLORS_RGB[f]``)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n_frames):
        r, g, b = _FRAME_COLORS_RGB[f]
        bgr = np.empty((h, w, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = b, g, r
        writer.write(bgr)
    writer.release()


def _fake_sfm_result(cam_ids: list[int]) -> ColmapResult:
    """A stand-in COLMAP result: one opencv camera + identity pose per discovered camera."""
    camera_by_image = {
        f"cam{cid:02d}.png": ColmapCamera(
            model=CameraModelType.OPENCV, width=64, height=48, params=[50.0, 50.0, 32.0, 24.0, 0.0, 0.0, 0.0, 0.0]
        )
        for cid in cam_ids
    }
    c2w_by_image = {f"cam{cid:02d}.png": np.eye(4, dtype=np.float32) for cid in cam_ids}
    return ColmapResult(
        camera_by_image=camera_by_image,
        c2w_by_image=c2w_by_image,
        points_xyz=np.zeros((5, 3), dtype=np.float32),
        points_rgb=np.zeros((5, 3), dtype=np.uint8),
    )


def _dominant_channel(image_path: Path) -> int:
    """Dominant RGB channel of a written image (proves the BGR<->RGB order survived)."""
    bgr = cv2.imread(str(image_path))
    rgb_mean = bgr[..., ::-1].reshape(-1, 3).mean(axis=0)
    return int(np.argmax(rgb_mean))


def test_extraction_count_order_and_bgr_to_rgb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Frames extracted in order, BGR->RGB correct, count = min video length; SfM is stubbed."""
    _write_video(tmp_path / "cam00.mp4", n_frames=3)
    _write_video(tmp_path / "cam01.mp4", n_frames=4)  # longer -> min is 3
    monkeypatch.setattr(video_rig_mod, "run_sfm", lambda *a, **k: _fake_sfm_result([0, 1]))

    importer = DyNeRFImporter(tmp_path, tmp_path / "work", fps_cap=30.0, colmap=ColmapOptions(dense=False))
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    doc = json.loads((out / "train_dataset.json").read_text())
    assert importer._count == 3  # noqa: SLF001  clamped to the shorter video
    assert {c["camera_id"] for c in doc["captures"]} == {0, 1}
    assert len(doc["captures"]) == 6  # 3 frames x 2 cameras
    for cid in (0, 1):
        for frame_id in range(3):
            img = out / f"images/cam{cid:02d}_{frame_id:04d}.jpg"
            assert img.exists()
            assert _dominant_channel(img) == frame_id  # dominant channel identifies frame -> BGR->RGB ok
    # Timestamps: 30fps source, no decimation -> frame_id / 30.
    for cap in doc["captures"]:
        expected = int(Path(cap["image_path"]).stem.split("_")[1]) / 30.0
        assert cap["timestamp"] == pytest.approx(expected)


def test_decimation_halves_a_60fps_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 60fps source capped at 30 keeps every 2nd frame; timestamps use the 30fps effective rate."""
    _write_video(tmp_path / "cam00.mp4", n_frames=4, fps=60.0)
    monkeypatch.setattr(video_rig_mod, "run_sfm", lambda *a, **k: _fake_sfm_result([0]))

    importer = DyNeRFImporter(tmp_path, tmp_path / "work", fps_cap=30.0, colmap=ColmapOptions(dense=False))
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    assert importer._decim_k == 2  # noqa: SLF001
    assert importer._count == 2  # noqa: SLF001  ceil(4 / 2)
    doc = json.loads((out / "train_dataset.json").read_text())
    assert len(doc["captures"]) == 2
    # Kept source frames 0, 2 -> dominant channels R(0), B(2); effective fps 30.
    assert _dominant_channel(out / "images/cam00_0000.jpg") == 0
    assert _dominant_channel(out / "images/cam00_0001.jpg") == 2
    assert [c["timestamp"] for c in doc["captures"]] == pytest.approx([0.0, 1.0 / 30.0])


def test_max_frames_clamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_frames`` caps the output frame count below the video length."""
    _write_video(tmp_path / "cam00.mp4", n_frames=4)
    _write_video(tmp_path / "cam01.mp4", n_frames=4)
    monkeypatch.setattr(video_rig_mod, "run_sfm", lambda *a, **k: _fake_sfm_result([0, 1]))

    importer = DyNeRFImporter(tmp_path, tmp_path / "work", max_frames=2, colmap=ColmapOptions(dense=False))
    importer.prepare()
    assert importer._count == 2  # noqa: SLF001
    assert len(importer.captures()) == 2 * 2  # 2 frames x 2 cameras


def test_dense_failure_falls_back_to_sparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When dense MVS raises, the importer keeps the sparse cloud instead of failing the import."""
    _write_video(tmp_path / "cam00.mp4", n_frames=2)
    monkeypatch.setattr(video_rig_mod, "run_sfm", lambda *a, **k: _fake_sfm_result([0]))

    def _boom(*_a, **_k):
        raise RuntimeError("simulated out-of-VRAM")

    monkeypatch.setattr(video_rig_mod, "run_dense_mvs", _boom)
    importer = DyNeRFImporter(tmp_path, tmp_path / "work", colmap=ColmapOptions(dense=True))
    xyz, _ = importer.point_cloud()
    assert xyz.shape == (5, 3)  # the fake sparse cloud, unchanged by the failed densify


def test_filter_radial_outliers_drops_far_points() -> None:
    """The dense-cloud outlier filter removes far-flung points and keeps the bulk (with colors)."""
    from anycam4d.data.prep.colmap import _filter_radial_outliers

    rng = np.random.default_rng(0)
    bulk = rng.normal(0.0, 1.0, (1000, 3)).astype(np.float32)
    outliers = np.array([[500.0, 0.0, 0.0], [0.0, -800.0, 0.0]], dtype=np.float32)
    xyz = np.vstack([bulk, outliers])
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)

    fxyz, frgb = _filter_radial_outliers(xyz, rgb)
    assert fxyz.shape[0] == 1000  # both far outliers dropped, the whole bulk kept
    assert frgb.shape[0] == 1000


def test_run_dense_mvs_requires_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_dense_mvs refuses cleanly when the pycolmap build lacks CUDA (no spawn, clear error)."""
    from anycam4d.data.prep import colmap as colmap_mod

    monkeypatch.setattr(colmap_mod.pycolmap, "has_cuda", False)
    with pytest.raises(RuntimeError, match="CUDA"):
        colmap_mod.run_dense_mvs(tmp_path / "sparse", tmp_path / "imgs", tmp_path / "work")


def test_sparse_camera_ids_parsed(tmp_path: Path) -> None:
    """Non-contiguous filenames cam00/cam02/cam04 -> ids {0, 2, 4}, no contiguity assumed."""
    for cid in (0, 2, 4):
        (tmp_path / f"cam{cid:02d}.mp4").touch()

    importer = DyNeRFImporter(tmp_path, tmp_path / "work")
    importer._discover_videos()  # noqa: SLF001
    assert importer._cam_ids == [0, 2, 4]  # noqa: SLF001


def test_no_videos_raises(tmp_path: Path) -> None:
    """A source dir with no cam*.mp4 fails loudly."""
    importer = DyNeRFImporter(tmp_path, tmp_path / "work")
    with pytest.raises(FileNotFoundError, match="No camNN.mp4"):
        importer._discover_videos()  # noqa: SLF001


@pytest.fixture(scope="session")
def dynerf_scene_dir() -> Path:
    """Path to the real DyNeRF coffee_martini scene (``ANYCAM4D_DYNERF_DIR``), or skip if absent."""
    path = Path(os.environ.get("ANYCAM4D_DYNERF_DIR", "data/raw/dynerf/coffee_martini"))
    if not path.exists() or not list(path.glob("cam*.mp4")):
        pytest.skip("DyNeRF coffee_martini videos not available")
    return path


@pytest.mark.slow
def test_end_to_end_import_builds_loadable_dataset(dynerf_scene_dir: Path, tmp_path: Path) -> None:
    """Real SfM -> canonical opencv dataset that MultiViewSequenceDataset loads (gated)."""
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset

    importer = DyNeRFImporter(dynerf_scene_dir, tmp_path / "work", max_frames=2)
    out = tmp_path / "out"
    build_dataset(importer, out, eval_cams=(0,), image_format="jpg")

    ds = MultiViewSequenceDataset(out, split="train")
    assert ds.num_timesteps == 2
    assert ds.camera_model is CameraModelType.OPENCV
