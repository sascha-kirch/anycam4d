"""``DeepViewImporter`` on synthetic fixtures; the real Welder import is ``slow``."""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest

import cv2

from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.importers import deepview as deepview_mod
from anycam4d.data.prep.importers.deepview import DeepViewImporter
from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.enums import CameraModelType


_FRAME_COLORS_RGB = [(230, 10, 10), (10, 230, 10), (10, 10, 230), (230, 230, 10)]


def _write_video(path: Path, n_frames: int, size: tuple[int, int] = (64, 48)) -> None:
    """Write a solid-color-per-frame mp4 (frame f uses ``_FRAME_COLORS_RGB[f]``)."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for f in range(n_frames):
        r, g, b = _FRAME_COLORS_RGB[f]
        bgr = np.empty((h, w, 3), dtype=np.uint8)
        bgr[..., 0], bgr[..., 1], bgr[..., 2] = b, g, r
        writer.write(bgr)
    writer.release()


def _model_entry(name: str, position: list[float], orientation: list[float]) -> dict:
    """A models.json entry in the verified DeepView schema (64x48, KB distortion)."""
    return {
        "name": name,
        "width": 64.0,
        "height": 48.0,
        "focal_length": 40.0,
        "pixel_aspect_ratio": 1.0,
        "principal_point": [32.0, 24.0],
        "radial_distortion": [0.05, -0.01, 0.0],
        "position": position,
        "orientation": orientation,
        "projection_type": "fisheye",
    }


def _make_scene(source: Path, entries: list[dict], n_frames: int, *, uncalibrated_video: str | None = None) -> None:
    """Write models.json + a video per calibrated entry (+ an optional uncalibrated extra video)."""
    source.mkdir(parents=True, exist_ok=True)
    with open(source / "models.json", "w") as f:
        json.dump(entries, f)
    for entry in entries:
        _write_video(source / f"{entry['name']}.mp4", n_frames)
    if uncalibrated_video is not None:
        _write_video(source / f"{uncalibrated_video}.mp4", n_frames)


def _fake_cloud(*_a, **_k):
    """Stand-in for triangulate_with_known_poses: (sparse_dir, xyz, rgb)."""
    xyz = np.zeros((7, 3), dtype=np.float32)
    rgb = np.zeros((7, 3), dtype=np.uint8)
    return Path("unused"), xyz, rgb


def test_models_json_parse_to_fisheye_opencv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """models.json -> fisheye_opencv params [fx=fy=focal, cx, cy, k1, k2, 0, 0]; dataset loads."""
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset

    entries = [
        _model_entry("camera_0002", [0.0, 0.0, 0.4], [0.01, -0.02, 0.03]),
        _model_entry("camera_0005", [0.3, 0.0, 0.4], [0.0, 0.1, 0.0]),
    ]
    _make_scene(tmp_path, entries, n_frames=2)
    importer = DeepViewImporter(tmp_path, tmp_path / "work", colmap=ColmapOptions(dense=False))
    for cam in importer.cameras():
        assert cam.model is CameraModelType.FISHEYE_OPENCV
        assert cam.params == [40.0, 40.0, 32.0, 24.0, 0.05, -0.01, 0.0, 0.0]
        assert (cam.width, cam.height) == (64, 48)

    monkeypatch.setattr(deepview_mod, "triangulate_with_known_poses", _fake_cloud)
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    ds = MultiViewSequenceDataset(out, split="train")
    assert ds.camera_model is CameraModelType.FISHEYE_OPENCV
    assert ds.num_timesteps == 2


def test_pose_transpose_and_reprojection(tmp_path: Path) -> None:
    """c2w = [Rodrigues(orientation)^T | position]; a point in front projects inside the frame."""
    orientation = [0.1, -0.2, 0.15]
    position = [0.05, -0.03, 0.4]
    entries = [_model_entry("camera_0001", position, orientation)]
    _make_scene(tmp_path, entries, n_frames=1)
    importer = DeepViewImporter(tmp_path, tmp_path / "work")
    importer.prepare()
    calib = importer._calib[1]  # noqa: SLF001

    r_w2c, _ = cv2.Rodrigues(np.array(orientation))
    np.testing.assert_allclose(calib.c2w[:3, :3], r_w2c.T, atol=1e-6)
    np.testing.assert_allclose(calib.c2w[:3, 3], position, atol=1e-6)

    x_world = calib.c2w[:3, :3] @ np.array([0.0, 0.0, 1.0]) + calib.c2w[:3, 3]
    k = np.array([[40.0, 0, 32.0], [0, 40.0, 24.0], [0, 0, 1.0]])
    dist = np.array([0.05, -0.01, 0.0, 0.0])
    tvec = -r_w2c @ np.array(position)  # world-to-camera translation = -R @ center
    uv, _ = cv2.fisheye.projectPoints(x_world.reshape(1, 1, 3), cv2.Rodrigues(r_w2c)[0], tvec, k, dist)
    u, v = uv.ravel()
    assert 0 <= u < 64 and 0 <= v < 48


def test_known_pose_view_is_world_to_camera(tmp_path: Path) -> None:
    """``_known_pose_view`` yields cam_from_world = [Rodrigues(orientation) | -R@position] (w2c)."""
    orientation = [0.1, -0.2, 0.15]
    position = [0.05, -0.03, 0.4]
    _make_scene(tmp_path, [_model_entry("camera_0003", position, orientation)], n_frames=1)
    importer = DeepViewImporter(tmp_path, tmp_path / "work")
    importer.prepare()

    view = importer._known_pose_view(3)  # noqa: SLF001
    r_w2c, _ = cv2.Rodrigues(np.array(orientation))
    np.testing.assert_allclose(view.cam_from_world[:, :3], r_w2c, atol=1e-6)
    np.testing.assert_allclose(view.cam_from_world[:, 3], -r_w2c @ np.array(position), atol=1e-6)
    assert view.model == "OPENCV_FISHEYE"
    assert view.name == "camera_0003.png"


def test_video_discovery_skips_uncalibrated(tmp_path: Path) -> None:
    """Non-contiguous ids parsed; a video with no models.json entry (46/45 case) is skipped."""
    entries = [
        _model_entry("camera_0002", [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]),
        _model_entry("camera_0005", [0.3, 0.0, 0.4], [0.0, 0.0, 0.0]),
        _model_entry("camera_0010", [0.0, 0.3, 0.4], [0.0, 0.0, 0.0]),
    ]
    _make_scene(tmp_path, entries, n_frames=1, uncalibrated_video="camera_0036")
    importer = DeepViewImporter(tmp_path, tmp_path / "work")
    importer.prepare()
    assert importer._cam_ids == [2, 5, 10]  # noqa: SLF001  (36 skipped: no model entry)


def test_model_without_video_skipped(tmp_path: Path) -> None:
    """A calibrated entry with no matching video is skipped (robustness)."""
    entries = [
        _model_entry("camera_0002", [0.0, 0.0, 0.4], [0.0, 0.0, 0.0]),
        _model_entry("camera_0009", [0.3, 0.0, 0.4], [0.0, 0.0, 0.0]),
    ]
    _make_scene(tmp_path, entries, n_frames=1)
    (tmp_path / "camera_0009.mp4").unlink()  # calibrated but no video
    importer = DeepViewImporter(tmp_path, tmp_path / "work")
    importer.prepare()
    assert importer._cam_ids == [2]  # noqa: SLF001


@pytest.fixture(scope="session")
def welder_scene_dir() -> Path:
    """Path to the real Welder scene (``ANYCAM4D_WELDER_DIR``), or skip if absent."""
    path = Path(os.environ.get("ANYCAM4D_WELDER_DIR", "data/deepview/01_Welder"))
    if not path.exists() or not (path / "models.json").exists():
        pytest.skip("DeepView Welder scene not available")
    return path


@pytest.mark.slow
def test_end_to_end_import_builds_loadable_dataset(welder_scene_dir: Path, tmp_path: Path) -> None:
    """Real Welder -> canonical fisheye_opencv dataset with a triangulated cloud that loads (gated)."""
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset

    importer = DeepViewImporter(welder_scene_dir, tmp_path / "work", max_frames=2, colmap=ColmapOptions(dense=False))
    out = tmp_path / "out"
    build_dataset(importer, out, image_format="jpg")

    ds = MultiViewSequenceDataset(out, split="train")
    assert ds.num_timesteps == 2
    assert ds.camera_model is CameraModelType.FISHEYE_OPENCV

    from anycam4d.point_ops.ply import read_ply

    xyz, _ = read_ply(out / "points3d.ply")
    assert xyz.shape[0] > 0  # real triangulation produced a cloud in the provided frame
