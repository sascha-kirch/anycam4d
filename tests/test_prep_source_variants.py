"""Tests for anycam4d/data/prep/source.py, variants.py, and the pose.py numpy round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.prep.source import Anycam4DSource
from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.writer import PrepWriter
from anycam4d.data.prep.variants import VariantSpec
from anycam4d.data.prep.variants import build_variant
from anycam4d.data.prep.pose import quat_translation_to_c2w
from anycam4d.data.prep.pose import rotation_matrix_to_quat
from anycam4d.data.prep.stages.resize import Resize
from anycam4d.data.prep.stages.to_pinhole import ToPinhole
from anycam4d.enums import CameraModelType
from tests.prep_writing import write_views


_TRAIN_CAM_IDS = (0, 1)
_EVAL_CAM_ID = 2
_FRAME_TIMESTAMPS = (0.0, 1.0 / 30.0)
_HW = 32


def _c2w(
    translation: tuple[float, float, float], euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    c2w[:3, 3] = translation
    return c2w


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (_HW, _HW, 3), dtype=np.uint8)


def _make_view(
    cam_id: int, frame_id: int, timestamp: float, model: CameraModelType, params: list[float]
) -> CalibratedView:
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return CalibratedView(
        cam_id=cam_id,
        frame_id=frame_id,
        timestamp=timestamp,
        image=_make_image(cam_id * 100 + frame_id),
        mask=None,
        model=model,
        K=k,
        params=list(params),
        # Camera behind the origin (z<0) looking down +z at the point cloud, so build_variant's
        # geometry validation (L2) sees the cloud in-frame.
        c2w=_c2w(translation=(cam_id * 0.5, 0.0, -3.0), euler_deg=(cam_id * 5.0, 0.0, 0.0)),
    )


def _build_canonical(
    out_dir: Path,
    model: CameraModelType = CameraModelType.PINHOLE,
    params: list[float] | None = None,
    with_points: bool = True,
) -> Path:
    """2 frames x (2 train cams + 1 eval cam), all sharing one camera model."""
    params = params if params is not None else [50.0, 50.0, 16.0, 16.0]
    writer = PrepWriter(out_dir, eval_cams={_EVAL_CAM_ID})
    for frame_id, ts in enumerate(_FRAME_TIMESTAMPS):
        views = [_make_view(cam_id, frame_id, ts, model, params) for cam_id in (*_TRAIN_CAM_IDS, _EVAL_CAM_ID)]
        write_views(writer, views)

    if with_points:
        rng = np.random.default_rng(0)
        points = rng.uniform(-1.0, 1.0, (5, 3)).astype(np.float32)
        colors = rng.integers(0, 256, (5, 3), dtype=np.uint8)
        writer.finalize(points=points, colors=colors)
    else:
        writer.finalize()
    return out_dir


def test_pose_round_trip() -> None:
    rng = np.random.default_rng(42)
    for _ in range(20):
        euler = rng.uniform(-180.0, 180.0, 3)
        translation = rng.uniform(-5.0, 5.0, 3)
        r = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = r
        c2w[:3, 3] = translation

        quat = rotation_matrix_to_quat(r)
        reconstructed = quat_translation_to_c2w(quat, translation.tolist())

        np.testing.assert_allclose(reconstructed, c2w, atol=1e-6)


def test_read_back(tmp_path: Path) -> None:
    _build_canonical(tmp_path, params=[50.0, 50.0, 16.0, 16.0])
    src = Anycam4DSource(tmp_path)

    assert src.eval_camera_ids == {_EVAL_CAM_ID}
    assert src.frame_ids() == [0, 1]

    batches = list(src.stream(batch_frames=1))
    assert len(batches) == 2

    expected_k = np.array([[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    for frame_id, (batch, ts) in enumerate(zip(batches, _FRAME_TIMESTAMPS, strict=True)):
        assert len(batch) == 3
        assert {v.cam_id for v in batch} == {0, 1, 2}
        for view in batch:
            assert view.frame_id == frame_id
            assert view.timestamp == pytest.approx(ts)
            assert view.model is CameraModelType.PINHOLE
            assert view.params == pytest.approx([50.0, 50.0, 16.0, 16.0])
            np.testing.assert_allclose(view.K, expected_k)


def test_point_cloud(tmp_path: Path) -> None:
    _build_canonical(tmp_path / "with_points", with_points=True)
    src = Anycam4DSource(tmp_path / "with_points")
    xyz, rgb = src.point_cloud()
    assert xyz.shape == (5, 3)
    assert rgb.shape == (5, 3)

    _build_canonical(tmp_path / "no_points", with_points=False)
    src_no_points = Anycam4DSource(tmp_path / "no_points")
    assert src_no_points.point_cloud() is None


def test_identity_round_trip(tmp_path: Path) -> None:
    canon = tmp_path / "canon"
    _build_canonical(
        canon, model=CameraModelType.OPENCV, params=[50.0, 50.0, 16.0, 16.0, 0.05, -0.02, 0.001, 0.001, 0.0]
    )
    out = tmp_path / "identity"
    build_variant(canon, out, VariantSpec("identity", []), jobs=2)

    for filename in ("train_dataset.json", "eval_dataset.json"):
        canon_raw = json.loads((canon / filename).read_text())
        out_raw = json.loads((out / filename).read_text())
        _assert_calibration_equal(canon_raw, out_raw)

    out_train = json.loads((out / "train_dataset.json").read_text())
    out_eval = json.loads((out / "eval_dataset.json").read_text())
    assert out_train.get("pointcloud_path") == "points3d.ply"
    assert "pointcloud_path" not in out_eval
    assert {c["camera_id"] for c in out_eval["captures"]} == {_EVAL_CAM_ID}


def _assert_calibration_equal(canon_raw: dict, out_raw: dict) -> None:
    canon_caps = sorted(canon_raw["captures"], key=lambda c: (c["camera_id"], c["timestamp"]))
    out_caps = sorted(out_raw["captures"], key=lambda c: (c["camera_id"], c["timestamp"]))
    assert len(canon_caps) == len(out_caps)
    for c, o in zip(canon_caps, out_caps, strict=True):
        assert c["camera_id"] == o["camera_id"]
        assert c["timestamp"] == pytest.approx(o["timestamp"])
        assert c["rotation"] == pytest.approx(o["rotation"], abs=1e-5)
        assert c["translation"] == pytest.approx(o["translation"], abs=1e-5)

    canon_cams = {c["camera_id"]: c for c in canon_raw["cameras"]}
    out_cams = {c["camera_id"]: c for c in out_raw["cameras"]}
    assert canon_cams.keys() == out_cams.keys()
    for cid in canon_cams:
        assert canon_cams[cid]["model"] == out_cams[cid]["model"]
        assert canon_cams[cid]["params"] == pytest.approx(out_cams[cid]["params"])


def test_resize_variant(tmp_path: Path) -> None:
    canon = tmp_path / "canon"
    _build_canonical(canon, params=[50.0, 50.0, 16.0, 16.0])
    out = tmp_path / "r2"
    build_variant(canon, out, VariantSpec("r2", [Resize(factor=2)]), jobs=2)

    raw = json.loads((out / "train_dataset.json").read_text())
    for cam in raw["cameras"]:
        assert cam["width"] == _HW // 2
        assert cam["height"] == _HW // 2
        assert cam["params"][:4] == pytest.approx([25.0, 25.0, 8.0, 8.0])
        assert cam["model"] == "pinhole"

    dataset = MultiViewSequenceDataset(out)
    assert len(dataset) == len(_TRAIN_CAM_IDS) * len(_FRAME_TIMESTAMPS)


def test_to_pinhole_variant(tmp_path: Path) -> None:
    canon = tmp_path / "canon"
    _build_canonical(
        canon, model=CameraModelType.OPENCV, params=[50.0, 50.0, 16.0, 16.0, 0.05, -0.02, 0.001, 0.001, 0.0]
    )
    out = tmp_path / "undistorted"
    build_variant(canon, out, VariantSpec("undistorted", [ToPinhole()]), jobs=2)

    native_params = [50.0, 50.0, 16.0, 16.0, 0.05, -0.02, 0.001, 0.001, 0.0]
    raw = json.loads((out / "train_dataset.json").read_text())
    for cam in raw["cameras"]:
        assert cam["model"] == "pinhole"
        assert len(cam["params"]) == 4
        # The native lens survives the undistort as provenance, so validity stays resolvable at load.
        assert cam["reproject_source"] == {
            "model": "opencv",
            "width": _HW,
            "height": _HW,
            "params": pytest.approx(native_params),
        }
    MultiViewSequenceDataset(out)


def test_pinhole_variant_gains_no_reproject_source(tmp_path: Path) -> None:
    """A pinhole-source dataset is untouched by the provenance change: no key, same calibration."""
    canon = tmp_path / "canon"
    _build_canonical(canon, params=[50.0, 50.0, 16.0, 16.0])
    out = tmp_path / "undistorted"
    build_variant(canon, out, VariantSpec("undistorted", [ToPinhole()]), jobs=2)

    canon_raw = json.loads((canon / "train_dataset.json").read_text())
    out_raw = json.loads((out / "train_dataset.json").read_text())
    _assert_calibration_equal(canon_raw, out_raw)
    for cam in out_raw["cameras"]:
        assert "reproject_source" not in cam
