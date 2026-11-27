"""COLMAP camera-model mapping and pose conversion on hand-built ``pycolmap`` objects."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from scipy.spatial.transform import Rotation

from anycam4d.camera.registry import param_count_range
from anycam4d.data.prep.colmap import _colmap_camera_to_anycam
from anycam4d.data.prep.colmap import c2w_from_cam_from_world
from anycam4d.enums import CameraModelType


def test_opencv_camera_maps_to_anycam_opencv() -> None:
    """A COLMAP OPENCV camera -> anycam4d opencv, params in canonical order, count in range."""
    cam = pycolmap.Camera.create(1, "OPENCV", 500.0, 800, 600)
    cam.params = [520.0, 530.0, 401.0, 299.0, 0.1, -0.2, 0.01, -0.01]  # fx, fy, cx, cy, k1, k2, p1, p2

    result = _colmap_camera_to_anycam(cam)

    assert result.model is CameraModelType.OPENCV
    assert result.width == 800
    assert result.height == 600
    assert result.params == [520.0, 530.0, 401.0, 299.0, 0.1, -0.2, 0.01, -0.01]
    lo, hi = param_count_range(CameraModelType.OPENCV)
    assert lo <= len(result.params) <= hi


def test_pinhole_camera_maps_to_anycam_pinhole() -> None:
    """A COLMAP PINHOLE camera -> anycam4d pinhole with exactly [fx, fy, cx, cy]."""
    cam = pycolmap.Camera.create(1, "PINHOLE", 500.0, 800, 600)
    cam.params = [510.0, 520.0, 400.0, 300.0]

    result = _colmap_camera_to_anycam(cam)

    assert result.model is CameraModelType.PINHOLE
    assert result.params == [510.0, 520.0, 400.0, 300.0]
    lo, hi = param_count_range(CameraModelType.PINHOLE)
    assert lo <= len(result.params) <= hi


def test_unsupported_model_raises() -> None:
    """An unmapped COLMAP model must fail loudly, not silently emit garbage."""
    cam = pycolmap.Camera.create(1, "SIMPLE_RADIAL", 500.0, 800, 600)
    with pytest.raises(ValueError, match="Unsupported COLMAP camera model"):
        _colmap_camera_to_anycam(cam)


def test_pycolmap_quaternion_order_is_xyzw() -> None:
    """Guards the pose math: pycolmap ``Rotation3d`` takes/returns scipy-order [x, y, z, w]."""
    rot = Rotation.from_euler("xyz", [0.3, -0.2, 0.5])
    quat_xyzw = rot.as_quat()
    rot3d = pycolmap.Rotation3d(quat_xyzw)
    np.testing.assert_allclose(rot3d.matrix(), rot.as_matrix(), atol=1e-9)
    np.testing.assert_allclose(rot3d.quat, quat_xyzw, atol=1e-9)


def test_cam_from_world_to_c2w_roundtrip() -> None:
    """A known world->camera (R, t) -> OpenCV c2w with R_c2w = R^T, C = -R^T t."""
    r_w2c = Rotation.from_euler("xyz", [0.1, 0.4, -0.3]).as_matrix()
    t = np.array([1.0, -2.0, 3.0])
    cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_matrix(r_w2c).as_quat()), t)

    c2w = c2w_from_cam_from_world(cam_from_world)

    np.testing.assert_allclose(c2w[:3, :3], r_w2c.T, atol=1e-6)
    np.testing.assert_allclose(c2w[:3, 3], -r_w2c.T @ t, atol=1e-6)
    np.testing.assert_allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    # Camera center equals COLMAP's own projection_center().
    image = pycolmap.Image(name="x", camera_id=1)
    image.cam_from_world = cam_from_world
    np.testing.assert_allclose(c2w[:3, 3], np.asarray(image.projection_center()), atol=1e-6)
