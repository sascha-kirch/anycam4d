"""Pixel index ``i`` is continuous coordinate ``i + 0.5`` everywhere host-side, as the rasterizer samples; OpenCV converts by ±0.5."""

import numpy as np
import pytest
import torch

from anycam4d.camera.validity import compute_validity
from anycam4d.enums import CameraModelType
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.lens_domain import PIXEL_CENTRE


def _pinhole_ray(K: np.ndarray, u: float, v: float) -> np.ndarray:
    """The distortion-free ray for one continuous image coordinate, normalised."""
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    return d / np.linalg.norm(d)


def test_a_ray_is_the_one_the_rasterizer_draws_at_that_pixel() -> None:
    """``camera_rays`` samples ``i + 0.5``, which is the shader's ``(px + 0.5 - cx) / fx``."""
    width, height = 32, 24
    K = np.array([[40.0, 0.0, 15.5], [0.0, 40.0, 11.5], [0.0, 0.0, 1.0]])
    rays, valid = camera_rays(
        torch.tensor(K, dtype=torch.float32), torch.zeros(12), CameraModelType.PINHOLE, width, height
    )
    assert bool(valid.all())

    for py, px in ((0, 0), (5, 13), (height - 1, width - 1)):
        drawn = _pinhole_ray(K, px + PIXEL_CENTRE, py + PIXEL_CENTRE)
        integer = _pinhole_ray(K, float(px), float(py))
        got = rays[py, px].numpy()
        np.testing.assert_allclose(got, drawn, atol=1e-6)
        # The rejected convention is genuinely a different direction, not a rounding difference.
        assert not np.allclose(got, integer, atol=1e-4)


def test_the_optical_axis_lands_on_the_pixel_that_contains_it() -> None:
    """With ``cx = w/2`` the axis falls on the boundary between the two central pixels, symmetrically."""
    width, height = 16, 16
    K = np.array([[20.0, 0.0, width / 2.0], [0.0, 20.0, height / 2.0], [0.0, 0.0, 1.0]])
    rays, _ = camera_rays(torch.tensor(K, dtype=torch.float32), torch.zeros(12), CameraModelType.PINHOLE, width, height)
    left = rays[height // 2, width // 2 - 1].numpy()
    right = rays[height // 2, width // 2].numpy()
    assert left[0] == pytest.approx(-right[0])
    assert left[1] == pytest.approx(right[1])


def test_validity_and_the_ray_solve_agree_pixel_for_pixel() -> None:
    """Both are the same lens domain, now sampled at the same point, so no rim survives."""
    K = torch.tensor([[24.0, 0.0, 31.5], [0.0, 24.0, 23.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
    dist = torch.tensor([-0.2, 0.0, 0.0, 0.0], dtype=torch.float32)

    footprint = compute_validity(CameraModelType.FISHEYE_OPENCV, K, dist, 64, 48)
    _, ray_valid = camera_rays(K, dist, CameraModelType.FISHEYE_OPENCV, 64, 48)

    assert not bool(footprint.all()) and bool(footprint.any())
    assert torch.equal(footprint, ray_valid)
