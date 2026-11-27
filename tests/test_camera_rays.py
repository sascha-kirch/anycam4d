"""The per-pixel ray solve converges: every ray projects back onto its own pixel.

OpenCV's default undistort stops after a fixed iteration count and returns unconverged rays on wide,
strongly distorted lenses.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from anycam4d.enums import CameraModelType
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.lens_domain import PIXEL_CENTRE


def _roundtrip_error(K: np.ndarray, dist: np.ndarray, model: CameraModelType, w: int, h: int) -> np.ndarray:
    """Per-pixel |project(camera_rays(p)) - p| in pixels."""
    rays, _ = camera_rays(torch.tensor(K, dtype=torch.float32), torch.tensor(dist, dtype=torch.float32), model, w, h)
    obj = rays.reshape(-1, 3).double().numpy().reshape(-1, 1, 3)
    zero = np.zeros((3, 1))
    if model in (CameraModelType.FISHEYE_OPENCV, CameraModelType.FISHEYE_EQUIDISTANT):
        pix, _ = cv2.fisheye.projectPoints(obj, zero, zero, K, dist)
    else:
        pix, _ = cv2.projectPoints(obj, zero, zero, K, dist)
    pix = pix.reshape(-1, 2)
    # ``camera_rays`` samples pixel ``i`` at ``i + PIXEL_CENTRE``, so that is the coordinate the
    # ray has to project back onto.
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.hypot(pix[:, 0] - (uu.ravel() + PIXEL_CENTRE), pix[:, 1] - (vv.ravel() + PIXEL_CENTRE))


def _K(w: int, h: int, r_corner: float) -> np.ndarray:
    """Intrinsics whose frame corner sits at normalised radius ``r_corner`` (≈ tan of half the
    diagonal field), the quantity that actually governs solver difficulty — not k1 alone."""
    f = float(np.hypot(w / 2, h / 2) / r_corner)
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


# k1 > 0, no turning point: sub-pixel everywhere is exact. OpenCV's default leaves 18.4 px at r_corner = 1.5.
@pytest.mark.parametrize("r_corner", [0.8, 1.2, 1.5])
@pytest.mark.parametrize("k1", [0.2, 0.5])
def test_pinhole_ray_solve_converges_on_a_monotone_lens(k1, r_corner):
    w, h = 320, 240
    dist = np.zeros(12)
    dist[0] = k1
    err = _roundtrip_error(_K(w, h, r_corner), dist, CameraModelType.OPENCV, w, h)
    assert np.isfinite(err).all()
    assert err.max() < 1e-3, f"unconverged ray solve: max round-trip error {err.max():.3f} px"


def test_pinhole_ray_solve_converges_with_tangential_and_prism():
    """The full 12-slot model, not just the radial terms."""
    w, h = 320, 240
    dist = np.zeros(12)
    dist[:5] = [0.3, 0.08, 0.001, -0.002, 0.01]  # k1,k2,p1,p2,k3
    dist[8:10] = [0.0005, -0.0005]  # s1,s2
    err = _roundtrip_error(_K(w, h, 1.2), dist, CameraModelType.OPENCV, w, h)
    assert err.max() < 1e-2, f"max round-trip error {err.max():.3f} px"


@pytest.mark.parametrize("k1", [0.0, 0.1, -0.05])
def test_fisheye_ray_solve_converges(k1):
    """Kannala-Brandt, inside the modelled angular domain."""
    w, h = 320, 240
    dist = np.zeros(4)
    dist[0] = k1
    K = np.array([[200.0, 0.0, w / 2], [0.0, 200.0, h / 2], [0.0, 0.0, 1.0]])
    err = _roundtrip_error(K, dist, CameraModelType.FISHEYE_OPENCV, w, h)
    assert np.isfinite(err).all()
    assert err.max() < 1e-3, f"unconverged fisheye ray solve: max error {err.max():.3f} px"


def test_zero_distortion_is_the_exact_pinhole_ray():
    """No distortion ⇒ the ray is exactly normalize((u-cx)/fx, (v-cy)/fy, 1), no solver involved."""
    w, h = 64, 48
    K = _K(w, h, 0.8)
    rays, valid = camera_rays(torch.tensor(K, dtype=torch.float32), torch.zeros(12), CameraModelType.PINHOLE, w, h)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    uu, vv = uu + PIXEL_CENTRE, vv + PIXEL_CENTRE
    expected = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], axis=-1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)

    assert bool(valid.all()), "an undistorted pinhole observes every pixel"
    assert np.allclose(rays.numpy(), expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(rays.numpy(), axis=-1), 1.0, atol=1e-6)


# Unit-sphere contract: OpenCV's fisheye undistort returns z = 1 points and clamps at theta_d = pi/2.

# 64x64 fisheye, f = 20, k1 = 0.1: corner incidence 98.5 deg, monotone everywhere.
_WIDE_KB_W = _WIDE_KB_H = 64
_WIDE_KB_K = np.array([[20.0, 0.0, 31.5], [0.0, 20.0, 31.5], [0.0, 0.0, 1.0]])
_WIDE_KB_DIST = np.array([0.1, 0.0, 0.0, 0.0])


def _kb_theta(K: np.ndarray, dist: np.ndarray, w: int, h: int) -> np.ndarray:
    """True incidence angle per pixel of a Kannala-Brandt lens, by bisection on theta_d(theta)."""
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    theta_d = np.hypot((uu + PIXEL_CENTRE - K[0, 2]) / K[0, 0], (vv + PIXEL_CENTRE - K[1, 2]) / K[1, 1])
    lo = np.zeros_like(theta_d)
    hi = np.full_like(theta_d, np.pi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = mid * mid
        below = mid * (1.0 + s * (dist[0] + s * (dist[1] + s * (dist[2] + s * dist[3])))) < theta_d
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _kb_project(dirs: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Kannala-Brandt projection with ``atan2(r, z)``, valid past a quarter turn."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    s = theta * theta
    theta_d = theta * (1.0 + s * (dist[0] + s * (dist[1] + s * (dist[2] + s * dist[3]))))
    scale = np.divide(theta_d, r, out=np.zeros_like(r), where=r > 0.0)
    return np.stack([K[0, 0] * x * scale + K[0, 2], K[1, 1] * y * scale + K[1, 2]], axis=-1)


def _wide_kb_rays() -> tuple[np.ndarray, np.ndarray]:
    rays, valid = camera_rays(
        torch.tensor(_WIDE_KB_K, dtype=torch.float32),
        torch.tensor(_WIDE_KB_DIST, dtype=torch.float32),
        CameraModelType.FISHEYE_OPENCV,
        _WIDE_KB_W,
        _WIDE_KB_H,
    )
    return rays.double().numpy(), valid.numpy()


def test_rays_are_unit_length():
    """The contract is the unit sphere, not the z = 1 plane."""
    rays, valid = _wide_kb_rays()
    assert np.allclose(np.linalg.norm(rays[valid], axis=-1), 1.0, atol=1e-6)


def test_past_quarter_turn_rays_point_backwards():
    """A pixel whose incidence exceeds 90 deg gets a negative z, not a mirrored forward ray."""
    rays, valid = _wide_kb_rays()
    theta = _kb_theta(_WIDE_KB_K, _WIDE_KB_DIST, _WIDE_KB_W, _WIDE_KB_H)
    past = theta > np.pi / 2

    assert past.any(), "fixture no longer reaches past a quarter turn"
    assert valid.all(), "this lens is monotone everywhere, so every pixel is invertible"
    # The defect this pins: no ray may claim to point forward where the true one points backwards.
    assert not (rays[..., 2] > 0.0)[past].any()
    assert np.allclose(rays[..., 2], np.cos(theta), atol=1e-6)


def test_incidence_angle_survives_opencvs_theta_d_clamp():
    """Every pixel past theta_d = pi/2 keeps its own angle instead of collapsing onto one value."""
    rays, _ = _wide_kb_rays()
    theta = _kb_theta(_WIDE_KB_K, _WIDE_KB_DIST, _WIDE_KB_W, _WIDE_KB_H)
    clamped = theta > 1.33  # where OpenCV's theta_d = pi/2 cap engages for k1 = 0.1

    recovered = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
    assert clamped.any()
    assert np.allclose(recovered[clamped], theta[clamped], atol=1e-6)
    assert recovered[clamped].max() - recovered[clamped].min() > 0.3  # not collapsed onto a value


def test_valid_pixels_round_trip_to_their_own_pixel():
    """Projecting a returned ray back through its lens lands on the pixel it came from."""
    rays, valid = _wide_kb_rays()
    uu, vv = np.meshgrid(np.arange(_WIDE_KB_W, dtype=np.float64), np.arange(_WIDE_KB_H, dtype=np.float64))
    back = _kb_project(rays, _WIDE_KB_K, _WIDE_KB_DIST)

    err = np.hypot(back[..., 0] - (uu + PIXEL_CENTRE), back[..., 1] - (vv + PIXEL_CENTRE))[valid]
    assert err.max() < 1e-4, f"ray does not invert its own lens: max {err.max():.4f} px"


def test_outside_the_monotone_domain_is_invalid_and_nan():
    """Past the fold of the radial map there is no determinable ray, and none is fabricated."""
    dist = np.array([-0.2, 0.0, 0.0, 0.0])  # fold at theta = 1.291 rad, theta_d = 0.861
    rays, valid = camera_rays(
        torch.tensor(_WIDE_KB_K, dtype=torch.float32),
        torch.tensor(dist, dtype=torch.float32),
        CameraModelType.FISHEYE_OPENCV,
        _WIDE_KB_W,
        _WIDE_KB_H,
    )
    rays, valid = rays.double().numpy(), valid.numpy()

    assert not valid.all() and valid.any()
    assert bool(valid[32, 32]), "the optical axis is always invertible"
    assert not bool(valid[0, 0]), "the corner is far past the fold"
    assert np.isnan(rays[~valid]).all(), "an invalid pixel must not carry a plausible direction"
    assert np.isfinite(rays[valid]).all()
