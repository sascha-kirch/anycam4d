"""Observation validity per camera branch and the resolution precedence chain."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from anycam4d.camera.validity import compute_validity
from anycam4d.camera.validity import resolve_validity
from anycam4d.data.schema import CameraSchema
from anycam4d.data.schema import ReprojectSourceSchema
from anycam4d.enums import CameraModelType
from anycam4d.math.lens_domain import PIXEL_CENTRE


def _K(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _normalized_radius_grid(K: torch.Tensor, width: int, height: int) -> np.ndarray:
    """Per-pixel normalised radius at the pixel centres the rasterizer samples (``+ 0.5``)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return np.hypot((uu + 0.5 - cx) / fx, (vv + 0.5 - cy) / fy)


def _pinhole_camera(width: int, height: int, focal: float, **kwargs: object) -> CameraSchema:
    return CameraSchema(
        camera_id=0,
        model=CameraModelType.PINHOLE,
        width=width,
        height=height,
        params=[focal, focal, (width - 1) / 2, (height - 1) / 2],
        **kwargs,
    )


@pytest.mark.parametrize(("width", "height"), [(64, 48), (200, 200), (1920, 1080)])
def test_pinhole_no_distortion_all_true(width: int, height: int) -> None:
    """A distortion-free pinhole observes its whole rectangle at any resolution."""
    K = _K(width * 0.8, width * 0.8, width / 2, height / 2)
    valid = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height)

    assert valid.shape == (height, width)
    assert valid.dtype is torch.bool
    assert bool(valid.all())


def test_recorded_field_limit_bounds_a_bare_pinhole() -> None:
    """A distortion-free pinhole with a recorded field limit is not assumed fully observed."""
    width, height = 64, 48
    K = _K(20.0, 20.0, width / 2, height / 2)  # corner at ~63 deg incidence
    bare = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height)
    bounded = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height, max_incidence_deg=45.0)

    assert bool(bare.all())
    assert not bool(bounded[0, 0]) and bool(bounded[height // 2, width // 2])


_FISHEYE_NATIVE = ReprojectSourceSchema(
    model=CameraModelType.FISHEYE_OPENCV,
    width=110,
    height=90,
    params=[60.0, 60.0, 54.5, 44.5, 0.02, -0.01, 0.0, 0.0],
)


def _expected_reproject_mask(pinhole_K: torch.Tensor, width: int, height: int) -> np.ndarray:
    """Independent recomputation: unproject the pinhole grid, project through the native fisheye."""
    fx, fy = float(pinhole_K[0, 0]), float(pinhole_K[1, 1])
    cx, cy = float(pinhole_K[0, 2]), float(pinhole_K[1, 2])
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    u_n = (uu.ravel() + PIXEL_CENTRE - cx) / fx
    v_n = (vv.ravel() + PIXEL_CENTRE - cy) / fy
    dirs = np.stack([u_n, v_n, np.ones(width * height)], axis=-1)

    K_src = np.array([[60.0, 0.0, 54.5], [0.0, 60.0, 44.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([0.02, -0.01, 0.0, 0.0])
    pix, _ = cv2.fisheye.projectPoints(dirs.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K_src, dist)
    u, v = pix[:, 0, 0], pix[:, 0, 1]
    inside = (u >= 0) & (u < 110) & (v >= 0) & (v < 90)
    return inside.reshape(height, width)


def test_reproject_source_matches_independent_replay() -> None:
    """The provenance branch equals an independently recomputed unproject→project→in-bounds test."""
    width, height = 128, 96
    K = _K(50.0, 50.0, 63.5, 47.5)  # wide pinhole: rays leave the native fisheye image
    valid = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height, source=_FISHEYE_NATIVE)

    expected = _expected_reproject_mask(K, width, height)
    assert valid.shape == (height, width)
    np.testing.assert_array_equal(valid.numpy(), expected)


def test_reproject_source_wide_pinhole_has_invalid_ring() -> None:
    """A pinhole opened past the native footprint grows genuinely unobserved corners."""
    width, height = 128, 96
    K = _K(30.0, 30.0, 63.5, 47.5)  # very wide: the corners map outside the native image
    valid = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height, source=_FISHEYE_NATIVE)

    assert not bool(valid.all())
    assert bool(valid[height // 2, width // 2])  # centre is always observed
    assert not bool(valid[0, 0])  # a corner is not


def test_reproject_source_cropped_inside_footprint_all_true() -> None:
    """A narrow pinhole entirely inside the native footprint is fully observed (no false invalids)."""
    width, height = 64, 48
    K = _K(200.0, 200.0, 31.5, 23.5)  # long focal ⇒ tiny angular extent, well inside the fisheye
    valid = compute_validity(CameraModelType.PINHOLE, K, torch.zeros(12), width, height, source=_FISHEYE_NATIVE)

    assert bool(valid.all())


def test_resolve_reproject_source_branch() -> None:
    """A pinhole carrying reproject_source resolves through the provenance branch."""
    camera = _pinhole_camera(128, 96, 30.0, reproject_source=_FISHEYE_NATIVE)
    valid = resolve_validity(camera, None, 128, 96)

    assert not bool(valid.all())


# Self-describing native lens: valid(p) iff p is the image of the monotone disc; expectations are derived in closed form.


def _pinhole_dist(k1: float = 0.0, **slots: float) -> torch.Tensor:
    """Canonical 12-slot pinhole distortion vector: ``[k1, k2, p1, p2, k3, k4, k5, k6, s1..s4]``."""
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
    dist = torch.zeros(12)
    dist[0] = k1
    for name, value in slots.items():
        dist[names.index(name)] = value
    return dist


def _K_for_corner(width: int, height: int, r_corner: float) -> torch.Tensor:
    """Intrinsics whose frame corner sits at normalised radius ``r_corner`` (≈ tan of half the
    diagonal field) — the quantity that actually governs whether a lens folds inside its frame."""
    focal = math.hypot(width, height) / 2.0 / r_corner
    return _K(focal, focal, (width - 1) / 2, (height - 1) / 2)


@pytest.mark.parametrize(("width", "height"), [(80, 60), (160, 120)])
@pytest.mark.parametrize("k1", [0.2, 0.5])
@pytest.mark.parametrize("r_corner", [0.8, 1.2, 1.5])
def test_monotone_lens_is_entirely_valid(width: int, height: int, k1: float, r_corner: float) -> None:
    """A pincushion lens (``k1 > 0``) is monotone everywhere, so every pixel is observed at any resolution."""
    K = _K_for_corner(width, height, r_corner)
    valid = compute_validity(CameraModelType.OPENCV, K, _pinhole_dist(k1), width, height)

    assert bool(valid.all())


@pytest.mark.parametrize(("width", "height"), [(80, 60), (160, 120)])
@pytest.mark.parametrize("k1", [-0.2, -0.3, -0.5])
def test_folded_lens_is_exactly_the_analytic_disc(width: int, height: int, k1: float) -> None:
    """A barrel lens is observed exactly out to ``r_d(r_max)``, ``r_max = sqrt(-1/(3k1))``."""
    K = _K_for_corner(width, height, 1.2)
    valid = compute_validity(CameraModelType.OPENCV, K, _pinhole_dist(k1), width, height)

    r_max = math.sqrt(-1.0 / (3.0 * k1))
    radius = r_max * (1.0 + k1 * r_max * r_max)
    expected = _normalized_radius_grid(K, width, height) <= radius

    assert not bool(valid.all())  # the fixture must actually fold inside the frame
    np.testing.assert_array_equal(valid.numpy(), expected)


def test_tangential_and_prism_footprint_is_star_shaped_within_the_displacement_bound() -> None:
    """With tangential/prism terms the footprint stays star-shaped and within the displacement bound of the radial disc.

    On ``|x| = r_max``: ``|Δx| <= (|p1| + 3|p2| + |s1|) r² + |s2| r⁴``, ``|Δy| <= (3|p1| + |p2| + |s3|) r² + |s4| r⁴``.
    """
    width, height = 96, 96
    K = _K_for_corner(width, height, 1.2)
    k1, p1, p2, s1, s2, s3, s4 = -0.3, 0.01, -0.015, 0.004, -0.002, 0.003, 0.001
    dist = _pinhole_dist(k1, p1=p1, p2=p2, s1=s1, s2=s2, s3=s3, s4=s4)
    valid = compute_validity(CameraModelType.OPENCV, K, dist, width, height).numpy()

    r_max = math.sqrt(-1.0 / (3.0 * k1))
    radius = r_max * (1.0 + k1 * r_max**2)
    r2, r4 = r_max**2, r_max**4
    delta = math.hypot(
        (abs(p1) + 3 * abs(p2) + abs(s1)) * r2 + abs(s2) * r4,
        (3 * abs(p1) + abs(p2) + abs(s3)) * r2 + abs(s4) * r4,
    )
    r_norm = _normalized_radius_grid(K, width, height)

    assert not valid.all()
    assert valid[r_norm <= radius - delta].all(), "footprint must contain the shrunk radial disc"
    assert not valid[r_norm > radius + delta].any(), "footprint must stay inside the grown radial disc"

    cx, cy = float(K[0, 2]) - 0.5, float(K[1, 2]) - 0.5
    for angle in np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False):
        t = np.arange(0.0, min(width, height) / 2.0, 0.25)
        u = np.rint(cx + t * math.cos(angle)).astype(int)
        v = np.rint(cy + t * math.sin(angle)).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        along = valid[v[inside], u[inside]]
        transitions = int(np.count_nonzero(along[1:] != along[:-1]))
        assert transitions <= 1, f"ray at {angle:.2f} rad crosses the boundary {transitions} times"


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_folded_pinhole_validity_resize_invariant(factor: int) -> None:
    """Validity recomputed at half resolution with scaled intrinsics keeps the analytic area share."""
    width, height = 256 // factor, 256 // factor
    K = _K(96.0 / factor, 96.0 / factor, 127.5 / factor, 127.5 / factor)
    valid = compute_validity(CameraModelType.OPENCV, K, _pinhole_dist(-0.3), width, height)

    r_max = math.sqrt(-1.0 / (3.0 * -0.3))
    radius_px = r_max * (1.0 - 0.3 * r_max * r_max) * 96.0  # in full-resolution pixels
    expected = math.pi * radius_px**2 / 256.0**2

    assert float(valid.float().mean()) == pytest.approx(expected, abs=0.01)


def test_fisheye_equidistant_is_bounded_by_a_half_turn_not_by_a_fold() -> None:
    """An equidistant lens ends at θ = π (image circle ``f·π``), not ``f·π/2``."""
    width, height = 256, 256
    focal = 30.0
    K = _K(focal, focal, 127.5, 127.5)
    valid = compute_validity(CameraModelType.FISHEYE_EQUIDISTANT, K, torch.zeros(4), width, height)

    r = _normalized_radius_grid(K, width, height)
    np.testing.assert_array_equal(valid.numpy(), r <= math.pi)

    inside = _K(90.0, 90.0, 99.5, 99.5)  # corner at r = 141 px < 90·π ⇒ fully observed
    assert bool(compute_validity(CameraModelType.FISHEYE_EQUIDISTANT, inside, torch.zeros(4), 200, 200).all())


def test_fisheye_fold_shrinks_the_image_circle_below_the_half_turn_bound() -> None:
    """Kannala-Brandt with k1 = -0.15 is bounded by its fold at θ ≈ 1.49, not by π."""
    width, height = 200, 200
    focal = 60.0
    k1 = -0.15
    K = _K(focal, focal, 99.5, 99.5)
    valid = compute_validity(CameraModelType.FISHEYE_OPENCV, K, torch.tensor([k1, 0.0, 0.0, 0.0]), width, height)

    theta_max = math.sqrt(-1.0 / (3.0 * k1))
    assert theta_max < math.pi
    radius = theta_max * (1.0 + k1 * theta_max**2)
    np.testing.assert_array_equal(valid.numpy(), _normalized_radius_grid(K, width, height) <= radius)


def test_resolve_model_branch_for_native_fisheye() -> None:
    """A self-describing fisheye with no provenance resolves through the model branch."""
    camera = CameraSchema(
        camera_id=3,
        model=CameraModelType.FISHEYE_EQUIDISTANT,
        width=200,
        height=200,
        params=[20.0, 20.0, 99.5, 99.5],  # corner at θ = 7.07 rad, far past the half-turn bound
    )
    valid = resolve_validity(camera, None, 200, 200)

    assert not bool(valid.all())


def test_resolve_scales_intrinsics_to_requested_resolution() -> None:
    """Resolving at half resolution rescales the manifest intrinsics instead of resampling."""
    camera = CameraSchema(
        camera_id=3,
        model=CameraModelType.FISHEYE_EQUIDISTANT,
        width=200,
        height=200,
        params=[20.0, 20.0, 99.5, 99.5],
    )
    full = resolve_validity(camera, None, 200, 200)
    half = resolve_validity(camera, None, 100, 100)

    assert half.shape == (100, 100)
    assert not bool(full.all())  # the fixture must have a non-trivial footprint to compare
    assert float(half.float().mean()) == pytest.approx(float(full.float().mean()), abs=0.02)


def test_on_disk_mask_wins() -> None:
    """An explicit mask overrides every procedural branch, verbatim."""
    mask = torch.zeros((96, 128), dtype=torch.bool)
    mask[10:20, 10:20] = True
    camera = _pinhole_camera(128, 96, 50.0, reproject_source=_FISHEYE_NATIVE)

    assert resolve_validity(camera, mask, 128, 96) is mask


def test_reproject_source_round_trips() -> None:
    """A camera with well-formed provenance survives model_dump → model_validate unchanged."""
    camera = _pinhole_camera(128, 96, 50.0, reproject_source=_FISHEYE_NATIVE)
    restored = CameraSchema.model_validate(camera.model_dump())

    assert restored.reproject_source is not None
    assert restored.reproject_source.model is CameraModelType.FISHEYE_OPENCV
    assert restored.reproject_source.width == 110
    assert restored.reproject_source.params == _FISHEYE_NATIVE.params


def test_reproject_source_rejects_wrong_param_count() -> None:
    """The native block validates its param count against its own model."""
    with pytest.raises(ValidationError, match="reproject_source"):
        ReprojectSourceSchema(
            model=CameraModelType.FISHEYE_OPENCV,
            width=160,
            height=120,
            params=[60.0, 60.0, 79.5, 59.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # one too many
        )


def test_reproject_source_optional_on_plain_pinhole() -> None:
    """A genuine native pinhole leaves the block None — there is no present-iff rule."""
    camera = _pinhole_camera(64, 48, 50.0)
    assert camera.reproject_source is None
