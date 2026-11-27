"""The ray solve's residual moves no pixel across the central/peripheral boundary."""

from __future__ import annotations

import numpy as np
import pytest

from tests.region_boundary import DYNOPSIS_SCENE1_LENSES
from tests.region_boundary import LensSpec
from tests.region_boundary import measure_lens


@pytest.fixture(scope="module", params=DYNOPSIS_SCENE1_LENSES, ids=lambda lens: lens.name)
def bound(request: pytest.FixtureRequest):
    """The measured boundary bound for one dynopsis scene-1 lens."""
    return measure_lens(request.param)


def test_ray_solve_converges_on_the_representable_field(bound) -> None:
    """Median round-trip error is sub-milli-pixel."""
    assert bound.roundtrip_median_px < 1e-3, f"median round trip {bound.roundtrip_median_px:.2e} px"


def test_the_solver_residual_moves_no_pixel_across_the_region_boundary(bound) -> None:
    """The whole point: refining every ray to machine precision changes no region membership."""
    assert bound.solver_flips == 0, (
        f"{bound.solver_flips} of {bound.observed_px} observed pixels change region when the ray "
        f"solve is refined — the region split would be solver-dependent"
    )


def test_rays_past_the_hemisphere_are_confined_to_the_peripheral_region(bound) -> None:
    """Rays past 90 degrees, which ``z == 1`` cannot represent, lie entirely in the peripheral region."""
    assert bound.unsolved_all_peripheral, (
        f"{bound.unsolved_px} pixels with an unsolvable ray are not all peripheral — "
        f"the smallest sits at {bound.unsolved_min_angle_deg:.1f} deg of field angle"
    )
    if bound.unsolved_px:
        assert bound.unsolved_min_angle_deg > 60.0


def test_a_lens_with_no_distortion_has_no_boundary_error_at_all() -> None:
    """The degenerate control: with nothing to invert, every source of error must vanish."""
    lens = next(lens for lens in DYNOPSIS_SCENE1_LENSES if not lens.distortion)
    b = measure_lens(lens)
    assert b.solver_flips == 0
    assert b.central_px == b.observed_px  # a pinhole is its own reference: nothing is peripheral


def test_a_strongly_pincushioned_lens_still_converges() -> None:
    """A monotone ``k1 = +0.5`` lens converges sub-pixel out to radius 1.5."""
    from anycam4d.enums import CameraModelType  # noqa: PLC0415 — local to this one case

    w, h = 320, 240
    f = float(np.hypot(w / 2, h / 2) / 1.5)
    lens = LensSpec("pincushion", CameraModelType.OPENCV, f, f, w / 2, h / 2, w, h, (0.5,))
    b = measure_lens(lens)
    assert b.roundtrip_max_px < 1e-3
    assert b.solver_flips == 0
