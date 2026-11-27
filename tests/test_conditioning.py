"""Tests for anycam4d.primitives.fourdgaussians.conditioning."""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.fourdgaussians.conditioning import condition_at_t
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def test_conditioned_opacities_are_sigmoid_times_wt_not_squared() -> None:
    """Conditioned opacity is ``sigmoid(raw) · w_t``, not ``· w_t²``."""
    torch.manual_seed(42)
    N = 30
    model = FourDGaussians.from_random(N)

    with torch.no_grad():
        model.splats["mu_t"].fill_(0.5)
        # Moderate temporal spread so w_t is clearly between 0 and 1 at tau=0.7
        model.splats["scales_4d_log"][:, 3] = -1.0

    tau = 0.7  # displaced from center → w_t < 1

    cond = condition_at_t(model, tau=tau, prefilter_threshold=0.0)
    w_t = cond.temporal_weights_raw

    expected_single = torch.sigmoid(model.splats["opacities"]) * w_t  # correct
    expected_double = torch.sigmoid(model.splats["opacities"]) * w_t * w_t  # bug

    err_single = (cond.opacity_cond - expected_single).abs().max().item()
    err_double = (cond.opacity_cond - expected_double).abs().max().item()

    assert err_single < 1e-5, f"cond.opacity_cond does not match sigmoid*w_t (err={err_single:.2e})"
    assert err_double > 1e-3, "cond.opacity_cond equals sigmoid*w_t^2 — double application is present"


def test_filtered_opacity_is_zero() -> None:
    """When w_t < prefilter_threshold, ConditionedGaussians.opacities must be exactly 0."""
    torch.manual_seed(21)
    N = 10
    threshold = 0.05
    model = FourDGaussians.from_random(N)

    with torch.no_grad():
        # Place all temporal centers far from tau via very small temporal scale
        model.splats["mu_t"].fill_(0.0)
        model.splats["scales_4d_log"][:, 3] = -20.0  # cov_t_marg ≈ tiny → w_t ≈ 0 at tau=0.9

    tau = 0.9  # far from mu_t=0

    cond_raw = condition_at_t(model, tau=tau, prefilter_threshold=0.0)
    assert (cond_raw.temporal_weights_raw < threshold).all(), (
        f"Setup error: expected all w_t < {threshold}, got max w_t = {cond_raw.temporal_weights_raw.max():.4e}"
    )

    cond = condition_at_t(model, tau=tau, prefilter_threshold=threshold)
    assert (cond.opacity_cond == 0.0).all(), (
        f"Filtered Gaussians should have opacity==0, got max={cond.opacity_cond.max():.4e}"
    )


def _wt_at(sigma_t: float, dt: float) -> float:
    """Rendered temporal weight of one primitive at offset ``dt``, for temporal extent ``sigma_t``."""
    model = FourDGaussians.from_random(1)
    with torch.no_grad():
        model.splats["mu_t"].fill_(0.5)
        model.splats["scales_4d_log"][:, 3] = torch.log(torch.tensor(sigma_t))
        model.splats["scales_4d_log"][:, :3] = -4.0  # keep the rotation from mixing space into time
        model.splats["quats_l"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.splats["quats_r"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    cond = condition_at_t(model, tau=0.5 + dt)
    return (cond.opacity_cond / torch.sigmoid(model.splats["opacities"])).item()


def test_temporal_extent_below_the_variance_epsilon_is_unrenderable() -> None:
    """Below ``sqrt(eps)``, primitives decades apart in ``sigma_t`` render the same gate."""
    dt = TEMPORAL_SIGMA_FLOOR  # one floor-width away, where the gate is still well conditioned
    far_below = _wt_at(TEMPORAL_SIGMA_FLOOR / 100.0, dt)
    much_further_below = _wt_at(TEMPORAL_SIGMA_FLOOR / 10_000.0, dt)

    # Two decades under the floor the primitive's own variance is 1e-4 of the epsilon, so it can
    # still shift the gate in the fifth decimal; four decades under, it cannot shift it at all.
    assert far_below == pytest.approx(much_further_below, abs=1e-4)
    assert far_below == pytest.approx(math.exp(-0.5), rel=1e-3), "the epsilon alone sets the width"


def test_temporal_extent_above_the_floor_is_renderable() -> None:
    """The floor is a floor, not a ceiling: above it, ``sigma_t`` still moves the rendered gate."""
    dt = TEMPORAL_SIGMA_FLOOR
    at_ten_floors = _wt_at(TEMPORAL_SIGMA_FLOOR * 10.0, dt)
    at_the_floor = _wt_at(TEMPORAL_SIGMA_FLOOR, dt)

    assert at_ten_floors > at_the_floor + 0.1, "sigma_t must be visible to the renderer above the floor"


def test_temporal_floor_matches_the_declared_constant() -> None:
    """``TEMPORAL_SIGMA_FLOOR == sqrt(TEMPORAL_VARIANCE_EPS)``, and the literal is pinned: changing it re-renders every checkpoint."""
    assert TEMPORAL_SIGMA_FLOOR == pytest.approx(math.sqrt(TEMPORAL_VARIANCE_EPS))
    assert TEMPORAL_SIGMA_FLOOR == pytest.approx(3.162e-3, rel=1e-3)
