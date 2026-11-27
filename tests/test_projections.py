"""The ``mu_t`` projection and the truncated aliveness criterion.

``clamp_temporal_mean`` bounds a parameter with no restoring force. ``observed_mass`` replaces the
published mass, whose integral over all time credits primitives that render nothing and pays any
primitive to widen.
"""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.config.model import ModelConfig
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.strategy._ops import marginal_temporal_std
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


# Normalized time spans [-0.5, 0.5]; ``mu_t`` has no restoring force and random-walks under noise.


def _mu_t_model(mu_t: list[float], sigma_t: float = 0.05, margin: float = -1.0) -> FourDGaussians:
    """A model whose primitives carry the given temporal means and a common temporal extent."""
    model = FourDGaussians.from_random(n_points=len(mu_t), sh_degree=1, mu_t_margin_sigmas=margin)
    model.splats["mu_t"].data = torch.tensor(mu_t)
    model.splats["scales_4d_log"].data[:, 3] = math.log(sigma_t)
    return model


def _guarded_width(sigma_t: float) -> float:
    """The width the renderer's gate divides by for an axis-aligned primitive: ``sqrt(sigma_t² + eps)``."""
    return math.sqrt(sigma_t * sigma_t + TEMPORAL_VARIANCE_EPS)


def _auto_limit(sigma_t: float, threshold: float = 0.05) -> float:
    """The bound the auto policy resolves to: half the span plus k times the *rendered* width."""
    return 0.5 + math.sqrt(-2.0 * math.log(threshold)) * _guarded_width(sigma_t)


def test_clamp_bounds_runaway_temporal_means() -> None:
    """Means that have walked off the sequence come back to the edge of what can render."""
    model = _mu_t_model([0.0, 0.4, -0.4, 5.0, -5.0, 9.7])
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    limit = _auto_limit(0.05)
    torch.testing.assert_close(
        model.splats["mu_t"], torch.tensor([0.0, 0.4, -0.4, limit, -limit, limit]), rtol=1e-5, atol=1e-6
    )


def test_the_bound_is_exactly_where_the_prefilter_would_discard_it() -> None:
    """At the bound, the weight at the nearest timestamp equals ``prefilter_threshold`` exactly."""
    sigma_t = 0.05
    threshold = 0.05
    mu_t = _auto_limit(sigma_t, threshold)
    # Nearest timestamp the sequence contains, gated exactly as the renderer gates it.
    w_t = math.exp(-0.5 * (((0.5) - mu_t) / _guarded_width(sigma_t)) ** 2)
    assert w_t == pytest.approx(threshold, rel=1e-6)


def test_the_bound_scales_with_each_primitive_s_own_extent() -> None:
    """The bound is per primitive: a broad primitive may sit further outside the span than a sharp one."""
    model = FourDGaussians.from_random(n_points=2, sh_degree=1)
    model.splats["mu_t"].data = torch.tensor([9.0, 9.0])
    model.splats["scales_4d_log"].data[:, 3] = torch.log(torch.tensor([0.02, 1.0]))
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    got = model.splats["mu_t"]
    assert got[0].item() == pytest.approx(_auto_limit(0.02), rel=1e-5)
    assert got[1].item() == pytest.approx(_auto_limit(1.0), rel=1e-5)
    assert got[1] > got[0]


def test_clamp_leaves_a_healthy_population_bit_identical() -> None:
    """Means inside the sequence are untouched, exactly — the spatial arm must not be perturbed."""
    model = _mu_t_model([-0.49, -0.2, 0.0, 0.3, 0.49])
    before = model.splats["mu_t"].clone()
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    torch.testing.assert_close(model.splats["mu_t"], before, rtol=0, atol=0)


def test_zero_margin_pins_means_inside_the_sequence() -> None:
    """``0`` is the strict policy: no primitive may sit outside the observed span at all."""
    model = _mu_t_model([0.9, -0.9, 0.1], margin=0.0)
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    torch.testing.assert_close(model.splats["mu_t"], torch.tensor([0.5, -0.5, 0.1]), rtol=1e-6, atol=0)


def test_a_large_margin_is_how_the_bound_is_ablated() -> None:
    """The pre-fix behaviour must remain reachable so the fix itself can be measured against it."""
    model = _mu_t_model([50.0, -50.0], margin=1e6)
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    torch.testing.assert_close(model.splats["mu_t"], torch.tensor([50.0, -50.0]), rtol=0, atol=0)


def test_clamp_touches_only_the_temporal_mean() -> None:
    """scales_4d_log and means are governed by their own projections, not by this one."""
    model = _mu_t_model([9.0, -9.0])
    scales_before = model.splats["scales_4d_log"].clone()
    means_before = model.splats["means"].clone()
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    torch.testing.assert_close(model.splats["scales_4d_log"], scales_before, rtol=0, atol=0)
    torch.testing.assert_close(model.splats["means"], means_before, rtol=0, atol=0)


def test_clamp_is_idempotent() -> None:
    """Runs after every optimizer step, so repeated application must be a fixed point."""
    model = _mu_t_model([0.0, 9.7, -9.7])
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    once = model.splats["mu_t"].clone()
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    torch.testing.assert_close(model.splats["mu_t"], once, rtol=0, atol=0)


def test_clamp_preserves_gradient_flow() -> None:
    """The projection edits ``.data``; the parameter must stay a differentiable leaf."""
    model = _mu_t_model([9.0])
    model.clamp_temporal_mean(model.mu_t_margin_sigmas, model.prefilter_threshold)
    mu_t = model.splats["mu_t"]
    assert mu_t.requires_grad
    assert mu_t.is_leaf
    mu_t.sum().backward()
    assert torch.isfinite(mu_t.grad).all()


def test_random_init_places_means_inside_the_sequence() -> None:
    """``from_random`` draws ``mu_t`` in ``[-0.5, 0.5]`` like ``from_points``."""
    mu_t = FourDGaussians.from_random(n_points=512, sh_degree=1).splats["mu_t"]
    assert mu_t.min() >= -0.5
    assert mu_t.max() <= 0.5
    assert mu_t.mean().abs() < 0.1


def _mass_params(mu_t: list[float], sigma_t: list[float]) -> dict[str, torch.nn.Parameter]:
    n = len(mu_t)
    # Identity quaternions: the marginal width is the intrinsic one plus epsilon.
    ident = torch.zeros(n, 4)
    ident[:, 0] = 1.0
    return {
        "opacities": torch.nn.Parameter(torch.zeros(n)),  # sigmoid(0) = 0.5
        "mu_t": torch.nn.Parameter(torch.tensor(mu_t)),
        "scales_4d_log": torch.nn.Parameter(
            torch.stack([torch.zeros(n)] * 3 + [torch.log(torch.tensor(sigma_t))], dim=-1)
        ),
        "quats_l": torch.nn.Parameter(ident),
        "quats_r": torch.nn.Parameter(ident.clone()),
    }


def _untruncated_mass(params: dict[str, torch.nn.Parameter]) -> torch.Tensor:
    """Published 4DGS aliveness ``alpha · width · sqrt(2π)``, the reference ``observed_mass`` truncates."""
    alpha = torch.sigmoid(params["opacities"].flatten())
    return alpha * marginal_temporal_std(params) * math.sqrt(2.0 * math.pi)
