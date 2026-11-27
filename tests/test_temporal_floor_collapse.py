"""Why the temporal-variance epsilon exists: without it a temporally collapsed primitive cannot recover.

Every escape gradient (``mu_t``, ``sigma_t``, opacity) carries ``w_t``, which underflows to exactly zero;
the prefilter cuts it off sooner; densification never touches it. The epsilon keeps ``w_t`` O(1).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from anycam4d.config.model import ModelConfig
from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.fourdgaussians.conditioning import condition_at_t
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.strategy._ops import split_full_4d
from anycam4d.trainer.optimizers.group import DenseAdamGroup


# The floor is 0.30 ticks at 96 frames, 0.95 at 300, 1.71 at 540; these tests use 300.
N_FRAMES = 300
DELTA = 1.0 / N_FRAMES
FLOOR = TEMPORAL_SIGMA_FLOOR


def _isotropic_model(scale_t_log: float, n: int = 4) -> FourDGaussians:
    """Population whose marginal temporal width is exactly ``exp(scale_t_log)``: identity rotation, equal scales."""
    model = FourDGaussians.from_random(n_points=n, sh_degree=0)
    with torch.no_grad():
        model.splats["quats_l"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.splats["quats_r"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model.splats["scales_4d_log"][:] = scale_t_log
        model.splats["mu_t"][:] = 0.0
    return model


def test_the_gate_stops_moving_once_the_parameter_is_under_the_floor() -> None:
    """Four decades of collapse below the floor leave the rendered gate unchanged."""
    dt = FLOOR
    # At k decades under, the residual is ~5e-7 at k=3 and unmeasurable beyond.
    gates = [
        condition_at_t(_isotropic_model(math.log(FLOOR / k)), tau=dt).temporal_weights_raw[0].item()
        for k in (1e3, 1e5, 1e7)
    ]
    assert gates[0] == pytest.approx(gates[-1], abs=1e-6)
    assert gates[0] == pytest.approx(math.exp(-0.5), rel=1e-5), "the floor alone sets the width"


def test_the_prefilter_mask_rate_is_pinned_by_the_floor() -> None:
    """A collapsed primitive is prefiltered exactly like one at the floor."""
    threshold = 0.05
    reach = FLOOR * math.sqrt(-2.0 * math.log(threshold))  # dt at which the gate hits the threshold
    for scale_t_log in (math.log(FLOOR / 1e4), math.log(FLOOR / 1e8)):
        model = _isotropic_model(scale_t_log)
        # ``temporal_weights_raw`` is deliberately unmasked (strategies read it); the prefilter
        # shows up in the composited opacity, which is what the rasterizer receives.
        inside = condition_at_t(model, tau=reach * 0.9, prefilter_threshold=threshold)
        outside = condition_at_t(model, tau=reach * 1.1, prefilter_threshold=threshold)
        assert (inside.opacity_cond > 0).all(), "culled before the floor's own reach"
        assert (outside.opacity_cond == 0).all(), "survived past the floor's own reach"


def test_the_mu_t_margin_bottoms_out_at_the_floor_the_renderer_draws() -> None:
    """The ``mu_t`` margin converges to ``0.5 + k·sqrt(eps)`` as ``sigma_t`` collapses; it never grows."""
    limits = []
    for scale_t_log in (math.log(FLOOR * 10), math.log(FLOOR), math.log(FLOOR / 1e6)):
        model = _isotropic_model(scale_t_log, n=2)
        with torch.no_grad():
            model.splats["mu_t"][:] = 10.0  # far outside; the clamp lands it exactly on the limit
        model.clamp_temporal_mean(-1.0, 0.05)
        limits.append(model.splats["mu_t"][0].item())

    k = math.sqrt(-2.0 * math.log(0.05))
    assert limits[0] > limits[1] > limits[2], "the allowance must shrink with sigma_t, not grow"
    assert limits[-1] == pytest.approx(0.5 + k * FLOOR, rel=1e-6), "the floored width sets the bound"
    assert all(limit > 0.5 for limit in limits), "the margin never cuts inside the sequence itself"


def test_repeated_splits_cannot_reach_a_non_finite_scale() -> None:
    """A lineage split past float32 ``exp`` underflow keeps a finite log scale."""
    n_gauss = 2
    params = nn.ParameterDict(
        {
            "means": nn.Parameter(torch.zeros(n_gauss, 3)),
            "mu_t": nn.Parameter(torch.zeros(n_gauss)),
            "quats_l": nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_gauss, 1)),
            "quats_r": nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n_gauss, 1)),
            "scales_4d_log": nn.Parameter(torch.full((n_gauss, 4), -100.0)),
            "opacities": nn.Parameter(torch.zeros(n_gauss)),
            "sh0": nn.Parameter(torch.zeros(n_gauss, 1, 3)),
            "shN": nn.Parameter(torch.zeros(n_gauss, 0, 3)),
        }
    )
    group = DenseAdamGroup({k: torch.optim.Adam([v], lr=0.0) for k, v in params.items()})

    # Split the *last* primitive each round: children are appended, so this follows one lineage
    # down rather than repeatedly splitting a survivor that never moves.
    for _ in range(12):
        mask = torch.zeros(params["scales_4d_log"].shape[0], dtype=torch.bool)
        mask[-1] = True
        split_full_4d(params, group, {}, mask, N=2, sigma_t_floor=0.0)  # floor off — the case under test

    scale_t_log = params["scales_4d_log"][:, 3]
    assert torch.isfinite(scale_t_log).all(), "log(exp(x)/d) round-trip produced a non-finite scale"
    assert (scale_t_log < -103.3).any(), "the lineage never crossed the float32 underflow boundary"
    assert torch.isfinite(torch.exp(scale_t_log)).all()


def _grads_at(sigma_t: float, *, eps: float, off_tick: float, prefilter: float = 0.0) -> tuple[float, float]:
    """Gradients reaching ``mu_t`` and ``log sigma_t`` through the bare gate for a primitive ``off_tick`` from a tick."""
    log_s = torch.tensor([math.log(sigma_t)], requires_grad=True)
    mu_t = torch.tensor([off_tick], requires_grad=True)
    w_t = torch.exp(-0.5 * mu_t**2 / (torch.exp(2 * log_s) + eps))
    if prefilter > 0.0:
        w_t = w_t * (w_t >= prefilter).float()  # exactly what conditioning.py does
    ((1.0 - w_t) ** 2).backward()
    return mu_t.grad.item(), log_s.grad.item()


def test_without_the_epsilon_an_off_tick_collapse_is_an_absorbing_state() -> None:
    """Without the epsilon, every gradient of an off-tick collapsed primitive is exactly zero."""
    off_tick = DELTA / 2.0  # worst placement: exactly between two training timestamps

    assert _grads_at(0.02 * DELTA, eps=0.0, off_tick=off_tick) == (0.0, 0.0), "expected a dead parameter"

    weak, _ = _grads_at(0.1 * DELTA, eps=0.0, off_tick=off_tick)
    floored, _ = _grads_at(0.1 * DELTA, eps=TEMPORAL_VARIANCE_EPS, off_tick=off_tick)
    assert abs(floored) > 100 * abs(weak), "the epsilon must keep mu_t steerable, not merely finite"

    # With the epsilon the gradient does not narrow as the collapse deepens (5% band).
    deep, _ = _grads_at(1e-6 * DELTA, eps=TEMPORAL_VARIANCE_EPS, off_tick=off_tick)
    assert deep == pytest.approx(floored, rel=0.05)


def test_the_prefilter_is_the_cliff_not_a_second_guard() -> None:
    """Crossing the prefilter zeroes the gradient at once; the boundary sits at ``sigma_t = 0.204·Δ``."""
    threshold = ModelConfig().prefilter_threshold
    off_tick = DELTA / 2.0
    cliff = 0.5 / math.sqrt(-2.0 * math.log(threshold))  # = 0.204, in ticks

    above, _ = _grads_at(1.2 * cliff * DELTA, eps=0.0, off_tick=off_tick, prefilter=threshold)
    below, _ = _grads_at(0.9 * cliff * DELTA, eps=0.0, off_tick=off_tick, prefilter=threshold)
    assert abs(above) > 1.0, "a primitive just inside the prefilter must still be steerable"
    assert below == 0.0, "the prefilter zeroes the gradient outright — it cannot rescue anything"

    # The shipped epsilon clears the cliff on sequences of 65 frames or more.
    for n_frames, expected_ticks in ((96, 0.304), (300, 0.949), (540, 1.708)):
        ticks = TEMPORAL_SIGMA_FLOOR * n_frames
        assert ticks == pytest.approx(expected_ticks, rel=1e-2)
        assert ticks > cliff, f"{n_frames}-frame sequence is under-floored"
    assert TEMPORAL_SIGMA_FLOOR * 64 < cliff, "the crossover is around 65 frames"
