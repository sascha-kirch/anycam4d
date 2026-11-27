"""MCMC regularizers: rendered opacity manufactures the relocation dead set; L1 scale; birth opacity; the warning."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from anycam4d.config.training import TrainingConfig
from anycam4d.losses.regularization import opacity_reg
from anycam4d.losses.regularization import scale_reg
from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.trainer.trainer import warn_if_mcmc_misregularized


def _axis_aligned_params(opacities: torch.Tensor, mu_t: torch.Tensor, width: torch.Tensor) -> dict:
    """Identity-rotation params whose marginal temporal width equals ``width`` exactly."""
    n = opacities.shape[0]
    scales_4d_log = torch.zeros(n, 4)
    scales_4d_log[:, 3] = torch.log(width)
    ident = torch.zeros(n, 4)
    ident[:, 0] = 1.0
    return {
        "opacities": opacities,
        "mu_t": mu_t,
        "scales_4d_log": scales_4d_log,
        "quats_l": ident,
        "quats_r": ident.clone(),
    }


def _dead_fraction(opacities: torch.Tensor, min_opacity: float = 0.005) -> float:
    return (torch.sigmoid(opacities) < min_opacity).float().mean().item()


def test_opacity_rendered_creates_dead_set() -> None:
    """Minimizing ``opacity_reg`` at random tau drives opacities below ``min_opacity``."""
    torch.manual_seed(0)
    min_opacity = 0.005
    n = 512
    # start well above the dead threshold so the dead fraction is initially ~0
    opacities = torch.nn.Parameter(torch.full((n,), 2.0))
    mu_t = torch.zeros(n)
    sigma_t = torch.full((n,), 0.1)
    opt = torch.optim.Adam([opacities], lr=0.5)
    before = _dead_fraction(opacities, min_opacity)
    for step in range(200):
        opt.zero_grad()
        tau = -0.5 + (step % 20) / 19.0  # sweep the span, as the frame sampler does
        opacity_reg(_axis_aligned_params(opacities, mu_t, sigma_t), tau).backward()
        opt.step()
    after = _dead_fraction(opacities, min_opacity)
    assert before == 0.0
    assert after > 0.0, "rendered opacity reg did not manufacture a dead set"


def test_opacity_rendered_has_no_width_escape_hatch() -> None:
    """The property that retired the tau-blind term: the penalty at a primitive's own centre is
    ``sigmoid(alpha)`` whatever its width, so widening sigma_t buys no discount."""
    opacities = torch.full((4,), 1.0)
    mu_t = torch.zeros(4)
    at_centre = [
        opacity_reg(_axis_aligned_params(opacities, mu_t, torch.full((4,), w)), tau=0.0).item()
        for w in (0.01, 0.1, 1.0, 100.0)
    ]
    assert at_centre == pytest.approx([torch.sigmoid(torch.tensor(1.0)).item()] * 4, rel=1e-4)


def test_scale_l1_is_mean_scale() -> None:
    """scale_reg is the mean of the activated scales; minimizing it shrinks the mean."""
    torch.manual_seed(0)
    scales_raw = torch.randn(256, 3)
    assert torch.allclose(scale_reg(scales_raw), scales_raw.mean())

    log_scales = torch.nn.Parameter(torch.zeros(256, 3))
    opt = torch.optim.Adam([log_scales], lr=0.1)
    before = torch.exp(log_scales).mean().item()
    for _ in range(50):
        opt.zero_grad()
        scale_reg(torch.exp(log_scales)).backward()
        opt.step()
    after = torch.exp(log_scales).mean().item()
    assert after < before


def test_mcmc_reg_warns_when_unregularized() -> None:
    """MCMC + lambda=0 warns (relocation would starve); lambda > 0 is silent."""
    mcmc = MCMCSpatialStrategy(cap_max=1000)

    with pytest.warns(RuntimeWarning):
        warn_if_mcmc_misregularized(mcmc, TrainingConfig(opacity_reg_lambda=0.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_mcmc_misregularized(mcmc, TrainingConfig(opacity_reg_lambda=0.002))


def test_opacity_reg_falls_off_away_from_the_centre() -> None:
    """Off-centre the penalty depends on the width: presence at tau is what is priced."""
    opacities = torch.zeros(1)
    mu_t = torch.full((1,), 0.5)
    narrow = opacity_reg(_axis_aligned_params(opacities, mu_t, torch.full((1,), 0.01)), 0.6).item()
    wide = opacity_reg(_axis_aligned_params(opacities, mu_t, torch.full((1,), 1.0)), 0.6).item()

    assert narrow < 1e-6
    assert wide == pytest.approx(0.5, abs=0.01)


def test_opacity_reg_uses_the_renderer_s_own_temporal_floor() -> None:
    """Below the floor the loss's gate and the renderer's gate agree (both add the epsilon)."""
    opacities = torch.zeros(1)
    mu_t = torch.full((1,), 0.5)
    tau = 0.5 + TEMPORAL_SIGMA_FLOOR
    below = opacity_reg(_axis_aligned_params(opacities, mu_t, torch.full((1,), TEMPORAL_SIGMA_FLOOR / 1000)), tau)
    at_zero = opacity_reg(_axis_aligned_params(opacities, mu_t, torch.zeros(1)), tau)

    assert below.item() == pytest.approx(at_zero.item(), abs=1e-6)
    assert below.item() == pytest.approx(0.5 * math.exp(-0.5), rel=1e-3)


def test_opacity_reg_pushes_opacity_down_only_where_rendered() -> None:
    """Only the primitive visible at tau receives opacity gradient."""
    opacities = torch.zeros(2, requires_grad=True)
    params = _axis_aligned_params(opacities, torch.tensor([0.5, 0.0]), torch.full((2,), 0.02))
    opacity_reg(params, 0.5).backward()

    assert opacities.grad[0] > 1e-3
    assert abs(opacities.grad[1].item()) < 1e-9


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_raising_the_birth_factor_lifts_children_clear_of_the_dead_test(factor: float) -> None:
    """Above 1.0 a freshly born child survives its next dead test; the dead test itself does not move."""
    strat = MCMCSpatialStrategy(cap_max=1000, min_opacity=0.005, birth_opacity_factor=factor)

    assert strat.birth_opacity() == pytest.approx(0.005 * factor)
    assert strat.dead_threshold() == 0.005
    assert not strat.dead_mask(torch.tensor([strat.birth_opacity()])).item()
