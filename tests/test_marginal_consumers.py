"""``opacity_reg``, ``clamp_temporal_mean`` and ``observed_mass`` read the marginal width the renderer draws.

``cov_4d[3, 3]`` combines all four squared scales, so a rotation can put a spatial axis on the time row.
Each test pairs an axis-aligned case (nothing moved) with that rotated case (moves as geometry predicts).
"""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.losses.regularization import opacity_reg
from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


PREFILTER = 0.05
K = math.sqrt(-2.0 * math.log(PREFILTER))

# q_l = (0, 0, 0, 1), q_r = (1, 0, 0, 0) puts the first spatial axis on the time row: cov_t_marg = exp(2 s_x).
_TIME_ROW_ON_X_L = [0.0, 0.0, 0.0, 1.0]
_TIME_ROW_ON_X_R = [1.0, 0.0, 0.0, 0.0]
_IDENTITY = [1.0, 0.0, 0.0, 0.0]


def _model(
    *,
    scale_t: list[float],
    scale_x: list[float],
    mu_t: list[float],
    time_row_on_x: bool,
) -> FourDGaussians:
    """Population whose rendered temporal width is ``scale_t`` (identity) or ``scale_x`` (``time_row_on_x``)."""
    n = len(scale_t)
    model = FourDGaussians.from_random(n_points=n, sh_degree=0, mu_t_margin_sigmas=-1.0)
    with torch.no_grad():
        model.splats["mu_t"].copy_(torch.tensor(mu_t))
        model.splats["opacities"].zero_()  # sigmoid = 0.5
        model.splats["scales_4d_log"][:, 0] = torch.log(torch.tensor(scale_x))
        model.splats["scales_4d_log"][:, 1:3] = math.log(1e-3)
        model.splats["scales_4d_log"][:, 3] = torch.log(torch.tensor(scale_t))
        left, right = (_TIME_ROW_ON_X_L, _TIME_ROW_ON_X_R) if time_row_on_x else (_IDENTITY, _IDENTITY)
        model.splats["quats_l"].copy_(torch.tensor(left).expand(n, 4))
        model.splats["quats_r"].copy_(torch.tensor(right).expand(n, 4))
    return model


def _guarded(width: float) -> float:
    """``sqrt(width^2 + eps)`` — the width as the renderer's gate sees it."""
    return math.sqrt(width * width + TEMPORAL_VARIANCE_EPS)


def test_the_mu_t_bound_is_unchanged_where_the_two_widths_agree() -> None:
    """An axis-aligned primitive gets the same ``mu_t`` bound as with the intrinsic scale."""
    model = _model(scale_t=[0.05], scale_x=[0.4], mu_t=[9.0], time_row_on_x=False)
    model.clamp_temporal_mean(-1.0, PREFILTER)
    assert model.splats["mu_t"][0].item() == pytest.approx(0.5 + K * _guarded(0.05), rel=1e-6)


def test_the_mu_t_bound_moves_in_both_directions_with_the_rendered_width() -> None:
    """The bound widens for a marginal wider than the parameter and tightens for a narrower one."""
    wide_x, narrow_x = 0.4, 0.002
    model = _model(scale_t=[0.05, 0.05], scale_x=[wide_x, narrow_x], mu_t=[9.0, 9.0], time_row_on_x=True)
    model.clamp_temporal_mean(-1.0, PREFILTER)
    got = model.splats["mu_t"]

    intrinsic_bound = 0.5 + K * 0.05
    assert got[0].item() == pytest.approx(0.5 + K * _guarded(wide_x), rel=1e-5)
    assert got[1].item() == pytest.approx(0.5 + K * _guarded(narrow_x), rel=1e-5)
    assert got[0].item() > intrinsic_bound, "a wider rendered width must buy a looser bound"
    assert got[1].item() < intrinsic_bound, "a narrower rendered width must impose a tighter one"


def test_a_temporally_collapsed_primitive_keeps_the_bound_the_renderer_justifies() -> None:
    """A collapsed primitive keeps a limit wider than 0.5, because the renderer draws it at ``sqrt(eps)``."""
    model = _model(scale_t=[1e-12], scale_x=[1e-12], mu_t=[9.0], time_row_on_x=False)
    model.clamp_temporal_mean(-1.0, PREFILTER)
    limit = model.splats["mu_t"][0].item()
    assert math.isfinite(limit)
    assert limit == pytest.approx(0.5 + K * TEMPORAL_SIGMA_FLOOR, rel=1e-5)
    assert limit > 0.5


def test_the_opacity_penalty_is_unchanged_where_the_two_widths_agree() -> None:
    """Where the two widths agree, the shipped regularizer reduces to its single-width closed form."""
    scale_t, mu_t, tau = 0.05, 0.0, 0.12
    model = _model(scale_t=[scale_t], scale_x=[0.4], mu_t=[mu_t], time_row_on_x=False)

    got = opacity_reg(dict(model.splats), tau)
    old = 0.5 * math.exp(-0.5 * (tau - mu_t) ** 2 / (scale_t * scale_t + TEMPORAL_VARIANCE_EPS))
    assert got.item() == pytest.approx(old, rel=1e-6)


def test_the_opacity_penalty_follows_the_rendered_width_not_the_parameter() -> None:
    """Same parameters, different rotation: the wider-rendered primitive pays more."""
    tau = 0.12
    common = {"scale_t": [0.05], "mu_t": [0.0]}
    aligned = opacity_reg(dict(_model(**common, scale_x=[0.4], time_row_on_x=False).splats), tau)
    wider = opacity_reg(dict(_model(**common, scale_x=[0.4], time_row_on_x=True).splats), tau)
    narrower = opacity_reg(dict(_model(**common, scale_x=[0.002], time_row_on_x=True).splats), tau)

    assert wider.item() > aligned.item(), "a primitive present at tau must pay for that presence"
    assert narrower.item() < aligned.item(), "a primitive the renderer barely draws at tau must pay less"


def test_the_opacity_penalty_now_reaches_the_quaternions() -> None:
    """The opacity penalty has a gradient path into the quaternions."""
    # A generic rotation: the axis-aligned fixture is a stationary point with zero quaternion gradient.
    torch.manual_seed(0)
    model = _model(scale_t=[0.05], scale_x=[0.4], mu_t=[0.0], time_row_on_x=False)
    with torch.no_grad():
        model.splats["quats_l"].copy_(torch.randn(1, 4))
        model.splats["quats_r"].copy_(torch.randn(1, 4))
    params = dict(model.splats)
    opacity_reg(params, 0.12).backward()

    for key in ("quats_l", "quats_r", "scales_4d_log", "opacities", "mu_t"):
        grad = params[key].grad
        assert grad is not None, f"{key} received no gradient"
        assert torch.isfinite(grad).all(), f"{key} gradient is not finite"
    assert params["quats_l"].grad.abs().sum() > 0, "the rotation must be a live degree of freedom"
    assert params["quats_r"].grad.abs().sum() > 0
