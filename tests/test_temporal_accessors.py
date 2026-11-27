"""Scalar temporal accessors return ``[N]``: an ``[N, 1]`` broadcasts against ``[N]`` into a plausible ``[N, N]``."""

from __future__ import annotations

import pytest
import torch

from anycam4d.losses.regularization import opacity_reg
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.fourdgaussians.model import FourDGaussians

N = 9


@pytest.fixture
def model() -> FourDGaussians:
    """Population with a distinct temporal scale per primitive, so ``[N, 1]`` broadcasts cannot pass by accident."""
    torch.manual_seed(0)
    model = FourDGaussians.from_random(n_points=N, sh_degree=0)
    model.splats["scales_4d_log"].data = torch.randn(N, 4) - 3.0
    return model


def test_every_scalar_temporal_accessor_is_flat(model: FourDGaussians) -> None:
    assert model.marginal_temporal_variance().shape == (N,)
    assert model.marginal_temporal_std().shape == (N,)
    assert model.intrinsic_spatial_scales().shape == (N, 3)


def test_the_temporal_mean_bound_stays_per_primitive(model: FourDGaussians) -> None:
    """``clamp_temporal_mean`` builds ``limit`` from an accessor; ``[N, 1]`` would clamp against a matrix."""
    model.splats["mu_t"].data.fill_(50.0)
    model.clamp_temporal_mean(margin_sigmas=2.0, prefilter_threshold=0.05)
    expected = 0.5 + 2.0 * model.marginal_temporal_std()
    assert model.splats["mu_t"].shape == (N,)
    torch.testing.assert_close(model.splats["mu_t"].data, expected, rtol=1e-6, atol=0)


def test_the_rendered_opacity_penalty_reduces_over_primitives_not_pairs(model: FourDGaussians) -> None:
    """The opacity penalty reduces row-wise over primitives, not over an ``[N, N]`` gate."""
    got = opacity_reg(dict(model.splats), tau=0.1)

    alpha = torch.sigmoid(model.splats["opacities"])
    cov_t = model.marginal_temporal_variance()
    dt = 0.1 - model.splats["mu_t"]
    row_wise = (alpha * torch.exp(-0.5 * dt * dt / (cov_t + TEMPORAL_VARIANCE_EPS))).mean()
    pairwise = (
        alpha.unsqueeze(-1) * torch.exp(-0.5 * dt.unsqueeze(-1) ** 2 / (cov_t.unsqueeze(0) + TEMPORAL_VARIANCE_EPS))
    ).mean()

    assert got.shape == ()
    torch.testing.assert_close(got, row_wise, rtol=1e-6, atol=0)
    assert not torch.isclose(got, pairwise)
