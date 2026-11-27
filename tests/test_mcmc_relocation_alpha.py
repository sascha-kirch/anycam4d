"""MCMC relocation preserves composited alpha in 4D.

gsplat's binomial identity assumes siblings share one rendered alpha; relocation copies ``mu_t`` and
``sigma_t``, so siblings share ``w_t`` at every tau.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.trainer.optimizers.group import DenseAdamGroup


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="compute_relocation is CUDA-only")

# Import the renderer's epsilon; never re-type it.
_WT_EPS = TEMPORAL_VARIANCE_EPS


def alpha_at(params: dict[str, nn.Parameter], tau: float) -> torch.Tensor:
    """Rendered alpha ``sigmoid(opacity) · w_t(tau)``, with the renderer's epsilon."""
    opac = torch.sigmoid(params["opacities"].detach())
    mu_t = params["mu_t"].detach()
    cov_t_marg = torch.exp(2.0 * params["scales_4d_log"].detach()[:, 3])
    dt = tau - mu_t
    w_t = torch.exp(-0.5 * dt * dt / (cov_t_marg + _WT_EPS))
    return opac * w_t


def composite(alphas: torch.Tensor) -> torch.Tensor:
    """Composited alpha over a group: ``1 - prod(1 - alpha_k)``."""
    return 1.0 - torch.prod(1.0 - alphas)


def _make_params(n: int, opacity_val: float = 0.0) -> tuple[dict, DenseAdamGroup]:
    quats_init = torch.zeros(n, 4, device=DEVICE)
    quats_init[:, 0] = 1.0
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3, device=DEVICE)),
        "mu_t": nn.Parameter(torch.rand(n, device=DEVICE) - 0.5),
        "quats_l": nn.Parameter(quats_init.clone()),
        "quats_r": nn.Parameter(quats_init.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -1.0, device=DEVICE)),
        "opacities": nn.Parameter(torch.full((n,), opacity_val, device=DEVICE)),
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    return params, DenseAdamGroup(optimizers)


def test_relocation_identity_residual() -> None:
    """1 - (1 - sigma_new)^ratio == sigma for sigma_new = 1 - (1 - sigma)^(1/ratio)."""
    sigmas = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
    max_residual = 0.0
    for ratio in range(2, 9):
        sigma_new = 1.0 - (1.0 - sigmas) ** (1.0 / ratio)
        recomposed = 1.0 - (1.0 - sigma_new) ** ratio
        residual = (recomposed - sigmas).abs().max().item()
        max_residual = max(max_residual, residual)
        assert residual < 1e-12, f"ratio={ratio}: identity residual {residual}"
    assert max_residual < 1e-12
