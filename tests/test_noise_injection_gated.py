"""Gated MCMC noise equals the full-population formula, and ``empty_cache_on_refine`` gates ``empty_cache``."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch import nn

from anycam4d.strategy._ops import NOISE_GATE_EPS
from anycam4d.strategy._ops import _op_sigmoid
from anycam4d.strategy._ops import inject_noise_spatial
from anycam4d.math.covariance import intrinsic_spatial_from_rot4d
from anycam4d.enums import SpatialTrigger
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _params(n: int, seed: int = 0, opac_fill: float | None = None) -> dict[str, nn.Parameter]:
    """Params with half low-opacity (gate ~1) and half high; ``opac_fill`` sets every logit."""
    torch.manual_seed(seed)
    quats = torch.randn(n, 4, device=DEVICE)  # non-identity so cov_3d_marg is a real full 3x3
    opac = torch.full((n,), 4.0, device=DEVICE)
    opac[: n // 2] = -8.0
    if opac_fill is not None:
        opac = torch.full((n,), opac_fill, device=DEVICE)
    return {
        "means": nn.Parameter(torch.randn(n, 3, device=DEVICE)),
        "mu_t": nn.Parameter(torch.rand(n, device=DEVICE) - 0.5),
        "quats_l": nn.Parameter(quats.clone()),
        "quats_r": nn.Parameter(quats.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -1.0, device=DEVICE)),
        "opacities": nn.Parameter(opac),
    }


@torch.no_grad()
def _full_spatial_inject(params: dict[str, nn.Parameter], scaler: float) -> None:
    """Ungated full-population spatial noise, applied in place like the real injector."""
    opacities = torch.sigmoid(params["opacities"].flatten())
    sigma_xyz = intrinsic_spatial_from_rot4d(
        params["quats_l"], params["quats_r"], params["scales_4d_log"]
    )  # [N, 3, 3] — matches the injector's default arm
    noise = torch.randn_like(params["means"])
    gate = _op_sigmoid(1.0 - opacities).unsqueeze(-1)
    params["means"].data.add_(gate * scaler * torch.einsum("bij,bj->bi", sigma_xyz, noise))


@requires_cuda
def test_gated_noise_matches_full_spatial() -> None:
    """Gated ``inject_noise_spatial`` is allclose to the full-population path (mixed gate)."""
    scaler, seed = 5e5, 123
    gated = _params(64, seed=1)
    full = _params(64, seed=1)

    torch.manual_seed(seed)
    inject_noise_spatial(gated, {}, scaler)
    torch.manual_seed(seed)
    _full_spatial_inject(full, scaler)

    assert torch.allclose(gated["means"].data, full["means"].data)
    # sanity: the mixed fixture actually skipped rows (not a vacuous all-active pass)
    opac = torch.sigmoid(gated["opacities"].flatten())
    assert (_op_sigmoid(1.0 - opac) <= NOISE_GATE_EPS).any()


@requires_cuda
def test_gated_noise_all_active_equals_full() -> None:
    """Every gate > eps ⇒ no row skipped ⇒ gated == full exactly (same formula, full domain)."""
    scaler, seed = 5e5, 7
    gated = _params(48, seed=2, opac_fill=-8.0)  # gate ~1 everywhere
    full = _params(48, seed=2, opac_fill=-8.0)
    opac = torch.sigmoid(gated["opacities"].flatten())
    assert (_op_sigmoid(1.0 - opac) > NOISE_GATE_EPS).all()

    torch.manual_seed(seed)
    inject_noise_spatial(gated, {}, scaler)
    torch.manual_seed(seed)
    _full_spatial_inject(full, scaler)
    assert torch.equal(gated["means"].data, full["means"].data)


@requires_cuda
def test_gated_noise_empty_active_is_noop() -> None:
    """Every gate < eps ⇒ empty idx ⇒ no parameter move and no crash."""
    params = _params(48, seed=3, opac_fill=10.0)  # sigmoid≈1 ⇒ gate≈0 everywhere
    opac = torch.sigmoid(params["opacities"].flatten())
    assert (_op_sigmoid(1.0 - opac) <= NOISE_GATE_EPS).all()

    before = params["means"].data.clone()
    inject_noise_spatial(params, {}, 5e5)
    assert torch.equal(params["means"].data, before)


def test_empty_cache_flag_gates_call() -> None:
    """``empty_cache_on_refine`` decides whether a refine calls ``torch.cuda.empty_cache``."""
    n = 8

    def _cpu_params() -> tuple[dict[str, nn.Parameter], DenseAdamGroup]:
        torch.manual_seed(0)
        quats = torch.zeros(n, 4)
        quats[:, 0] = 1.0
        params: dict[str, nn.Parameter] = {
            "means": nn.Parameter(torch.randn(n, 3)),
            "mu_t": nn.Parameter(torch.rand(n) - 0.5),
            "quats_l": nn.Parameter(quats.clone()),
            "quats_r": nn.Parameter(quats.clone()),
            "scales_4d_log": nn.Parameter(torch.full((n, 4), -3.0)),
            "opacities": nn.Parameter(torch.zeros(n)),
        }
        opts = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
        return params, DenseAdamGroup(opts)

    def _run(flag: bool) -> int:
        params, group = _cpu_params()
        strat = DefaultSpatialStrategy(
            spatial_trigger=SpatialTrigger.SCREENSPACE,
            refine_start_iter=0,
            refine_stop_iter=100,
            refine_every=1,
            empty_cache_on_refine=flag,
        )
        state = strat.initialize_state()
        means2d = torch.randn(1, n, 2, requires_grad=True)
        means2d.retain_grad()
        info = {"width": 64, "height": 64, "n_cameras": 1, "radii": torch.ones(1, n, 2), "means2d": means2d}
        strat.step_pre_backward(params=params, optimizers=group, state=state, step=1, info=info)
        (means2d.sum() * 0.0 + params["means"].sum()).backward()
        with mock.patch("torch.cuda.empty_cache") as ec:
            strat.step_post_backward(params=params, optimizers=group, state=state, step=1, info=info)
            return ec.call_count

    assert _run(flag=True) >= 1
    assert _run(flag=False) == 0
