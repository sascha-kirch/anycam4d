"""Densification ops keep the sparse optimizer's Adam moments row-aligned: children start cold, other rows keep theirs."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anycam4d.strategy._ops import relocate_spatial
from anycam4d.strategy._ops import sample_add_spatial
from anycam4d.strategy._ops import split_full_4d
from anycam4d.strategy._ops import split_spatial
from anycam4d.strategy.mcmc import _build_binoms
from anycam4d.trainer.optimizers.group import SpatioTemporalSparseAdamGroup
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamReference


BETAS = (0.9, 0.999)

# MCMC ops (relocate / sample_add) call gsplat's compute_relocation, a CUDA-only kernel.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="compute_relocation is CUDA-only")


def _make_group(
    n: int, seed: int = 0, device: str = "cpu"
) -> tuple[dict[str, nn.Parameter], SpatioTemporalSparseAdamGroup]:
    """Build a FourDGaussians-schema param dict + sparse optimizer manager."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1.0
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3, device=dev)),
        "mu_t": nn.Parameter(torch.rand(n, device=dev) - 0.5),
        "quats_l": nn.Parameter(quats.clone()),
        "quats_r": nn.Parameter(quats.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -3.0, device=dev)),
        "opacities": nn.Parameter(torch.zeros(n, device=dev)),
    }
    opts = {
        name: SpatioTemporalSparseAdamReference([{"params": p, "lr": 1e-2, "name": name}], lr=1e-2, betas=BETAS)
        for name, p in params.items()
    }
    group = SpatioTemporalSparseAdamGroup(opts)
    return params, group


def _populate_moments(params: dict[str, nn.Parameter], group: SpatioTemporalSparseAdamGroup) -> None:
    """Run one active step with per-row-distinct grads so every moment row is unique and non-zero."""
    n = params["means"].shape[0]
    for p in params.values():
        row = (torch.arange(n, dtype=p.dtype, device=p.device) + 1.0).view(n, *([1] * (p.dim() - 1)))
        p.grad = torch.ones_like(p) * row
    group.step(visibility=torch.ones(n, dtype=torch.bool, device=params["means"].device))
    for p in params.values():
        p.grad = None


def _moment(group: SpatioTemporalSparseAdamGroup, name: str, key: str) -> torch.Tensor:
    """Read a live moment tensor (``exp_avg`` / ``exp_avg_sq``) for a managed parameter."""
    opt = group.optimizers[name]
    return opt.state[opt.param_groups[0]["params"][0]][key]


@pytest.mark.parametrize("split", [split_spatial, split_full_4d], ids=["spatial", "full_4d"])
def test_split_children_start_cold_survivors_keep_moments(split) -> None:  # noqa: ANN001
    """Survivors keep their moment rows in order; the N children per parent start at zero."""
    n, n_children = 6, 2
    params, group = _make_group(n)
    _populate_moments(params, group)
    before = {k: _moment(group, "means", k).clone() for k in ("exp_avg", "exp_avg_sq")}
    mask = torch.zeros(n, dtype=torch.bool)
    mask[[1, 4]] = True

    split(params, group, {}, mask, n_children)

    rest = torch.where(~mask)[0]
    for key, old in before.items():
        now = _moment(group, "means", key)
        assert now.shape[0] == params["means"].shape[0] == len(rest) + 2 * n_children
        assert torch.equal(now[: len(rest)], old[rest])
        assert not now[len(rest) :].any()


@requires_cuda
def test_relocate_zeroes_dead_and_source_rows_only() -> None:
    """Relocated (dead) rows and their sampled sources restart cold; every other row keeps its moments."""
    params, group = _make_group(6, device="cuda")
    with torch.no_grad():
        params["opacities"][:2] = -10.0  # dead (rows 0, 1)
        params["opacities"][2:] = 4.0
    _populate_moments(params, group)
    before = _moment(group, "means", "exp_avg").clone()
    mask = torch.sigmoid(params["opacities"].flatten()) <= 0.005

    relocate_spatial(params, group, {}, mask, _build_binoms().cuda())

    now = _moment(group, "means", "exp_avg")
    zeroed = ~now.any(dim=1)
    assert zeroed[:2].all()
    assert torch.equal(now[~zeroed], before[~zeroed])
    assert now.shape[0] == params["means"].shape[0] == 6


@requires_cuda
def test_sample_add_appends_cold_rows() -> None:
    """Added children are appended with zero moments; existing rows are untouched."""
    params, group = _make_group(6, device="cuda")
    with torch.no_grad():
        params["opacities"].fill_(4.0)
    _populate_moments(params, group)
    before = _moment(group, "means", "exp_avg").clone()

    sample_add_spatial(params, group, {}, 3, _build_binoms().cuda())

    now = _moment(group, "means", "exp_avg")
    assert now.shape[0] == params["means"].shape[0] == 9
    assert torch.equal(now[:6], before)
    assert not now[6:].any()
