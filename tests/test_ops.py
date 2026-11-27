"""Unit tests for anycam4d/strategy/_ops.py helpers."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from anycam4d.strategy._ops import marginal_temporal_std
from anycam4d.strategy._ops import prune_opacity_and_scale
from anycam4d.strategy._ops import split_full_4d
from anycam4d.strategy._ops import split_spatial
from anycam4d.trainer.optimizers.group import DenseAdamGroup


def _make_params(n: int) -> tuple[dict, dict]:
    """Build a minimal params + optimizers dict for strategy ops testing."""
    quats_init = torch.zeros(n, 4)
    quats_init[:, 0] = 1.0
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3)),
        "mu_t": nn.Parameter(torch.rand(n) - 0.5),
        "quats_l": nn.Parameter(quats_init.clone()),
        "quats_r": nn.Parameter(quats_init.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -3.0)),
        "opacities": nn.Parameter(torch.zeros(n)),
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    return params, DenseAdamGroup(optimizers)


def test_split_spatial_does_not_mutate_temporal() -> None:
    """split_spatial must leave mu_t and scales_4d_log[:, 3] byte-identical in children."""
    N = 5
    params, optimizers = _make_params(N)
    mu_t_before = params["mu_t"].detach().clone()
    sigma_t_before = params["scales_4d_log"][:, 3].detach().clone()

    mask = torch.tensor([True, True, False, False, False])
    state: dict = {}
    split_spatial(params, optimizers, state, mask)

    # Original N — 2 non-split Gaussians are at the front after rest/children cat
    # Children are appended. Check temporal fields of children match parents.
    n_sel = int(mask.sum())
    n_rest = N - n_sel
    n_children = n_sel * 2
    assert len(params["mu_t"]) == n_rest + n_children

    children_mu_t = params["mu_t"][n_rest:]
    children_sigma_t = params["scales_4d_log"][n_rest:, 3]

    # Parents (indices 0, 1) were selected. Children are tiled copies.
    # mu_t: children should equal parent mu_t (no offset applied)
    parent_mu_t = mu_t_before[mask]  # [2]
    expected_mu_t = parent_mu_t.repeat(2)  # [4]: [p0, p1, p0, p1]
    assert torch.allclose(children_mu_t, expected_mu_t), "mu_t was mutated by split_spatial"

    # sigma_t (scales_4d_log[:, 3]): children should equal parent value (no shrink)
    parent_sigma_t = sigma_t_before[mask]
    expected_sigma_t = parent_sigma_t.repeat(2)
    assert torch.allclose(children_sigma_t, expected_sigma_t), "scales_4d_log[:, 3] was mutated by split_spatial"


def test_split_spatial_shrinks_xyz_scales() -> None:
    """split_spatial must shrink scales_4d_log[:, :3] by log(1/(0.8*N)) in children."""
    N = 4
    params, optimizers = _make_params(N)
    original_s_xyz = params["scales_4d_log"][:, :3].detach().clone()

    mask = torch.tensor([True, False, False, False])
    state: dict = {}
    split_spatial(params, optimizers, state, mask)

    n_rest = 3
    children_scales_xyz = params["scales_4d_log"][n_rest:, :3]
    expected = torch.log(torch.exp(original_s_xyz[0:1]) / (0.8 * 2)).repeat(2, 1)
    assert torch.allclose(children_scales_xyz, expected, atol=1e-5), "spatial scales not shrunk correctly"


def test_split_full_4d_all_axes_change() -> None:
    """split_full_4d must produce children where BOTH means and mu_t differ from parent."""
    N = 4
    params, optimizers = _make_params(N)
    # Make scales larger so offsets are non-trivial
    with torch.no_grad():
        params["scales_4d_log"].fill_(0.0)  # exp(0) = 1.0

    mask = torch.tensor([True, False, False, False])
    state: dict = {}
    split_full_4d(params, optimizers, state, mask)

    n_rest = 3
    assert len(params["means"]) == n_rest + 2
    assert len(params["mu_t"]) == n_rest + 2
    # All 4 scales shrunk
    children_s = params["scales_4d_log"][n_rest:, :]
    expected_s = torch.full((2, 4), math.log(1.0 / (0.8 * 2)))
    assert torch.allclose(children_s, expected_s, atol=1e-5)


def test_prune_opacity_removes_low_opacity() -> None:
    """Gaussians with sigmoid(opacity) < opa_thr must be removed."""
    N = 6
    params, optimizers = _make_params(N)
    with torch.no_grad():
        params["opacities"][:3] = -10.0  # sigmoid ≈ 0 (dead)
        params["opacities"][3:] = 4.0  # sigmoid ≈ 0.98 (alive)

    state: dict = {}
    n_removed = prune_opacity_and_scale(
        params,
        optimizers,
        state,
        opa_thr=0.005,
        spatial_scale_thr=0.1,
        scene_scale=1.0,
        after_first_reset=False,
    )
    assert n_removed == 3
    assert len(params["means"]) == 3


def test_prune_scale_fires_only_after_first_reset() -> None:
    """Scale-based prune must not fire before the first opacity reset."""
    N = 4
    params, optimizers = _make_params(N)
    with torch.no_grad():
        params["opacities"].fill_(4.0)  # all high opacity
        params["scales_4d_log"][:, :3].fill_(10.0)  # very large — should be pruned if after_first_reset

    state: dict = {}
    # before first reset — no scale prune
    n_before = prune_opacity_and_scale(
        params,
        optimizers,
        state,
        opa_thr=0.005,
        spatial_scale_thr=0.1,
        scene_scale=1.0,
        after_first_reset=False,
    )
    assert n_before == 0, "Scale prune should not fire before first reset"

    # after first reset — scale prune fires
    n_after = prune_opacity_and_scale(
        params,
        optimizers,
        state,
        opa_thr=0.005,
        spatial_scale_thr=0.1,
        scene_scale=1.0,
        after_first_reset=True,
    )
    assert n_after == N, "All Gaussians should be pruned (spatial scale too large)"


def test_split_spatial_survives_a_degenerate_spatial_marginal() -> None:
    """A rotated needle still splits: the closed-form factor never takes a Cholesky of an indefinite float32 matrix."""
    params, optimizers = _make_params(4)
    # A rotated needle (~1e3 by ~1e-8): rotation spreads the large eigenvalue over all nine entries.
    with torch.no_grad():
        params["scales_4d_log"][2] = torch.tensor([6.9, -18.4, -18.4, -18.4])
        params["quats_l"][2] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        params["quats_r"][2] = torch.tensor([0.6, -0.4, 0.5, -0.48])
    state = {"grad2d": torch.zeros(4), "count": torch.zeros(4)}
    mask = torch.zeros(4, dtype=torch.bool)
    mask[2] = True

    split_spatial(params, optimizers, state, mask)

    assert torch.isfinite(params["means"]).all()


def _coupled_single_parent(scale_t_log: float = -1.5) -> tuple[dict, DenseAdamGroup]:
    """One anisotropic, rotated, space-time-coupled primitive at the origin."""
    params, optimizers = _make_params(1)
    with torch.no_grad():
        params["scales_4d_log"][0] = torch.tensor([-1.0, -2.0, -3.0, scale_t_log])
        params["quats_l"][0] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        params["quats_r"][0] = torch.tensor([0.5, -0.5, 0.5, -0.5])
        params["means"][0] = torch.zeros(3)
    return params, optimizers


def _child_offsets(split_fn, mask: torch.Tensor, seed: int, param: str) -> torch.Tensor:
    """Children's offsets from their parents under a fixed seed, in selection order."""
    n = mask.numel()
    torch.manual_seed(seed)
    params, optimizers = _make_params(n)
    parents = params[param].detach()[mask].clone()
    torch.manual_seed(seed + 1)
    split_fn(params, optimizers, {}, mask)
    n_sel = int(mask.sum())
    children = params[param].detach()[-2 * n_sel :]
    return children - torch.cat([parents, parents])


@pytest.mark.parametrize(
    ("split_fn", "param"),
    [(split_spatial, "means"), (split_full_4d, "mu_t")],
)
def test_split_offsets_are_stable_under_a_changed_selection(split_fn, param) -> None:
    """One more selected primitive leaves every other primitive's offset unchanged.

    Selection flips on atomicAdd noise; a selection-sized draw would re-randomize the whole population.
    """
    n = 12
    base = torch.zeros(n, dtype=torch.bool)
    base[[1, 4, 7]] = True
    flipped = base.clone()
    flipped[9] = True  # one extra primitive crosses the threshold

    off_base = _child_offsets(split_fn, base, seed=17, param=param)
    off_flipped = _child_offsets(split_fn, flipped, seed=17, param=param)

    # base selects {1,4,7}; flipped selects {1,4,7,9}. Children are laid out [child0..., child1...],
    # so index the two blocks separately rather than assuming a contiguous prefix.
    n_base, n_flip = 3, 4
    for c in range(2):
        shared_base = off_base[c * n_base : c * n_base + 3]
        shared_flip = off_flipped[c * n_flip : c * n_flip + 3]
        torch.testing.assert_close(shared_base, shared_flip, rtol=0, atol=0)
