"""Tests for MCMCSpatialStrategy and MCMCSpatioTemporalStrategy."""

from __future__ import annotations


import pytest
import torch
from torch import nn

from anycam4d.enums import ParentScore
from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.strategy._ops import relocate_spatial
from anycam4d.strategy._ops import primitive_speed
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A collapsed sigma_t still renders at sqrt(eps), so observed mass bottoms out at ~7.8e-3, not zero;
# thresholds for "collapsed" must sit above that floor.
COLLAPSED_IS_DEAD_BELOW = 0.1


def _make_params(n: int, opacity_val: float = 0.0) -> tuple[dict, dict]:
    """Build a minimal params + optimizers dict for MCMC strategy testing."""
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


def _make_info() -> dict:
    """Build a minimal render info dict (unused by MCMC strategies)."""
    return {}


def test_mcmc_spatial_noop_with_zero_lr() -> None:
    """With noise_lr=0 and refine_stop_iter=0, means must not change."""
    N = 8
    params, optimizers = _make_params(N, opacity_val=2.0)
    means_before = params["means"].detach().clone()

    strategy = MCMCSpatialStrategy(noise_lr=0.0, refine_stop_iter=0)
    state = strategy.initialize_state()
    strategy.step_post_backward(params, optimizers, state, step=1, info=_make_info(), lr=1e-3)

    assert torch.allclose(params["means"], means_before), "means changed with noise_lr=0 and refine_stop_iter=0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute_relocation is CUDA-only")
def test_mcmc_spatial_relocate_preserves_temporal() -> None:
    """Relocated copies keep their parent's mu_t, temporal marginal and velocity, whatever the 4D rotation."""
    N = 6
    params, optimizers = _make_params(N)
    with torch.no_grad():
        params["opacities"][:3] = -10.0  # dead
        params["opacities"][3:] = 4.0  # alive
        params["quats_l"][:] = torch.randn(N, 4, device=DEVICE)
        params["quats_r"][:] = torch.randn(N, 4, device=DEVICE)
        params["scales_4d_log"][:] = torch.randn(N, 4, device=DEVICE) - 2.0
        params["mu_t"][3:] = torch.tensor([0.1, 0.2, 0.3], device=DEVICE)
    parents = build_cov4d_rot(params["quats_l"][3:], params["quats_r"][3:], params["scales_4d_log"][3:]).detach()
    parent_mu_t = params["mu_t"][3:].detach().clone()

    strategy = MCMCSpatialStrategy(
        min_opacity=0.005, refine_start_iter=0, refine_stop_iter=10_000, refine_every=1, noise_lr=0.0
    )
    state = strategy.initialize_state()
    strategy.step_post_backward(params, optimizers, state, step=1, info=_make_info(), lr=0.0)

    assert len(params["means"]) == N
    after = build_cov4d_rot(params["quats_l"], params["quats_r"], params["scales_4d_log"]).detach()
    for i in range(N):
        j = int((parent_mu_t - params["mu_t"][i]).abs().argmin())
        assert torch.isclose(params["mu_t"][i], parent_mu_t[j]), "mu_t must be the parent's"
        assert torch.isclose(after[i, 3, 3], parents[j, 3, 3], rtol=1e-4), "temporal marginal changed"
        v_after, v_parent = after[i, :3, 3] / after[i, 3, 3], parents[j, :3, 3] / parents[j, 3, 3]
        assert torch.allclose(v_after, v_parent, rtol=1e-3, atol=1e-6), "velocity changed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute_relocation is CUDA-only")
def test_mcmc_temporal_partition_splits_time_and_stays_on_the_trajectory() -> None:
    """Partitioned copies narrow in time by their group size, sit on the parent's trajectory, and stay in the sequence."""
    torch.manual_seed(3)
    N = 8
    params, optimizers = _make_params(N)
    with torch.no_grad():
        params["opacities"][:6] = -10.0  # dead
        params["opacities"][6:] = 4.0  # two live parents
        params["quats_l"][:] = torch.randn(N, 4, device=DEVICE)
        params["quats_r"][:] = torch.randn(N, 4, device=DEVICE)
        params["scales_4d_log"][:] = torch.randn(N, 4, device=DEVICE) - 1.5
        params["mu_t"][6:] = torch.tensor([-0.45, 0.45], device=DEVICE)  # near the ends: offsets would leave
    cov_p = build_cov4d_rot(params["quats_l"][6:], params["quats_r"][6:], params["scales_4d_log"][6:]).detach()
    mean_p, mu_p = params["means"][6:].detach().clone(), params["mu_t"][6:].detach().clone()

    strategy = MCMCSpatialStrategy(
        min_opacity=0.005,
        refine_start_iter=0,
        refine_stop_iter=10,
        refine_every=1,
        noise_lr=0.0,
        cap_max=N,
        temporal_partition=True,
    )
    state = strategy.initialize_state()
    group_size = {}
    strategy.step_post_backward(params, optimizers, state, step=1, info=_make_info(), lr=0.0)

    assert params["mu_t"].abs().max() <= 0.5, "a piece left the normalized sequence"
    cov = build_cov4d_rot(params["quats_l"], params["quats_r"], params["scales_4d_log"]).detach()
    v_p = cov_p[:, :3, 3] / cov_p[:, 3, 3:]
    for i in range(N):
        # Each row descends from the parent whose velocity it carries.
        v_i = cov[i, :3, 3] / cov[i, 3, 3]
        j = int((v_p - v_i).norm(dim=-1).argmin())
        assert torch.allclose(v_i, v_p[j], rtol=1e-3, atol=1e-5), "velocity changed"
        group_size[j] = group_size.get(j, 0) + 1
        dt = params["mu_t"][i] - mu_p[j]
        assert torch.allclose(params["means"][i] - mean_p[j], v_p[j] * dt, rtol=1e-3, atol=1e-5), "off trajectory"
    for i in range(N):
        v_i = cov[i, :3, 3] / cov[i, 3, 3]
        j = int((v_p - v_i).norm(dim=-1).argmin())
        expected = cov_p[j, 3, 3] / group_size[j] ** 2
        assert torch.isclose(cov[i, 3, 3], expected, rtol=1e-3), "temporal extent is not parent / group size"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compute_relocation is CUDA-only")
def test_relocation_onto_one_parent_keeps_it_on_its_trajectory_and_clears_moved_rows() -> None:
    """Every draw hits the same parent: it takes one offset, not a mix of its duplicates', and the accumulator rows
    of the parent and of the relocated copies restart at zero instead of keeping the dead primitives' statistics."""
    torch.manual_seed(7)
    N = 8
    params, optimizers = _make_params(N)
    with torch.no_grad():
        params["opacities"][:7] = -10.0
        params["opacities"][7] = 4.0
        params["quats_l"][:] = torch.randn(N, 4, device=DEVICE)
        params["quats_r"][:] = torch.randn(N, 4, device=DEVICE)
        params["scales_4d_log"][:] = torch.randn(N, 4, device=DEVICE) - 1.5
    cov_p = build_cov4d_rot(params["quats_l"][7:], params["quats_r"][7:], params["scales_4d_log"][7:]).detach()[0]
    v_p = cov_p[:3, 3] / cov_p[3, 3]
    mean_p, mu_p = params["means"][7].detach().clone(), params["mu_t"][7].detach().clone()
    state = {"grad3d": torch.ones(N, device=DEVICE)}
    dead = torch.arange(N, device=DEVICE) < 7
    strategy = MCMCSpatialStrategy()
    strategy.initialize_state()

    relocate_spatial(params, optimizers, state, dead, strategy._binoms.to(DEVICE), temporal_partition=True)

    for i in range(N):
        dt = params["mu_t"][i] - mu_p
        assert torch.allclose(params["means"][i] - mean_p, v_p * dt, rtol=1e-3, atol=1e-5), f"row {i} off trajectory"
    assert (state["grad3d"] == 0).all(), "a moved row kept a dead primitive's accumulator"


def test_motion_score_skips_dead_rows_ranks_the_gradient_and_favours_moving_parents() -> None:
    """Motion weights are 0 on dead rows, follow the gradient's rank, and lean toward fast primitives."""
    torch.manual_seed(5)
    N = 400
    params, _ = _make_params(N, opacity_val=1.0)
    with torch.no_grad():
        params["opacities"][: N // 4] = -10.0  # dead
        params["quats_l"][:] = torch.randn(N, 4, device=DEVICE)
        params["quats_r"][:] = torch.randn(N, 4, device=DEVICE)
        params["scales_4d_log"][:] = torch.randn(N, 4, device=DEVICE) - 1.5
    # Heavy-tailed gradient: divided by its max it would be ~0 for almost every row.
    grad = torch.zeros(N, device=DEVICE)
    grad[N // 4 :] = torch.logspace(-6, 0, N - N // 4, device=DEVICE)[torch.randperm(N - N // 4, device=DEVICE)]

    speed = primitive_speed(params)
    ratios = {}
    for score in ParentScore:
        strategy = MCMCSpatialStrategy(min_opacity=0.005, parent_score=score)
        state = strategy.initialize_state()
        state["grad3d"], state["count3d"] = grad.clone(), torch.ones(N, device=DEVICE)
        weights = strategy.parent_weights(params, state)
        ratios[score] = ((weights * speed).sum() / weights.sum()).item()
    assert (weights[: N // 4] == 0).all(), "a dead row can be sampled as a parent"
    live = torch.arange(N // 4, N, device=DEVICE)
    order = grad[live].argsort()
    quarter = len(order) // 4
    top, bottom = weights[live][order[-quarter:]].mean(), weights[live][order[:quarter]].mean()
    assert top > 1.5 * bottom, "gradient term is flattened (normalized by its max rather than ranked?)"
    assert ratios[ParentScore.MOTION] > ratios[ParentScore.OPACITY]
