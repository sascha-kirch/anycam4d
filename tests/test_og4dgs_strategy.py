"""``OGFourDGSStrategy`` selects spatially but splits with the coupled full-4D kernel."""

from __future__ import annotations

import torch
from torch import nn

from anycam4d.enums import SpatialTrigger
from anycam4d.strategy.coupled import OGFourDGSStrategy
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup


def _make_params(n: int) -> tuple[dict[str, nn.Parameter], DenseAdamGroup]:
    """Minimal FourDGaussians-schema params + optimizer group (no rendering)."""
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3)),
        "quats_l": nn.Parameter(quats.clone()),
        "quats_r": nn.Parameter(quats.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -3.0)),
        "opacities": nn.Parameter(torch.full((n,), 4.0)),  # alive → never pruned
        "mu_t": nn.Parameter(torch.rand(n) - 0.5),
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    return params, DenseAdamGroup(optimizers)


def _split_all_strategy(cls: type[DefaultSpatialStrategy]) -> DefaultSpatialStrategy:
    """A strategy configured so every Gaussian is a *split* candidate (paramspace trigger)."""
    return cls(
        spatial_trigger=SpatialTrigger.PARAMSPACE,
        grow_grad3d=0.0,  # any gradient fires
        grow_scale3d=0.0,  # every Gaussian is "large" → split (not clone)
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100000,  # avoid opacity reset in the window
    )


def _drive_one_split(strategy: DefaultSpatialStrategy, params: dict, opt: DenseAdamGroup) -> dict:
    """Run one paramspace refine step with a strong world-space gradient on every Gaussian; return the state."""
    state = strategy.initialize_state(scene_scale=1.0)
    n = len(params["means"])
    meta: dict = {  # paramspace needs no means2d
        "radii": torch.ones(1, n, 2),
        "n_cameras": 1,
        "means_3d_cond": params["means"].detach() + torch.tensor([0.0, 0.0, 5.0]),
        "viewmats": torch.eye(4)[None],
    }
    strategy.step_pre_backward(params, opt, state, 1, meta)
    params["means"].grad = torch.ones(n, 3)
    meta["means_grad"] = params["means"].grad
    strategy.accumulate(params, state, 1, meta)
    strategy.step_post_backward(params, opt, state, 1, meta)
    return state


def test_og4dgs_split_is_full_4d() -> None:
    """A spatial-only split still moves every child's ``mu_t`` and ``sigma_t``."""
    torch.manual_seed(0)
    n = 6
    params, opt = _make_params(n)
    sigma_t_before = {round(v, 5) for v in params["scales_4d_log"][:, 3].tolist()}
    mu_t_before = {round(v, 5) for v in params["mu_t"].tolist()}

    state = _drive_one_split(_split_all_strategy(OGFourDGSStrategy), params, opt)
    assert state["refine_events"]["split_full_4d"] == n, "coupled splits must be counted as split_full_4d"
    assert state["refine_events"]["split_spatial"] == 0

    # Every output Gaussian is a coupled-split child: its temporal scale is shrunk off the parent set.
    for val in params["scales_4d_log"][:, 3].tolist():
        assert round(val, 5) not in sigma_t_before, "sigma_t was NOT mutated — coupling missing"
    # And its temporal mean was offset off the parent set.
    for val in params["mu_t"].tolist():
        assert round(val, 5) not in mu_t_before, "mu_t was NOT mutated — coupling missing"
    assert params["scales_4d_log"].isfinite().all()
    assert params["mu_t"].isfinite().all()


def test_default_spatial_split_unchanged() -> None:
    """Guard the default.py seam refactor: DefaultSpatial's split still leaves mu_t/sigma_t verbatim."""
    torch.manual_seed(0)
    n = 6
    params, opt = _make_params(n)
    sigma_t_before = {round(v, 5) for v in params["scales_4d_log"][:, 3].tolist()}
    mu_t_before = {round(v, 5) for v in params["mu_t"].tolist()}

    state = _drive_one_split(_split_all_strategy(DefaultSpatialStrategy), params, opt)
    assert state["refine_events"]["split_spatial"] == n
    assert state["refine_events"]["split_full_4d"] == 0

    for val in params["scales_4d_log"][:, 3].tolist():
        assert round(val, 5) in sigma_t_before, "DefaultSpatial mutated sigma_t — seam refactor broke it"
    for val in params["mu_t"].tolist():
        assert round(val, 5) in mu_t_before, "DefaultSpatial mutated mu_t — seam refactor broke it"


def test_og4dgs_selection_is_spatial_only() -> None:
    """With no spatial gradient, a temporal gradient produces no growth — OG has no temporal path."""
    n = 8
    params, opt = _make_params(n)
    strategy = OGFourDGSStrategy(
        spatial_trigger=SpatialTrigger.PARAMSPACE,
        grow_grad3d=1e-4,
        grow_scale3d=1e9,
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100000,
    )
    state = strategy.initialize_state(scene_scale=1.0)
    meta: dict = {"n_cameras": 1}

    n_before = len(params["means"])
    for step in range(1, 4):
        strategy.step_pre_backward(params, opt, state, step, meta)
        meta["radii"] = torch.ones(1, len(params["means"]), 2)
        meta["means_3d_cond"] = params["means"].detach()
        meta["viewmats"] = torch.eye(4)[None]
        # Temporal-only signal: mu_t carries a large gradient, means carries none.
        params["means"].grad = torch.zeros(len(params["means"]), 3)
        params["mu_t"].grad = torch.ones(len(params["mu_t"])) * 1e3
        meta["means_grad"] = params["means"].grad
        strategy.accumulate(params, state, step, meta)
        strategy.step_post_backward(params, opt, state, step, meta)

    assert len(params["means"]) == n_before, "temporal gradient must not trigger growth in OG"
