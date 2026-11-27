"""``sigma_t_floor``: the coupled split floors only the temporal scale, and train resolves the sentinel."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from anycam4d.strategy._ops import split_full_4d
from anycam4d.trainer.optimizers.group import DenseAdamGroup


def _make_params(n: int, sigma_t: float) -> tuple[dict[str, nn.Parameter], DenseAdamGroup]:
    """FourDGaussians-schema params + optimizer group; every sigma_t initialised to ``sigma_t``."""
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    scales = torch.full((n, 4), -3.0)
    scales[:, 3] = torch.log(torch.tensor(sigma_t))
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3)),
        "quats_l": nn.Parameter(quats.clone()),
        "quats_r": nn.Parameter(quats.clone()),
        "scales_4d_log": nn.Parameter(scales),
        "opacities": nn.Parameter(torch.full((n,), 4.0)),
        "mu_t": nn.Parameter(torch.rand(n) - 0.5),
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    return params, DenseAdamGroup(optimizers)


def test_split_full_4d_floors_only_temporal() -> None:
    """The coupled split floors sigma_t (col 3) but lets the three spatial scales shrink freely."""
    torch.manual_seed(0)
    n = 6
    floor = 0.1
    # spatial scales start small so their /(0.8*N) shrink lands well below the temporal floor value
    params, opt = _make_params(n, sigma_t=0.12)
    params["scales_4d_log"].data[:, :3] = torch.log(torch.tensor(0.02))
    mask = torch.ones(n, dtype=torch.bool)
    split_full_4d(params, opt, {}, mask, N=2, sigma_t_floor=floor)

    sigma_t = torch.exp(params["scales_4d_log"][:, 3])
    assert (sigma_t >= floor - 1e-6).all(), "temporal scale breached the floor"
    spatial = torch.exp(params["scales_4d_log"][:, :3])
    assert (spatial < floor).all(), "spatial scales were floored — floor must touch column 3 only"


def test_floor_inert_when_zero() -> None:
    """sigma_t_floor=0.0 is a no-op: children equal the exact parent/(0.8*N) shrink."""
    torch.manual_seed(0)
    n = 5
    parent_sigma = 0.02  # well below any plausible floor
    params, opt = _make_params(n, sigma_t=parent_sigma)
    mask = torch.ones(n, dtype=torch.bool)
    split_full_4d(params, opt, {}, mask, N=2, sigma_t_floor=0.0)
    expected = parent_sigma / (0.8 * 2)
    sigma_t = torch.exp(params["scales_4d_log"][:, 3])
    assert torch.allclose(sigma_t, torch.full_like(sigma_t, expected), atol=1e-6)


def _exp_cfg(temporal_spread_init: float, sigma_t_floor: float, grow_scale_t: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(temporal_spread_init=temporal_spread_init),
        strategy=SimpleNamespace(
            sigma_t_floor=sigma_t_floor, grow_scale_t=grow_scale_t, min_mass=-1.0, min_opacity=0.005
        ),
    )


def test_train_resolves_sigma_t_floor() -> None:
    """The -1.0 sentinel resolves to 1/n_frames; init guard clamps a below-floor init up to the floor."""
    from scripts.train import _resolve_temporal_scales

    n_frames = 30
    # -1.0 auto-resolve; init already above the floor → left untouched
    cfg = _exp_cfg(temporal_spread_init=5.0 / n_frames, sigma_t_floor=-1.0)
    _resolve_temporal_scales(cfg, n_frames)
    assert cfg.strategy.sigma_t_floor == 1.0 / n_frames
    assert cfg.model.temporal_spread_init == 5.0 / n_frames  # 5/n > 1/n → not clamped

    # user forces init below the floor → guard clamps it up to the floor
    cfg2 = _exp_cfg(temporal_spread_init=0.5 / n_frames, sigma_t_floor=-1.0)
    _resolve_temporal_scales(cfg2, n_frames)
    assert cfg2.strategy.sigma_t_floor == 1.0 / n_frames
    assert cfg2.model.temporal_spread_init == 1.0 / n_frames  # clamped up
