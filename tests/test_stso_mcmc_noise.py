"""MCMC noise and the sparse optimizer touch disjoint state."""

from __future__ import annotations

import copy

import torch

from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.strategy._ops import inject_noise_spatial
from anycam4d.trainer.optimizers.stso import SpatioTemporalSparseAdamReference


BETAS = (0.9, 0.999)
EPS = 1e-15
LR = 1e-2


def _means_optimizer(model: FourDGaussians) -> SpatioTemporalSparseAdamReference:
    """A sparse Adam over ``means`` only (the parameter MCMC noise displaces)."""
    p = model.splats["means"]
    return SpatioTemporalSparseAdamReference([{"params": p, "lr": LR}], lr=LR, betas=BETAS, eps=EPS)


def _fixed_grad(model: FourDGaussians, seed: int) -> None:
    """Populate a deterministic gradient on ``means``."""
    g = torch.Generator().manual_seed(seed)
    model.splats["means"].grad = torch.randn(model.splats["means"].shape, generator=g)


def test_inject_noise_leaves_moments_untouched() -> None:
    """``inject_noise`` is a pure parameter move: repeated calls leave the optimizer state untouched."""
    torch.manual_seed(0)
    model = FourDGaussians.from_random(n_points=64, sh_degree=1)
    n = len(model)
    opt = _means_optimizer(model)

    # One active step to give the primitives non-zero moments to freeze.
    _fixed_grad(model, seed=1)
    opt.step(torch.ones(n, dtype=torch.bool))

    state_before = copy.deepcopy(opt.state[model.splats["means"]])
    means_before = model.splats["means"].data.clone()

    # Repeated Langevin noise, no optimizer step.
    for _ in range(5):
        inject_noise_spatial(model.splats, state={}, scaler=LR)

    state_after = opt.state[model.splats["means"]]
    assert torch.equal(state_after["exp_avg"], state_before["exp_avg"])  # moments frozen
    assert torch.equal(state_after["exp_avg_sq"], state_before["exp_avg_sq"])
    assert not torch.equal(model.splats["means"].data, means_before)  # but the position did move


def test_reactivation_kick_is_a_bounded_adam_step() -> None:
    """A primitive displaced while inactive takes the same bounded Adam step on reactivation as an undisplaced twin."""
    torch.manual_seed(0)
    noised = FourDGaussians.from_random(n_points=64, sh_degree=1)
    control = copy.deepcopy(noised)
    n = len(noised)
    active = torch.ones(n, dtype=torch.bool)
    inactive = torch.zeros(n, dtype=torch.bool)

    opt_n, opt_c = _means_optimizer(noised), _means_optimizer(control)

    # Warmup: identical deterministic gradients mature both to the same moments.
    for s in range(20):
        _fixed_grad(noised, seed=s)
        _fixed_grad(control, seed=s)
        opt_n.step(active)
        opt_c.step(active)

    # Inactive phase: noise displaces `noised`; both optimizers are frozen (visibility all-False).
    # A large scaler emulates the real MCMC noise magnitude (lr * noise_lr) against the tiny
    # initial spatial covariance, so the displacement is unambiguous.
    for _ in range(30):
        inject_noise_spatial(noised.splats, state={}, scaler=200.0)
        opt_n.step(inactive)
        opt_c.step(inactive)

    # The freeze held: same moments on both despite the displacement.
    assert torch.allclose(
        opt_n.state[noised.splats["means"]]["exp_avg"], opt_c.state[control.splats["means"]]["exp_avg"]
    )
    # And the noised primitive really did move relative to the control.
    assert not torch.allclose(noised.splats["means"].data, control.splats["means"].data)

    # Reactivation: same fresh gradient into both; measure the step each takes.
    _fixed_grad(noised, seed=99)
    _fixed_grad(control, seed=99)
    pre_n = noised.splats["means"].data.clone()
    pre_c = control.splats["means"].data.clone()
    opt_n.step(active)
    opt_c.step(active)
    step_n = noised.splats["means"].data - pre_n
    step_c = control.splats["means"].data - pre_c

    # The displacement bypasses the optimizer, so the two reactivation steps are identical.
    assert torch.allclose(step_n, step_c, atol=1e-7)
    # And the kick is a bounded mature Adam step (per-element on the order of lr), not a divergent
    # jump — a stale-momentum divergence would be tied to the displacement and dwarf lr.
    assert step_n.abs().max().item() < 10.0 * LR
