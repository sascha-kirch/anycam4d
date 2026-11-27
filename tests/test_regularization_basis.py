"""Regularizers ported from 3DGS MCMC, pinned by the 4D properties they are argued from.

``opacity_reg_detach_gate`` stops the gradient at ``w_t`` so only opacity is pushed.
"""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.trainer.trainer import raise_on_non_finite_loss
from anycam4d.losses.regularization import opacity_reg
from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.covariance import schur_condition


COV_EPS = 1e-6
GEOMETRY_KEYS = ("mu_t", "quats_l", "quats_r", "scales_4d_log")


def _params(n: int = 64, *, seed: int = 0) -> dict[str, torch.nn.Parameter]:
    """A random 4D population in float64, every parameter a leaf requiring grad."""
    g = torch.Generator().manual_seed(seed)
    raw = {
        "opacities": torch.randn(n, generator=g, dtype=torch.float64),
        "mu_t": torch.randn(n, generator=g, dtype=torch.float64) * 0.3,
        "quats_l": torch.randn(n, 4, generator=g, dtype=torch.float64),
        "quats_r": torch.randn(n, 4, generator=g, dtype=torch.float64),
        "scales_4d_log": torch.randn(n, 4, generator=g, dtype=torch.float64) * 0.8,
    }
    return {k: torch.nn.Parameter(v) for k, v in raw.items()}


def test_the_conditional_trace_does_not_depend_on_tau() -> None:
    """The conditional trace is independent of tau, so the term is deterministic per primitive."""
    p = _params(256)
    cov_4d = build_cov4d_rot(p["quats_l"], p["quats_r"], p["scales_4d_log"])
    _, early = schur_condition(cov_4d, -4.2, p["mu_t"], eps=COV_EPS)
    _, late = schur_condition(cov_4d, 9.1, p["mu_t"], eps=COV_EPS)
    assert torch.allclose(torch.diagonal(early, dim1=-2, dim2=-1), torch.diagonal(late, dim1=-2, dim2=-1))


def test_detach_gate_leaves_the_loss_value_untouched() -> None:
    """Detaching changes gradients only. A different value would confound the comparison round."""
    live = opacity_reg(_params(128), 0.25, detach_gate=False)
    detached = opacity_reg(_params(128), 0.25, detach_gate=True)
    assert live.item() == pytest.approx(detached.item(), rel=1e-12)


def test_detach_gate_removes_every_geometry_gradient() -> None:
    """With the gate detached, only ``opacities`` receives gradient; with it live, geometry does too."""
    live = _params(128)
    opacity_reg(live, 0.25, detach_gate=False).backward()
    assert live["opacities"].grad.abs().sum() > 0
    assert all(live[k].grad.abs().sum() > 0 for k in GEOMETRY_KEYS), "live gate should reach geometry"

    detached = _params(128)
    opacity_reg(detached, 0.25, detach_gate=True).backward()
    assert detached["opacities"].grad.abs().sum() > 0, "the term must still push opacity down"
    for key in GEOMETRY_KEYS:
        grad = detached[key].grad
        assert grad is None or grad.abs().sum() == 0.0, f"detached gate still put a gradient on {key}"


def test_detach_gate_keeps_the_gate_selective() -> None:
    """A detached gate still pushes a primitive far from tau less than one centred on it."""
    params = {
        "opacities": torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)),
        "mu_t": torch.nn.Parameter(torch.tensor([0.0, 5.0], dtype=torch.float64)),
        "quats_l": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)),
        "quats_r": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64)),
        "scales_4d_log": torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.float64)),
    }
    opacity_reg(params, 0.0, detach_gate=True).backward()
    on_time, far_away = params["opacities"].grad.abs()
    assert on_time > far_away, "the detached gate must still scope the push to what is being rendered"


def test_a_non_finite_loss_stops_the_run() -> None:
    """The guard names every non-finite term and passes a healthy step."""
    raise_on_non_finite_loss(7, 0.5, {"l1": 0.1, "scale_reg": 0.01})  # healthy: must not raise

    with pytest.raises(RuntimeError, match=r"loss is not finite \(scale_reg, opacity_reg\)"):
        raise_on_non_finite_loss(11_600, 0.5, {"scale_reg": math.nan, "opacity_reg": math.inf})

    with pytest.raises(RuntimeError, match=r"Step 3: loss is not finite \(total\)"):
        raise_on_non_finite_loss(3, math.nan, {"l1": 0.1})
