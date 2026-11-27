"""Regularization loss functions for 4D Gaussian Splatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anycam4d.math.covariance import temporal_marginal_from_rot4d
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor
    from torch import nn


def opacity_reg(
    params: dict[str, nn.Parameter],
    tau: float | Float[Tensor, ""],
    *,
    detach_gate: bool = False,
) -> Float[Tensor, ""]:
    """L1 penalty on the opacity composited at this timestamp: ``mean(sigmoid(alpha) * w_t(tau))``.

    Manufactures MCMC's relocation dead set. It prices presence at ``tau`` rather than existence
    (which makes widening a free discount) or footprint (which switches itself off as the width
    shrinks). ``w_t`` uses the marginal temporal variance, the same gate the renderer applies.

    With a live gate the gradient also reaches ``mu_t``, the temporal scale and the 4D rotation, and
    in expectation over ``tau`` the term is monotone in the width, so shrinking the width can
    extinguish it. ``detach_gate=True`` keeps the gate as a weight and pushes on opacity only.

    Warning:
        Nothing bounds the temporal width. Any tau-blind opacity penalty, or MCMC selection monotone
        in the width, re-creates the width runaway.

    Args:
        params: Reads ``opacities``, ``mu_t``, ``quats_l``, ``quats_r``, ``scales_4d_log``.
        tau: Normalized timestamp being rendered.
        detach_gate: Stop the gradient at ``w_t``.

    Returns:
        Scalar loss.
    """
    cov_t_marg = temporal_marginal_from_rot4d(params["quats_l"], params["quats_r"], params["scales_4d_log"])
    dt = tau - params["mu_t"]
    w_t = torch.exp(-0.5 * dt * dt / (cov_t_marg + TEMPORAL_VARIANCE_EPS))
    if detach_gate:
        w_t = w_t.detach()
    return (torch.sigmoid(params["opacities"]) * w_t).mean()


def scale_reg(scales: Float[Tensor, "N 3"]) -> Float[Tensor, ""]:
    """L1 penalty on the intrinsic spatial scales: ``mean(scale)``.

    Args:
        scales: Activated intrinsic spatial scales ``[N, 3]``.

    Returns:
        Scalar loss.
    """
    return scales.mean()
