"""Conditioning: project 4D Gaussian parameters to 3D at a given timestamp."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.covariance import schur_condition
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.primitives.fourdgaussians.model import FourDGaussians


@dataclass(frozen=True)
class ConditionedGaussians:
    """3D Gaussians conditioned at one τ; every field is the conditional sibling of a marginal quantity.

    Invariant: ``opacity_cond = sigmoid(raw) · w_t``; never multiply by ``temporal_weights_raw`` again.
    ``temporal_weights_raw`` is for strategies only. ``quats``/``scales`` are ``None`` on the covariance path.
    """

    means_3d_cond: Float[Tensor, "N 3"]
    opacity_cond: Float[Tensor, "N"]
    temporal_weights_raw: Float[Tensor, "N"]  # for STRATEGY USE ONLY — never pass to renderer
    quats: Float[Tensor, "N 4"] | None = None
    scales: Float[Tensor, "N 3"] | None = None
    cov_3d_cond: Float[Tensor, "N 3 3"] | None = None
    colors: Float[Tensor, "N K 3"] | None = None  # SH coefficients [N, (sh_deg+1)^2, 3]


def temporal_marginal_std(cov_t_marg: Float[Tensor, "N"]) -> Float[Tensor, "N"]:
    """Guarded temporal width ``sqrt(cov_t_marg + TEMPORAL_VARIANCE_EPS)``, the width the gate divides by."""
    return torch.sqrt(cov_t_marg + TEMPORAL_VARIANCE_EPS)


def condition_at_t(
    model: FourDGaussians,
    tau: float | Float[Tensor, ""],
    *,
    prefilter_threshold: float = 0.0,
) -> ConditionedGaussians:
    """Condition 4D Gaussians at ``tau``, zeroing opacity where ``w_t < prefilter_threshold``."""
    return _condition_impl(model, tau, prefilter_threshold)


def _condition_impl(
    model: FourDGaussians,
    tau: float | Float[Tensor, ""],
    prefilter_threshold: float,
) -> ConditionedGaussians:
    """rot_4d covariance and Schur conditioning; sets ``cov_3d_cond`` (no quats or scales)."""
    splats = model.splats
    cov_eps = model.cov_eps

    cov_4d = build_cov4d_rot(splats["quats_l"], splats["quats_r"], splats["scales_4d_log"])
    mu_t = splats["mu_t"]

    mean_3d_corr, cov_3d_cond = schur_condition(cov_4d, tau, mu_t, eps=cov_eps)
    mean_3d_cond = splats["means"] + mean_3d_corr

    cov_t_marg = cov_4d[:, 3, 3]
    dt = tau - mu_t
    temporal_weights = torch.exp(-0.5 * dt * dt / (cov_t_marg + TEMPORAL_VARIANCE_EPS))

    if prefilter_threshold > 0.0:
        mask = temporal_weights >= prefilter_threshold
        temporal_weights_masked = temporal_weights * mask.float()
    else:
        temporal_weights_masked = temporal_weights

    opacity_cond = torch.sigmoid(splats["opacities"]) * temporal_weights_masked
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

    return ConditionedGaussians(
        means_3d_cond=mean_3d_cond,
        opacity_cond=opacity_cond,
        temporal_weights_raw=temporal_weights,
        cov_3d_cond=cov_3d_cond,
        quats=None,
        scales=None,
        colors=colors,
    )
