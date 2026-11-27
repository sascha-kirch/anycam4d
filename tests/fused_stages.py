"""The conditioning and UT-projection stages of the production fused kernel, called one at a time.

The fused UT kernel writes the conditioned primitive for every Gaussian regardless of culling, and its 2D outputs
are the tile bound the renderer uses, so tests read both stages from it rather than from separate kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor

from anycam4d.cuda.autograd import UTFusedFunction
from anycam4d.enums import AppearanceType
from anycam4d.primitives.appearance import APPEARANCE_CUDA_ID

_SH_STRIDE = 15  # shN rows at the maximum spatial degree; degree 0 reads none of them


def _fused(
    gaussians: tuple[Tensor, ...],
    viewmats: Tensor,
    Ks: Tensor,
    radial_coeffs: Tensor,
    tau: float,
    cov_eps: float,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    prefilter_threshold: float,
    width: int,
    height: int,
    family: int,
) -> tuple[Tensor, ...]:
    n, device = gaussians[0].shape[0], gaussians[0].device
    sh0 = torch.zeros(n, 1, 3, device=device)
    shN = torch.zeros(n, _SH_STRIDE, 3, device=device)
    no_temporal = torch.empty(0, device=device)
    kind = APPEARANCE_CUDA_ID[AppearanceType.SH4D_ADDITIVE]
    return UTFusedFunction.apply(
        *gaussians,
        sh0,
        shN,
        no_temporal,
        0,
        kind,
        0,
        0,
        viewmats,
        Ks,
        radial_coeffs,
        float(tau),
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter_threshold,
        width,
        height,
        family,
    )


def condition(
    quats_l: Tensor,
    quats_r: Tensor,
    scales_4d_log: Tensor,
    means: Tensor,
    mu_t: Tensor,
    opacities: Tensor,
    tau: float,
    cov_eps: float,
    prefilter_threshold: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(means_3d_cond, cov_factors, opacities, wt)`` at ``tau``; differentiable except ``wt``."""
    device = means.device
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.eye(3, device=device)[None]
    radial = torch.zeros(1, 12, device=device)
    out = _fused(
        (quats_l, quats_r, scales_4d_log, means, mu_t, opacities),
        viewmats,
        Ks,
        radial,
        tau,
        cov_eps,
        0.3,
        0.01,
        1e10,
        prefilter_threshold,
        1,
        1,
        0,
    )
    return out[5], out[6], out[7], out[8]


def fused_ut_project(
    quats_l: Tensor,
    quats_r: Tensor,
    scales_4d_log: Tensor,
    means: Tensor,
    mu_t: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    radial_coeffs: Tensor,
    tau: float,
    cov_eps: float,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    prefilter_threshold: float,
    image_width: int,
    image_height: int,
    family: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(radii [C, N, 2], means2d [C, N, 2], depths [C, N], conics [C, N, 3])``, the renderer's tile bound."""
    with torch.no_grad():
        means2d, conics, depths, radii, *_ = _fused(
            (quats_l, quats_r, scales_4d_log, means, mu_t, opacities),
            viewmats,
            Ks,
            radial_coeffs,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter_threshold,
            image_width,
            image_height,
            family,
        )
    return radii, means2d, depths, conics
