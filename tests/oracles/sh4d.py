"""PyTorch reference for the 4D SH colour the fused kernels evaluate.

Additive separable model ``color = SH(view_dir) + fourier(dir_t) @ sh_temporal + 0.5`` with
``2 · n_temporal_freqs`` temporal coefficients, not the product basis of the original 4DGS.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from gsplat import spherical_harmonics


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def temporal_cosine_basis(
    dir_t: float | Float[Tensor, ""] | Float[Tensor, "N"],
    n_freqs: int = 4,
) -> Float[Tensor, "T"] | Float[Tensor, "N T"]:
    """Interleaved ``[cos(kπ·dir_t), sin(kπ·dir_t)]`` for k = 1..n_freqs.

    Returns ``[2·n_freqs]`` for scalar ``dir_t`` and ``[N, 2·n_freqs]`` for ``[N]``.
    """
    if isinstance(dir_t, torch.Tensor):
        device = dir_t.device
        dir_t_f = dir_t.float()
    else:
        device = torch.device("cpu")
        dir_t_f = torch.tensor(dir_t, dtype=torch.float32)

    freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32, device=device)

    if dir_t_f.dim() == 0:
        # Scalar case → [n_freqs]
        angles = math.pi * freqs * dir_t_f  # [n_freqs]
        cos_vals = torch.cos(angles)  # [n_freqs]
        sin_vals = torch.sin(angles)  # [n_freqs]
        return torch.stack([cos_vals, sin_vals], dim=1).reshape(-1)  # [2*n_freqs]
    # Per-Gaussian case → [N, n_freqs]
    angles = math.pi * freqs[None, :] * dir_t_f[:, None]  # [N, n_freqs]
    cos_vals = torch.cos(angles)  # [N, n_freqs]
    sin_vals = torch.sin(angles)  # [N, n_freqs]
    return torch.stack([cos_vals, sin_vals], dim=-1).reshape(dir_t_f.shape[0], -1)  # [N, 2*n_freqs]


def eval_sh_4d(
    sh_spatial: Float[Tensor, "N K 3"],
    sh_temporal: Float[Tensor, "N T 3"],
    dirs: Float[Tensor, "N 3"],
    dir_t: Float[Tensor, "N"],
    sh_degree: int,
    n_temporal_freqs: int = 4,
) -> Float[Tensor, "N 3"]:
    """Per-Gaussian RGB ``[N, 3]``: spatial SH at ``dirs`` plus ``sh_temporal @ basis(dir_t)``."""
    # Spatial SH evaluation via gsplat (handles all SH degrees up to 3)
    dirs_norm = torch.nn.functional.normalize(dirs, dim=-1)  # [N, 3]
    spatial_colors = spherical_harmonics(sh_degree, dirs_norm, sh_spatial)  # [N, 3]

    # Per-Gaussian temporal basis [N, T_basis]; each row uses the relative time for that Gaussian
    temporal_feat = temporal_cosine_basis(dir_t, n_temporal_freqs)  # [N, T_basis]

    # Contract per-Gaussian basis with per-Gaussian coefficients [N, T_basis] x [N, T_basis, 3] → [N, 3]
    temporal_color = torch.einsum("nt,ntc->nc", temporal_feat, sh_temporal)  # [N, 3]

    # gsplat adds the +0.5 SH bias internally; this path bypasses it.
    return torch.clamp(spatial_colors + temporal_color + 0.5, min=0.0)  # [N, 3]


class Sh4dAppearanceModel:
    """4D SH appearance for a single camera per call."""

    def __init__(self, n_temporal_freqs: int = 4) -> None:
        """Initialize with the given number of temporal frequency components."""
        self.n_temporal_freqs = n_temporal_freqs

    def eval_colors(
        self,
        splats: nn.ParameterDict,
        means: Float[Tensor, "N 3"],
        viewmats: Float[Tensor, "C 4 4"],
        tau: float,
        sh_degree: int,
    ) -> Float[Tensor, "N 3"]:
        """Per-Gaussian RGB ``[N, 3]`` from ``viewmats[0]``; several cameras are refused."""
        if viewmats.shape[0] != 1:
            raise ValueError(f"eval_colors takes one camera per call, got {viewmats.shape[0]}; slice per camera.")
        viewmat = viewmats[0]  # [4, 4]
        R = viewmat[:3, :3]  # world-to-camera rotation
        t = viewmat[:3, 3]  # translation

        # Camera center in world coordinates: C = -R^T @ t
        camera_center = -(R.T @ t)  # [3]

        # Camera-to-Gaussian, as gsplat and published 4DGS; detached, so SH only optimizes coefficients.
        dirs = (means - camera_center.unsqueeze(0)).detach()  # [N, 3]

        sh_spatial = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # [N, K, 3]
        sh_temporal = splats["sh_temporal"]  # [N, T_basis, 3]

        dir_t = (tau - splats["mu_t"]).detach()  # [N]

        return eval_sh_4d(
            sh_spatial=sh_spatial,
            sh_temporal=sh_temporal,
            dirs=dirs,
            dir_t=dir_t,
            sh_degree=sh_degree,
            n_temporal_freqs=self.n_temporal_freqs,
        )
