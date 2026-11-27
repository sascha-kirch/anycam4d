"""Packed covariance factor L = [l00, l10, l20, l11, l21, l22] (Σ = L Lᵀ), the layout every kernel exchanges."""

from __future__ import annotations

import torch
from torch import Tensor

_ROWS = [0, 1, 2, 1, 2, 2]
_COLS = [0, 0, 0, 1, 1, 2]


def factor_of(cov: Tensor) -> Tensor:
    """[..., 3, 3] SPD → packed Cholesky factor [..., 6] (differentiable)."""
    L = torch.linalg.cholesky(cov)
    return L[..., _ROWS, _COLS]


def unpack(f: Tensor) -> Tensor:
    """Packed [..., 6] → lower-triangular [..., 3, 3]."""
    L = f.new_zeros(*f.shape[:-1], 3, 3)
    L[..., _ROWS, _COLS] = f
    return L


def cov_of(f: Tensor) -> Tensor:
    """Packed factor [..., 6] → Σ = L Lᵀ [..., 3, 3]."""
    L = unpack(f)
    return L @ L.transpose(-1, -2)


def upper_tri(cov: Tensor) -> Tensor:
    """[..., 3, 3] → gsplat upper-tri [σ00, σ01, σ02, σ11, σ12, σ22]."""
    return cov[..., [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]


def whitened_error(f: Tensor, f_ref: Tensor) -> Tensor:
    """‖L_ref⁻¹ L Lᵀ L_ref⁻ᵀ − I‖₂ per Gaussian: the relative error of every Mahalanobis distance."""
    L_ref = unpack(f_ref.double())
    X = torch.linalg.solve_triangular(L_ref, unpack(f.double()), upper=False)
    E = X @ X.transpose(-1, -2) - torch.eye(3, dtype=torch.float64, device=f.device)
    return torch.linalg.eigvalsh(E.cpu()).abs().amax(-1).to(f.device)


def from_upper_tri(u: Tensor) -> Tensor:
    """gsplat upper-tri [..., 6] → symmetric [..., 3, 3]."""
    row0 = torch.stack([u[..., 0], u[..., 1], u[..., 2]], -1)
    row1 = torch.stack([u[..., 1], u[..., 3], u[..., 4]], -1)
    row2 = torch.stack([u[..., 2], u[..., 4], u[..., 5]], -1)
    return torch.stack([row0, row1, row2], -2)
