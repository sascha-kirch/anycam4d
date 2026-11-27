"""Shared helpers used by multiple primitive types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scipy.spatial import KDTree


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def _knn_distances(points: Float[Tensor, "N 3"], k: int = 4) -> Float[Tensor, "N k"]:
    """Return Euclidean distances to the k nearest neighbors (including self at index 0)."""
    x_np = points.detach().cpu().numpy()
    tree = KDTree(x_np)
    distances, _ = tree.query(x_np, k=k)
    return torch.from_numpy(distances).to(points)


def RGB2SH(rgb: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3"]:
    """Convert RGB values in [0, 1] to the SH DC coefficient that reproduces them."""
    _C0 = 0.28209479177387814
    return (rgb.float() - 0.5) / _C0
