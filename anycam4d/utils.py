"""General utilities."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from jaxtyping import Float64
    from jaxtyping import Shaped
    from torch import Tensor


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch.

    Runs are still not bit-reproducible: the rasterizer backward accumulates with ``atomicAdd``.
    """
    random.seed(seed)
    # The global RNG, not a Generator: this pins third-party code that calls `np.random.*`.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_float64_numpy(value: Shaped[Tensor, "..."] | Shaped[np.ndarray, "..."]) -> Float64[np.ndarray, "..."]:
    """Coerce a tensor or array to a float64 numpy array (OpenCV's dtype)."""
    if torch.is_tensor(value):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)
