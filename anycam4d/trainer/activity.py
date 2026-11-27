"""Spatio-temporal activity mask for STSO: ``a_i(τ) = visible_i(τ) ∧ [w_t,i(τ) > ε_t]``.

The prefilter zeroes the opacity of temporally gated primitives, so they are culled like invisible ones;
the active set is therefore ``radii > 0``, exactly the primitives that receive gradient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Bool
    from torch import Tensor


def build_activity_mask(meta: dict | None, n_gaussians: int, device: torch.device) -> Bool[Tensor, "N"]:
    """Active set ``[N]`` read from render meta, without host sync.

    Sources, in order: packed ``gaussian_ids`` scattered over cameras; dense ``radii [C, N, 2]`` reduced
    over cameras and axes; all-true when meta carries neither.
    """
    meta = meta or {}
    if meta.get("packed") and meta.get("gaussian_ids") is not None:
        mask = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
        mask.scatter_(0, meta["gaussian_ids"], value=True)  # union over packed rows (all cameras)
        return mask

    radii = meta.get("radii")
    if radii is None:
        return torch.ones(n_gaussians, dtype=torch.bool, device=device)  # eval / NullStrategy / missing
    # radii [C, N, 2]: reduce cameras and the width/height axis, keeping the Gaussian axis.
    alive = radii > 0
    reduce_dims = tuple(d for d in range(alive.ndim) if d != 1)
    return alive.any(dim=reduce_dims) if reduce_dims else alive
