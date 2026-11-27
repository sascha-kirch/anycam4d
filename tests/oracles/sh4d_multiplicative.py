"""Published 4DGS 4D SH, from ``utils/sh_utils.py:eval_shfs_4d`` of https://github.com/fudan-zvg/4d-gaussian-splatting
(MIT License, Copyright (c) 2024 Fudan Zhang Vision Group; full notice in ``tests/test_og4dgs_port_fidelity.py``).

Their constants and their basis expressions verbatim — those are what a transcription can get wrong,
and theirs are written in a different form from ours (``2zz-xx-yy`` where the kernel has
``3z²-1`` folded into two constants), so agreeing is also a check that both read a unit direction.

Two deliberate differences from the published function, both of which leave the value alone:
the 48 written-out product terms become a loop over the bases, and the per-order stride is taken
from the tensor instead of being hardcoded to 16. Published 4DGS only ever evaluates this at
``deg = 3`` when ``deg_t > 0``, because its curriculum saturates the spatial degree first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def sh_bases(deg: int, dirs: Float[Tensor, "... 3"]) -> list[Tensor | float]:
    """Their real-SH basis values ``Y_k(dirs)`` up to ``deg``, in their order, for a unit direction."""
    bases: list[Tensor | float] = [C0]
    if deg < 1:
        return bases

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    bases += [-1 * C1 * y, C1 * z, -1 * C1 * x]
    if deg < 2:
        return bases

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    bases += [
        C2[0] * xy,
        C2[1] * yz,
        C2[2] * (2.0 * zz - xx - yy),
        C2[3] * xz,
        C2[4] * (xx - yy),
    ]
    if deg < 3:
        return bases

    bases += [
        C3[0] * y * (3 * xx - yy),
        C3[1] * xy * z,
        C3[2] * y * (4 * zz - xx - yy),
        C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
        C3[4] * x * (4 * zz - xx - yy),
        C3[5] * z * (xx - yy),
        C3[6] * x * (xx - 3 * yy),
    ]
    return bases


def eval_shfs_4d(
    deg: int,
    deg_t: int,
    sh: Float[Tensor, "... KT 3"],
    dirs: Float[Tensor, "... 3"],
    dirs_t: Float[Tensor, "..."],
    k_max: int,
    l: float = 1.0,  # noqa: E741  (their parameter name: the sequence time span)
) -> Float[Tensor, "... 3"]:
    """``Σ_n cos(2π n dirs_t / l) · Σ_k Y_k(dirs) · sh[n·k_max + k]``, orders 0…deg_t.

    ``k_max`` is the per-order stride, 16 in the published code; blocks past ``deg_t`` are ignored,
    as they are there when the curriculum has not reached them.
    """
    bases = sh_bases(deg, dirs)

    result = torch.zeros_like(sh[..., 0, :])
    for n in range(deg_t + 1):
        weight = 1.0 if n == 0 else torch.cos(2 * torch.pi * n * dirs_t / l).unsqueeze(-1)
        block = torch.zeros_like(result)
        for k, basis in enumerate(bases):
            basis_t = basis if isinstance(basis, float) else basis.unsqueeze(-1)
            block = block + basis_t * sh[..., n * k_max + k, :]
        result = result + weight * block

    return result
