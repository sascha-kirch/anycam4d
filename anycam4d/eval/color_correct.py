# Ported from gsplat examples/lib_bilagrid.py:
# Copyright 2024 Yuehao Wang (https://github.com/yuehaowang), from "Bilateral Guided Radiance Field Processing".
# Licensed under the Apache License, Version 2.0.

"""Per-channel affine color correction (ported from gsplat ``lib_bilagrid``).

Solves a per-channel quadratic-plus-affine least-squares mapping that warps a
rendered image's colors onto a reference (ground-truth) image. Used to report
color-corrected metrics that factor out global exposure/white-balance drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


def _is_unclipped(z: Float[Tensor, "..."], eps: float) -> Bool[Tensor, "..."]:
    """Boolean mask of values strictly inside ``[eps, 1 - eps]``."""
    return (z >= eps) & (z <= 1 - eps)


def color_correct(
    img: Float[Tensor, "... C"],
    ref: Float[Tensor, "... C"],
    num_iters: int = 5,
    eps: float = 0.5 / 255,
) -> Float[Tensor, "... C"]:
    """Warp ``img`` colors to match ``ref`` via iterative per-channel least squares.

    Fits a transformation (quadratic term + linear term + bias) per channel that maps
    the input colors onto the reference colors. The set of unsaturated pixels is
    re-estimated each iteration to reduce the impact of clipping. Both images must lie
    in ``[0, 1]``.

    Args:
        img: Image to be corrected, shape ``[..., C]`` (typically ``[H, W, 3]``).
        ref: Reference image to match, same trailing channel count as ``img``.
        num_iters: Number of refinement iterations.
        eps: Saturation guard band; pixels outside ``[eps, 1 - eps]`` are ignored.

    Returns:
        Color-corrected image with the same shape as ``img``.
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match")

    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    mask0 = _is_unclipped(img_mat, eps)

    for _ in range(num_iters):
        # Quadratic expansion of each pixel: per-channel products + linear + bias.
        a_terms = [img_mat[:, c : c + 1] * img_mat[:, c:] for c in range(num_channels)]
        a_terms.append(img_mat)
        a_terms.append(torch.ones_like(img_mat[:, :1]))
        a_mat = torch.cat(a_terms, dim=-1)

        warp = []
        for c in range(num_channels):
            b = ref_mat[:, c]
            # Ignore rows saturated in the input or in either current estimate.
            keep = mask0[:, c] & _is_unclipped(img_mat[:, c], eps) & _is_unclipped(b, eps)
            ma_mat = torch.where(keep[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(keep, b, torch.zeros_like(b))
            warp.append(torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0])
        warp = torch.stack(warp, dim=-1)
        img_mat = torch.clip(torch.matmul(a_mat, warp), 0, 1)

    return torch.reshape(img_mat, img.shape)
