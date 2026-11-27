"""Mask-based loss functions for 4D Gaussian Splatting."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


def background_alpha_loss(
    alphas: Float[Tensor, "C H W 1"],
    fg_masks: Float[Tensor, "C H W 1"],
    validity: Bool[Tensor, "C H W 1"] | None = None,
) -> Float[Tensor, ""]:
    """Mean rendered alpha in observed background regions (should be zero).

    Converts the foreground mask to a background mask and computes the
    mean alpha over background pixels, normalised by the number of background
    pixels so that scenes with small backgrounds are not underweighted.

    A pixel outside the camera's observation footprint is not background — it is unknown. Without
    the validity gate every unobserved pixel counts as background and pushes the rendered alpha
    there to zero, erasing geometry that another camera may legitimately observe. Gating both the
    numerator and the normaliser by ``v`` restricts the term to background that was actually seen.

    Args:
        alphas: Rendered alpha channel [C, H, W, 1], values in [0, 1].
        fg_masks: Ground-truth foreground mask [C, H, W, 1], values in {0, 1}.
        validity: Observation mask [C, H, W, 1] of dtype bool, or None when every pixel was
            observed (the dense path, identical to no validity support).

    Returns:
        Scalar loss in [0, 1].
    """
    bg_mask = 1.0 - fg_masks.float()
    if validity is not None:
        bg_mask = bg_mask * validity.to(bg_mask.dtype)
    bg_denom = bg_mask.sum().clamp_min(1.0)
    return (alphas * bg_mask).sum() / bg_denom
