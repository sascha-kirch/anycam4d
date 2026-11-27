"""Central/peripheral eval-region masks: a partition of camera rays by a reference pinhole's image.

The split is a property of the **ray**, not of any image grid: a ray is central when the reference
pinhole (the undistorted camera a model trained on) images it. In a distorted or fisheye render the
boundary curves, since the same ray set maps to a non-linear pixel grid.

Torch (not numpy/cv2): region masks are built from rendered-camera rays and consumed alongside
render/GT tensors at eval time, so they stay in the same tensor world as the rest of the eval
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


@dataclass(frozen=True)
class PinholeField:
    """The set of rays a pinhole camera images."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: float
    height: float


def central_mask_from_rays(rays: Float[Tensor, "H W 3"], field: PinholeField) -> Bool[Tensor, "H W"]:
    """Mark rays that land inside the reference pinhole's image; rays at or behind its plane do not.

    Args:
        rays: Camera-space ray directions of the render image (z forward).
        field: Reference pinhole, in the continuous pixel convention (``[0, width]``).
    """
    z = rays[..., 2]
    in_front = z > 0
    safe_z = torch.where(in_front, z, torch.ones_like(z))
    u = field.fx * rays[..., 0] / safe_z + field.cx
    v = field.fy * rays[..., 1] / safe_z + field.cy
    return in_front & (u >= 0) & (u <= field.width) & (v >= 0) & (v <= field.height)


def build_region_masks(
    rays: Float[Tensor, "H W 3"],
    field: PinholeField | None,
    valid: Bool[Tensor, "H W"] | None = None,
) -> dict[str, Bool[Tensor, "H W"]]:
    """Build the factored ``central`` / ``peripheral`` / ``valid`` masks for one camera.

    ``central``/``peripheral`` are static per camera; fold in any per-image semantic mask at eval time
    by ANDing it into both.

    Args:
        rays: Camera-space ray directions of the render image, ``[H, W, 3]``.
        field: Reference pinhole. ``None`` declares no split: both regions are empty, never a guess.
        valid: Optional in-FoV/observed mask ``[H, W]``; defaults to all-``True``.

    Returns:
        ``{"central": central & valid, "peripheral": ~central & valid, "valid": valid}``.
    """
    v = valid if valid is not None else torch.ones_like(rays[..., 0], dtype=torch.bool)
    if field is None:
        empty = torch.zeros_like(v)
        return {"central": empty, "peripheral": empty, "valid": v}
    central = central_mask_from_rays(rays, field)
    return {"central": central & v, "peripheral": (~central) & v, "valid": v}
