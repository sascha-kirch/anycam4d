"""Bilateral grid post-processor for per-camera appearance correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from fused_bilagrid import BilateralGrid
from fused_bilagrid import slice as bil_slice
from fused_bilagrid import total_variation_loss
from torch import nn


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class BilateralGridPostProcessor(nn.Module):
    """Per-camera appearance correction via fused bilateral grid.

    Implements the PostProcessor protocol. Applied after backend.render() produces
    colors. One bilateral grid per camera corrects per-camera exposure and tone
    variation in multi-camera setups.

    Args:
        n_cameras: Number of cameras in the dataset (one grid per camera).
        grid_shape: Bilateral grid dimensions as (grid_X, grid_Y, grid_W).
            Defaults to (8, 16, 8) — compact grid sufficient for smooth corrections.
    """

    def __init__(
        self,
        n_cameras: int,
        grid_shape: tuple[int, int, int] = (16, 16, 8),
    ) -> None:
        """Initialize bilateral grids for all cameras."""
        super().__init__()

        grid_X, grid_Y, grid_W = grid_shape
        self.grids = BilateralGrid(n_cameras, grid_X=grid_X, grid_Y=grid_Y, grid_W=grid_W)

    def forward(
        self,
        colors: Float[Tensor, "C H W D"],
        *,
        image_ids: list[int] | None = None,
        camera_ids: list[int] | None = None,
    ) -> Float[Tensor, "C H W D"]:
        """Apply per-camera bilateral grid correction to rendered colors.

        Uses camera_ids for grid indexing when available, falling back to image_ids.
        Only the first 3 channels (RGB) are corrected; remaining channels (e.g. depth)
        are passed through unchanged.

        Args:
            colors: Rendered color tensor [C, H, W, D].
            image_ids: Per-batch image indices (fallback when camera_ids is None).
            camera_ids: Per-batch camera indices (preferred for per-camera grids).

        Returns:
            Color-corrected tensor of the same shape as input.
        """
        ids = camera_ids if camera_ids is not None else image_ids
        if ids is None:
            return colors

        device = colors.device
        C, H, W = colors.shape[0], colors.shape[1], colors.shape[2]

        grid_y, grid_x = torch.meshgrid(
            (torch.arange(H, device=device) + 0.5) / H,
            (torch.arange(W, device=device) + 0.5) / W,
            indexing="ij",
        )
        # [1, H, W, 2] → [C, H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(C, -1, -1, -1)
        ids_tensor = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(-1)  # [C, 1]

        # Bilateral grid corrects RGB only; depth channel (if present) passes through.
        rgb = colors[..., :3]
        corrected_rgb = bil_slice(self.grids, grid_xy, rgb, ids_tensor)["rgb"]

        if colors.shape[-1] > 3:
            return torch.cat([corrected_rgb, colors[..., 3:]], dim=-1)

        return corrected_rgb

    def tv_loss(self) -> Float[Tensor, ""]:
        """Total variation regularization loss for spatial smoothness of grid parameters.

        Returns:
            Scalar TV loss over all camera grids.
        """
        return total_variation_loss(self.grids.grids)
