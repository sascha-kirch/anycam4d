"""Point cloud I/O: torch wrapper over the numpy ``point_ops.ply`` reader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from anycam4d.point_ops.ply import read_ply


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float
    from torch import Tensor


def load_ply(ply_path: Path) -> tuple[Float[Tensor, "N 3"], Float[Tensor, "N 3"] | None]:
    """Load XYZ positions and optional RGB colors from a PLY file.

    Args:
        ply_path: Path to the .ply file.

    Returns:
        (points, colors) where points is float32 [N, 3] and colors is
        float32 [N, 3] in [0, 1], or None if no color fields found.
    """
    xyz, rgb = read_ply(ply_path)
    points = torch.from_numpy(xyz)
    colors = torch.from_numpy(rgb.astype("float32") / 255.0) if rgb is not None else None
    return points, colors
