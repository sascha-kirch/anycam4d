"""Per-pixel camera rays for any registered camera model.

A torch adapter over :func:`anycam4d.math.lens_domain.unproject_pixels`, which owns the inverse of
both projection families. Rays are returned on the **unit sphere with a signed ``z``**, together
with an explicit per-pixel validity flag. Directions are ``NaN`` where invalid; ``valid`` is the
only thing a caller may branch on.

Used for region-split evaluation in a camera's own native space (`anycam4d.eval.native_evaluator`)
and by the validity chain when replaying a reprojection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.math.lens_domain import PIXEL_CENTRE
from anycam4d.math.lens_domain import unproject_pixels
from anycam4d.utils import to_float64_numpy


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.enums import CameraModelType


def camera_rays(
    K: Float[Tensor, "3 3"] | np.ndarray,
    distortion: Float[Tensor, "P"] | np.ndarray,
    model: CameraModelType,
    width: int,
    height: int,
) -> tuple[Float[Tensor, "H W 3"], Bool[Tensor, "H W"]]:
    """Per-pixel camera-space ray directions on the unit sphere, plus a per-pixel validity flag.

    Args:
        K: Intrinsic matrix ``[3, 3]``. Skew is not modelled, matching the renderer's camera ops,
            which read only ``fx, fy, cx, cy``.
        distortion: Canonical fixed-width distortion vector (``distortion_width(model)`` long) —
            the same tensor the renderer consumes.
        model: Camera model.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(rays [H, W, 3], valid [H, W])``. ``rays`` are unit vectors with a **signed** ``z``, one
        per pixel, sampled at ``index + 0.5`` — the point the world rasterizer draws that pixel at.
        ``valid`` is ``False`` where the lens model has no determinable ray for that pixel; ``rays``
        is ``NaN`` there, so a caller that ignores ``valid`` fails loudly.
    """
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    pixels = np.stack([uu.ravel(), vv.ravel()], axis=-1) + PIXEL_CENTRE
    family = CAMERA_REGISTRY[model].family
    dirs, valid = unproject_pixels(pixels, to_float64_numpy(K), to_float64_numpy(distortion), family)

    rays = torch.from_numpy(dirs.reshape(height, width, 3)).float()
    return rays, torch.from_numpy(valid.reshape(height, width))
