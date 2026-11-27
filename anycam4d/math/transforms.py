"""Rigid-transform math: quaternion + translation → 4x4 matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def quat_translation_to_c2w(
    rotation: list[float],
    translation: list[float],
) -> Float[Tensor, "4 4"]:
    """Build a 4x4 camera-to-world matrix from a unit quaternion and translation.

    Args:
        rotation: [qw, qx, qy, qz] unit quaternion.
        translation: [tx, ty, tz] camera origin in world coordinates.

    Returns:
        4x4 float32 camera-to-world transform.
    """
    qw, qx, qy, qz = rotation
    c2w = torch.zeros(4, 4, dtype=torch.float32)
    c2w[3, 3] = 1.0
    c2w[0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    c2w[0, 1] = 2.0 * (qx * qy - qz * qw)
    c2w[0, 2] = 2.0 * (qx * qz + qy * qw)
    c2w[1, 0] = 2.0 * (qx * qy + qz * qw)
    c2w[1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    c2w[1, 2] = 2.0 * (qy * qz - qx * qw)
    c2w[2, 0] = 2.0 * (qx * qz - qy * qw)
    c2w[2, 1] = 2.0 * (qy * qz + qx * qw)
    c2w[2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    c2w[0, 3] = translation[0]
    c2w[1, 3] = translation[1]
    c2w[2, 3] = translation[2]
    return c2w
