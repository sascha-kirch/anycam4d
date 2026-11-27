"""Numpy-only PLY point-cloud I/O: xyz + optional rgb, no normals (unused downstream)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from plyfile import PlyData
from plyfile import PlyElement


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float
    from jaxtyping import UInt8


def write_ply(
    path: Path,
    xyz: Float[np.ndarray, "N 3"],
    rgb: UInt8[np.ndarray, "N 3"] | None = None,
) -> None:
    """Write xyz + rgb (default gray 128 when uncolored) via plyfile."""
    xyz32 = np.asarray(xyz, dtype=np.float32)
    n = xyz32.shape[0]
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.zeros(n, dtype=dtype)
    elements["x"], elements["y"], elements["z"] = xyz32[:, 0], xyz32[:, 1], xyz32[:, 2]
    if rgb is not None:
        rgb8 = np.asarray(rgb, dtype=np.uint8)
        elements["red"], elements["green"], elements["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
    else:
        elements["red"] = elements["green"] = elements["blue"] = 128
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def read_ply(path: Path) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None]:
    """Read xyz (float32) + optional rgb (uint8), or ``(xyz, None)`` if no color fields."""
    data = PlyData.read(str(path))
    vertex = data["vertex"]

    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    xyz = np.stack([x, y, z], axis=-1)

    names = vertex.data.dtype.names
    if all(c in names for c in ("red", "green", "blue")):
        r = vertex["red"].astype(np.uint8)
        g = vertex["green"].astype(np.uint8)
        b = vertex["blue"].astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
    else:
        rgb = None

    return xyz, rgb
