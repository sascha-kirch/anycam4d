"""Numpy rotation<->quaternion conversion for the prep pivot's ``c2w`` pose.

Stays inside ``prep/`` rather than joining ``anycam4d/image_ops``/``anycam4d/point_ops`` (it's
pose math, not an image or point-cloud op) or ``anycam4d/math/transforms.py`` (that module's
package ``__init__.py`` eagerly imports torch; this stays numpy-only so ``prep/`` never pulls
torch in transitively).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation


if TYPE_CHECKING:
    from jaxtyping import Float


def rotation_matrix_to_quat(r: Float[np.ndarray, "3 3"]) -> list[float]:
    """Convert a 3x3 rotation matrix to ``[qw, qx, qy, qz]`` (scipy returns ``[x,y,z,w]``)."""
    q = Rotation.from_matrix(r).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def quat_translation_to_c2w(rotation: list[float], translation: list[float]) -> Float[np.ndarray, "4 4"]:
    """Convert ``[qw,qx,qy,qz]`` + camera-origin translation to a 4x4 OpenCV c2w matrix.

    Numpy inverse of :func:`rotation_matrix_to_quat` plus the translation column. Kept
    numpy-only (unlike ``anycam4d.math.transforms.quat_translation_to_c2w``) so ``prep/``
    never pulls torch in transitively — see the module docstring.
    """
    qw, qx, qy, qz = rotation
    r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r
    c2w[:3, 3] = translation
    return c2w
