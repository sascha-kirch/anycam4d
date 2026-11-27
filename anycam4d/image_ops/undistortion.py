"""Pure camera-model undistort ops: Brown-Conrady (pinhole-family) + Kannala-Brandt (fisheye-family).

No dataclass or ``Stage`` knowledge — these take raw ``K``/``params``/dims and return the maps a
caller remaps with. Model/family dispatch stays with the caller (``ToPinhole``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import UInt8


_FISHEYE_N_COEFFS = 4


def undistort_pinhole_family(
    K: Float[np.ndarray, "3 3"],
    params: list[float],
    w: int,
    h: int,
) -> tuple[Float[np.ndarray, "3 3"], np.ndarray, np.ndarray]:
    """Brown-Conrady undistort: optimal new K + rectify maps for pinhole-family models.

    Maps are emitted as fixed-point ``CV_16SC2`` (an integer coord map + an interpolation-weight
    table), which ``cv2.remap`` consumes markedly faster than two float ``CV_32FC1`` maps and at
    roughly half the memory — worth it since a cached map is remapped across every frame.
    """
    k64 = K.astype(np.float64)
    dist_coeffs = np.asarray(params[4:], dtype=np.float64)
    new_k, _ = cv2.getOptimalNewCameraMatrix(k64, dist_coeffs, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(k64, dist_coeffs, None, new_k, (w, h), cv2.CV_16SC2)
    return new_k, map1, map2


def undistort_fisheye_family(
    K: Float[np.ndarray, "3 3"],
    params: list[float],
    w: int,
    h: int,
) -> tuple[Float[np.ndarray, "3 3"], np.ndarray, np.ndarray]:
    """Kannala-Brandt undistort: optimal new K + rectify maps for fisheye-family models.

    Maps are emitted as fixed-point ``CV_16SC2`` (see :func:`undistort_pinhole_family`) for a
    faster, lighter ``cv2.remap`` across frames.
    """
    k64 = K.astype(np.float64)
    coeffs = list(params[4 : 4 + _FISHEYE_N_COEFFS])
    coeffs += [0.0] * (_FISHEYE_N_COEFFS - len(coeffs))
    d = np.asarray(coeffs, dtype=np.float64).reshape(_FISHEYE_N_COEFFS, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k64, d, (w, h), np.eye(3))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k64, d, np.eye(3), new_k, (w, h), cv2.CV_16SC2)
    return new_k, map1, map2


def remap_image(image: UInt8[np.ndarray, "H W 3"], map1: np.ndarray, map2: np.ndarray) -> UInt8[np.ndarray, "H W 3"]:
    """Remap an image with ``INTER_LINEAR``."""
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


def remap_mask(mask: UInt8[np.ndarray, "H W"], map1: np.ndarray, map2: np.ndarray) -> UInt8[np.ndarray, "H W"]:
    """Remap a mask with ``INTER_NEAREST`` — never introduces intermediate values."""
    return cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
