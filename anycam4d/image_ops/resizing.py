"""Pure image/mask resize ops: no dataclass or camera-model knowledge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2


if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import UInt8


def resize_image(image: UInt8[np.ndarray, "H W 3"], new_w: int, new_h: int) -> UInt8[np.ndarray, "H W 3"]:
    """Resize an image: ``INTER_AREA`` when shrinking, ``INTER_LINEAR`` otherwise."""
    shrinking = new_w < image.shape[1] or new_h < image.shape[0]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def resize_mask(mask: UInt8[np.ndarray, "H W"], new_w: int, new_h: int) -> UInt8[np.ndarray, "H W"]:
    """Resize a mask with ``INTER_NEAREST`` — never introduces intermediate values."""
    return cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
