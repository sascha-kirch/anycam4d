"""Resize stage: rescale image/mask/intrinsics; distortion coefficients unchanged."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from anycam4d.data.prep.stream import StageProto
from anycam4d.image_ops.resizing import resize_image
from anycam4d.image_ops.resizing import resize_mask


if TYPE_CHECKING:
    from anycam4d.data.prep.sample import CalibratedView


class Resize(StageProto):
    """Rescale a view's image, mask, and intrinsics; distortion coefficients pass through.

    Exactly one of ``factor`` or ``target_wh`` must be given.

    - ``factor``: matches the training loader's own downscale convention — ``fx, fy, cx, cy``
      are divided by ``factor``; output dims are ``floor(H/factor), floor(W/factor)``.
    - ``target_wh``: scales intrinsics by the exact per-axis ratios ``sx = new_w/W``,
      ``sy = new_h/H``.

    Radial/tangential distortion coefficients (``params[4:]``) are defined in normalized
    image coordinates and are scale-invariant, so they are never touched here — only
    ``params[:4]`` and ``K`` scale.
    """

    def __init__(self, factor: float | None = None, target_wh: tuple[int, int] | None = None) -> None:
        """Configure the resize; exactly one of ``factor``/``target_wh`` must be given."""
        if (factor is None) == (target_wh is None):
            raise ValueError("Resize needs exactly one of `factor` or `target_wh`")
        self._factor = factor
        self._target_wh = target_wh

    def __call__(self, view: CalibratedView) -> CalibratedView:
        """Return a resized copy of ``view`` with intrinsics scaled to match."""
        h, w = view.image.shape[0], view.image.shape[1]
        new_w, new_h, sx, sy = self._target_dims(w, h)

        image = resize_image(view.image, new_w, new_h)
        mask = resize_mask(view.mask, new_w, new_h) if view.mask is not None else None

        K = view.K.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy

        params = list(view.params)
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy

        return dataclasses.replace(view, image=image, mask=mask, K=K, params=params)

    def _target_dims(self, w: int, h: int) -> tuple[int, int, float, float]:
        """Return ``(new_w, new_h, sx, sy)`` for the configured resize mode."""
        if self._factor is not None:
            new_w, new_h = math.floor(w / self._factor), math.floor(h / self._factor)
            sx = sy = 1.0 / self._factor
        else:
            new_w, new_h = self._target_wh
            sx, sy = new_w / w, new_h / h
        return new_w, new_h, sx, sy
