"""ToPinhole stage: undistort any pinhole/fisheye-family view to a clean pinhole model."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.data.prep.stream import StageProto
from anycam4d.enums import CameraModelType
from anycam4d.image_ops.undistortion import remap_image
from anycam4d.image_ops.undistortion import remap_mask
from anycam4d.image_ops.undistortion import undistort_fisheye_family
from anycam4d.image_ops.undistortion import undistort_pinhole_family


if TYPE_CHECKING:
    from anycam4d.data.prep.sample import CalibratedView


class ToPinhole(StageProto):
    """Undistort a view into a distortion-free pinhole.

    Brown-Conrady models go through ``cv2.initUndistortRectifyMap``, the fisheye family through ``cv2.fisheye``. A
    ``pinhole`` view passes through unchanged and gains no ``reproject_source``; every other view records its native
    lens there, so the pixels the remap pulled from outside the source stay identifiable. Rectify maps depend only on
    intrinsics and resolution and are cached per ``(cam_id, width, height)``.
    """

    def __init__(self) -> None:
        """Start with an empty per-camera rectify-map cache."""
        self._maps: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __call__(self, view: CalibratedView) -> CalibratedView:
        """Return an undistorted, ``pinhole``-model copy of ``view``."""
        if view.model is CameraModelType.PINHOLE:
            return view

        h, w = view.image.shape[0], view.image.shape[1]
        new_k, map1, map2 = self._rectify_maps(view, w, h)

        image = remap_image(view.image, map1, map2)
        mask = remap_mask(view.mask, map1, map2) if view.mask is not None else None

        # Record the native lens before it is overwritten. The undistorted pinhole says "zero
        # distortion", under which every pixel looks observed — but the remap samples some target
        # pixels from outside the source image, and those were never observed. The native model,
        # its intrinsics, and its *pre-remap* dimensions make that footprint replayable at load.
        reproject_source = {
            "model": view.model.value,
            "width": int(w),
            "height": int(h),
            "params": [float(p) for p in view.params],
        }

        fx, fy, cx, cy = new_k[0, 0], new_k[1, 1], new_k[0, 2], new_k[1, 2]
        return dataclasses.replace(
            view,
            model=CameraModelType.PINHOLE,
            K=new_k.astype(np.float32),
            params=[float(fx), float(fy), float(cx), float(cy)],
            image=image,
            mask=mask,
            reproject_source=reproject_source,
            # The rectified frame is a different lens seeing a different set of directions, so the
            # source's cone does not describe it. Its footprint is replayed through
            # ``reproject_source`` instead, which bounds it by the source image rather than by an
            # angle.
            max_incidence_deg=None,
        )

    def _rectify_maps(self, view: CalibratedView, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(new_K, map1, map2)`` for this camera, building + caching them on first use."""
        key = (view.cam_id, w, h)
        cached = self._maps.get(key)
        if cached is not None:
            return cached

        family = CAMERA_REGISTRY[view.model].family
        if family is ProjectionFamily.PINHOLE_FAMILY:
            new_k, map1, map2 = undistort_pinhole_family(view.K, view.params, w, h)
        else:
            new_k, map1, map2 = undistort_fisheye_family(view.K, view.params, w, h)

        self._maps[key] = (new_k, map1, map2)
        return new_k, map1, map2
