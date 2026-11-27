"""Observation validity: which pixels of a camera were observed at all.

Static per camera and distinct from the per-frame semantic alpha. Unobserved pixels (outside an image
circle, past the field limit, outside a reprojection's source) must not be supervised.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.camera.registry import distortion_width
from anycam4d.data.schema import ReprojectSourceSchema
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.lens_domain import model_footprint
from anycam4d.utils import to_float64_numpy


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.schema import CameraSchema
    from anycam4d.enums import CameraModelType


def intrinsics_from_params(
    model: CameraModelType,
    params: list[float],
    scale_factor: float,
) -> tuple[Float[Tensor, "3 3"], Float[Tensor, "P"]]:
    """Split ``[fx, fy, cx, cy, dist...]`` into ``K`` and the canonical fixed-width distortion vector.

    The one place that knows the layout: the distortion prefix is copied into 12 (pinhole family) or
    4 (fisheye) zero-filled slots in OpenCV order. ``fx, fy, cx, cy`` are divided by ``scale_factor``.

    Returns:
        ``(K [3, 3], distortion [P])``.
    """
    K = torch.zeros(3, 3, dtype=torch.float32)
    K[0, 0] = params[0] / scale_factor
    K[1, 1] = params[1] / scale_factor
    K[0, 2] = params[2] / scale_factor
    K[1, 2] = params[3] / scale_factor
    K[2, 2] = 1.0

    dist = torch.zeros(distortion_width(model), dtype=torch.float32)
    coeffs = params[4:]
    dist[: len(coeffs)] = torch.tensor(coeffs, dtype=torch.float32)
    return K, dist


def _project_rays(
    dirs: Float[np.ndarray, "N 3"],
    K: Float[Tensor, "3 3"] | np.ndarray,
    distortion: Float[Tensor, "P"] | np.ndarray,
    model: CameraModelType,
) -> Float[np.ndarray, "N 2"]:
    """Project camera-space rays to pixels through a lens (canonical distortion); inverse of ``camera_rays``."""
    obj = np.ascontiguousarray(dirs, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    K_np = to_float64_numpy(K)
    dist_np = to_float64_numpy(distortion)

    if CAMERA_REGISTRY[model].family is ProjectionFamily.FISHEYE_FAMILY:
        pix, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, K_np, dist_np)
    else:
        pix, _ = cv2.projectPoints(obj, rvec, tvec, K_np, dist_np)
    return pix.reshape(-1, 2)


def _radians(degrees: float | None) -> float | None:
    """A recorded field limit in degrees as radians, passing ``None`` (nothing recorded) through."""
    return None if degrees is None else math.radians(degrees)


def _in_bounds(pixels: Float[np.ndarray, "N 2"], width: int, height: int) -> Bool[np.ndarray, "N"]:
    """Mark pixel coordinates that are finite and inside a ``width x height`` image."""
    u, v = pixels[:, 0], pixels[:, 1]
    finite = np.isfinite(u) & np.isfinite(v)
    return finite & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def _validity_reprojected(
    K: Float[Tensor, "3 3"],
    distortion: Float[Tensor, "P"],
    model: CameraModelType,
    width: int,
    height: int,
    source: ReprojectSourceSchema,
) -> Bool[Tensor, "H W"]:
    """Footprint of a resampled camera: replay the warp and keep target pixels whose ray lands in the source image."""
    rays, ray_valid = camera_rays(K, distortion, model, width, height)
    # Placeholder for NaN rays so projection stays finite; they are dropped by ray_valid.
    forward = torch.tensor([0.0, 0.0, 1.0])
    rays = torch.where(ray_valid[..., None], rays, forward)

    K_src, dist_src = intrinsics_from_params(source.model, source.params, 1.0)
    pixels = _project_rays(rays.reshape(-1, 3).numpy(), K_src, dist_src, source.model)
    in_bounds = torch.from_numpy(_in_bounds(pixels, source.width, source.height).reshape(height, width))
    return ray_valid & in_bounds


def _validity_monotone_domain(
    K: Float[Tensor, "3 3"],
    distortion: Float[Tensor, "P"],
    model: CameraModelType,
    width: int,
    height: int,
    max_incidence: float | None = None,
) -> Bool[Tensor, "H W"]:
    """Torch adapter over :func:`~anycam4d.math.lens_domain.model_footprint`."""
    family = CAMERA_REGISTRY[model].family
    footprint = model_footprint(to_float64_numpy(K), to_float64_numpy(distortion), family, width, height, max_incidence)
    return torch.from_numpy(footprint)


def compute_validity(
    model: CameraModelType,
    K: Float[Tensor, "3 3"],
    distortion: Float[Tensor, "P"],
    width: int,
    height: int,
    source: ReprojectSourceSchema | None = None,
    max_incidence_deg: float | None = None,
) -> Bool[Tensor, "H W"]:
    """Procedural validity mask for one camera, in its own image space.

    - ``source`` given: replay the reprojection.
    - Distortion-free pinhole family with no field limit: all observed (ambiguous with a reprojected
      pinhole that lost its provenance, which is why prep records ``reproject_source``).
    - Otherwise: the lens footprint, bounded by ``max_incidence_deg``.

    Args:
        model: Camera model.
        K: Intrinsics ``[3, 3]`` at ``(width, height)``.
        distortion: Canonical distortion vector.
        width: Image width.
        height: Image height.
        source: ``reproject_source`` provenance, or ``None``.
        max_incidence_deg: Recorded field limit in degrees, or ``None``.

    Returns:
        ``[H, W]`` bool CPU tensor.
    """
    if isinstance(source, ReprojectSourceSchema):
        valid = _validity_reprojected(K, distortion, model, width, height, source)
    elif (
        CAMERA_REGISTRY[model].family is ProjectionFamily.PINHOLE_FAMILY
        and not bool(distortion.any())
        and max_incidence_deg is None
    ):
        valid = torch.ones((height, width), dtype=torch.bool)
    else:
        valid = _validity_monotone_domain(K, distortion, model, width, height, _radians(max_incidence_deg))
    return valid


def resolve_validity(
    camera: CameraSchema,
    on_disk_mask: Bool[Tensor, "H W"] | None,
    width: int,
    height: int,
) -> Bool[Tensor, "H W"]:
    """Validity by precedence ``on-disk mask > provenance > model``, at the requested resolution.

    Intrinsics are rescaled rather than a mask resampled. No file I/O.

    Args:
        camera: Manifest entry.
        on_disk_mask: Loaded ``valid/<camera_id>`` mask, or ``None``.
        width: Target width.
        height: Target height.

    Returns:
        ``[H, W]`` bool mask.
    """
    if on_disk_mask is not None:
        return on_disk_mask
    K, distortion = intrinsics_from_params(camera.model, camera.params, camera.width / width)
    return compute_validity(
        camera.model, K, distortion, width, height, camera.reproject_source, camera.max_incidence_deg
    )
