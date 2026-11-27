"""Scene normalization utilities for camera-based coordinate normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset


def compute_scene_norm(cam_centers: Float[Tensor, "N 3"]) -> tuple[float, Float[Tensor, "3"]]:
    """Compute normalization scale and center from camera positions.

    Matches gsplat's camera-based approach: center at camera centroid,
    scale = max camera distance from centroid.

    Args:
        cam_centers: Camera positions in world space [N, 3].

    Returns:
        (scale, center) — apply as ``(points - center) / scale``.
    """
    pts64 = cam_centers.double()
    center = pts64.mean(dim=0)
    dists = (pts64 - center).norm(dim=-1)
    scale = float(dists.max().item())
    if scale == 0.0:
        scale = 1.0
    return scale, center.float()


@dataclass
class SceneNormalization:
    """Scene normalization parameters computed from camera positions.

    Attributes:
        norm_scale: Scale factor used to normalize point cloud to unit sphere.
        norm_center: World-space centroid of camera positions [3].
        scene_scale: Camera extent in normalized space (for optimizer LR scaling).
        T_denorm: 4x4 denormalization matrix (maps normalized → world coords).
    """

    norm_scale: float
    norm_center: Float[Tensor, "3"]
    scene_scale: float
    T_denorm: Float[Tensor, "4 4"]


def normalize_viewmat(viewmat: Float[Tensor, "4 4"], T_denorm: Float[Tensor, "4 4"]) -> Float[Tensor, "4 4"]:
    """World-to-camera ``viewmat`` into normalized scene space, kept rigid as gsplat's ``transform_cameras`` does.

    ``viewmat @ T_denorm`` has an ``s·R`` block; the kernels take the camera centre as ``-Rᵀt``, which holds only
    for a rotation, so the top rows are divided by ``s``. Camera space is then in normalized units.
    """
    out = viewmat @ T_denorm
    scale = torch.linalg.norm(T_denorm[:3, 0])
    return torch.cat([out[:3] / scale, out[3:]])


def compute_normalization(dataset: MultiViewSequenceDataset) -> SceneNormalization:
    """Compute scene normalization from camera positions in a dataset.

    Args:
        dataset: Training dataset with camera viewmats.

    Returns:
        SceneNormalization with scale, center, and denormalization transform.
    """
    cam_centers = torch.stack([torch.linalg.inv(v)[:3, 3] for v in dataset.viewmats])
    norm_scale, norm_center = compute_scene_norm(cam_centers)

    T_denorm = torch.eye(4, dtype=torch.float32)
    T_denorm[:3, :3] *= norm_scale
    T_denorm[:3, 3] = norm_center

    # scene_scale in normalized space (after applying T_denorm to viewmats)
    # We need to temporarily apply the transform to compute normalized camera distances
    viewmats_norm = torch.stack([normalize_viewmat(v, T_denorm) for v in dataset.viewmats])
    cam_centers_norm = torch.stack([torch.linalg.inv(v)[:3, 3] for v in viewmats_norm])
    cam_dists_norm = (cam_centers_norm - cam_centers_norm.mean(dim=0)).norm(dim=-1)
    scene_scale = float(cam_dists_norm.max().item()) * 1.1

    return SceneNormalization(
        norm_scale=norm_scale,
        norm_center=norm_center,
        scene_scale=scene_scale,
        T_denorm=T_denorm,
    )
