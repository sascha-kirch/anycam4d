"""Scene normalization."""

from __future__ import annotations

from pathlib import Path

import torch

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization


def test_normalized_viewmats_are_rigid_so_the_kernel_camera_centre_is_right(tmp_dataset: Path) -> None:
    """The fused kernels take the camera centre as ``-Rᵀt``; that holds only if the rotation block stays orthonormal."""
    train = MultiViewSequenceDataset(tmp_dataset, split="train")
    norm = compute_normalization(train)
    world = [v.clone() for v in train.viewmats]
    train.apply_viewmat_transform(norm.T_denorm)

    for v_world, v in zip(world, train.viewmats, strict=True):
        R, t = v[:3, :3], v[:3, 3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        centre_world = torch.linalg.inv(v_world)[:3, 3]
        assert torch.allclose(-R.T @ t, (centre_world - norm.norm_center) / norm.norm_scale, atol=1e-5)
