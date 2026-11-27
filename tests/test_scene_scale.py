"""Tests for camera-based scene normalization."""

from __future__ import annotations

import torch

from anycam4d.data.normalization import compute_scene_norm


def test_compute_scene_norm_from_cameras() -> None:
    """compute_scene_norm from camera positions: scale = max dist from centroid."""
    positions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    scale, center = compute_scene_norm(positions)
    # Centroid at origin, max distance = 1.0
    assert abs(scale - 1.0) < 0.01
    assert torch.allclose(center, torch.zeros(3), atol=1e-5)


def test_compute_scene_norm_single_camera() -> None:
    """Single camera at origin should give scale = 1.0 (fallback for zero)."""
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    scale, center = compute_scene_norm(positions)
    assert scale == 1.0
    assert torch.allclose(center, torch.zeros(3), atol=1e-5)
