"""``quat_translation_to_c2w``."""

from __future__ import annotations

import math

import torch

from anycam4d.math.transforms import quat_translation_to_c2w


def test_identity_quat_with_translation() -> None:
    """Identity rotation, non-zero translation → identity 3x3 block + translation in last column."""
    c2w = quat_translation_to_c2w([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert torch.allclose(c2w[:3, :3], torch.eye(3), atol=1e-6)
    assert torch.allclose(c2w[:3, 3], torch.tensor([1.0, 2.0, 3.0]), atol=1e-6)
    assert torch.allclose(c2w[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6)


def test_known_quaternion_90deg_about_z() -> None:
    """A 90° rotation about +z maps x̂ → ŷ in the rotation block."""
    half = math.pi / 4.0  # half-angle for 90°
    quat = [math.cos(half), 0.0, 0.0, math.sin(half)]  # [qw, qx, qy, qz]
    c2w = quat_translation_to_c2w(quat, [0.0, 0.0, 0.0])
    rotated_x = c2w[:3, :3] @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(rotated_x, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
