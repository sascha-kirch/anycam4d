"""PLY round trip and the torch loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from anycam4d.data.point_cloud import load_ply
from anycam4d.point_ops.ply import read_ply
from anycam4d.point_ops.ply import write_ply


def test_write_then_read_ply_round_trips_colored_cloud(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(20, 3), dtype=np.uint8)

    path = tmp_path / "cloud.ply"
    write_ply(path, xyz, rgb)
    xyz_out, rgb_out = read_ply(path)

    np.testing.assert_array_equal(xyz_out, xyz)
    np.testing.assert_array_equal(rgb_out, rgb)


def test_write_then_read_ply_round_trips_uncolored_cloud(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1.0, 1.0, size=(10, 3)).astype(np.float32)

    path = tmp_path / "gray_cloud.ply"
    write_ply(path, xyz, rgb=None)
    xyz_out, rgb_out = read_ply(path)

    np.testing.assert_array_equal(xyz_out, xyz)
    # write_ply defaults uncolored points to gray 128 — read_ply sees those columns as present.
    assert rgb_out is not None
    assert np.all(rgb_out == 128)


def test_load_ply_wraps_read_ply_as_torch_with_normalized_colors(tmp_path: Path) -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rgb = np.array([[0, 128, 255], [255, 0, 128]], dtype=np.uint8)

    path = tmp_path / "cloud.ply"
    write_ply(path, xyz, rgb)

    points, colors = load_ply(path)

    assert isinstance(points, torch.Tensor)
    assert points.dtype == torch.float32
    torch.testing.assert_close(points, torch.from_numpy(xyz))

    assert colors is not None
    assert isinstance(colors, torch.Tensor)
    assert colors.dtype == torch.float32
    assert colors.min() >= 0.0
    assert colors.max() <= 1.0
    torch.testing.assert_close(colors, torch.from_numpy(rgb.astype(np.float32) / 255.0))
