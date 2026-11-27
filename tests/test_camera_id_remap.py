"""Camera ids are remapped to a contiguous 0-based range at load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset


def _build_dataset(base: Path, camera_ids: list[int]) -> None:
    """Write a minimal 2-frame dataset whose cameras use the given (raw) ids."""
    (base / "images").mkdir(parents=True, exist_ok=True)
    frame_times = [0.0, 1.0 / 30.0]

    cameras = [
        {"camera_id": cid, "model": "pinhole", "width": 8, "height": 8, "params": [4.0, 4.0, 4.0, 4.0]}
        for cid in camera_ids
    ]

    captures: list[dict] = []
    for frame_idx, t in enumerate(frame_times):
        for cid in camera_ids:
            name = f"cam{cid:03d}_{frame_idx:04d}"
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(rgb, mode="RGB").save(base / "images" / f"{name}.png")
            captures.append(
                {
                    "image_path": f"images/{name}.png",
                    "camera_id": cid,
                    "frame_id": frame_idx,
                    "timestamp": t,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 3.0],
                }
            )

    schema = {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": cameras,
        "captures": captures,
    }
    with open(base / "train_dataset.json", "w") as f:
        json.dump(schema, f)


@pytest.mark.parametrize("raw_ids", [[1, 2, 3], [0, 2, 5], [10, 11, 12, 13]])
def test_camera_ids_remapped_to_contiguous(tmp_path: Path, raw_ids: list[int]) -> None:
    """1-based and sparse ids both report n_cameras == len(unique) with ids 0..n-1."""
    _build_dataset(tmp_path, raw_ids)
    ds = MultiViewSequenceDataset(tmp_path, split="train")

    n = len(set(raw_ids))
    assert ds.n_cameras == n

    emitted = {ds[i].camera_id for i in range(len(ds))}
    assert emitted == set(range(n)), f"ids must be 0..{n - 1}, got {sorted(emitted)}"


def test_non_uniform_camera_count_raises(tmp_path: Path) -> None:
    """Frames with differing camera counts are rejected at load (fail-fast)."""
    base = tmp_path
    (base / "images").mkdir(parents=True, exist_ok=True)
    cameras = [
        {"camera_id": cid, "model": "pinhole", "width": 8, "height": 8, "params": [4.0, 4.0, 4.0, 4.0]}
        for cid in (0, 1)
    ]
    # frame 0 has 2 cameras, frame 1 has 1 camera → non-uniform.
    plan = [(0.0, [0, 1]), (1.0 / 30.0, [0])]
    captures: list[dict] = []
    for frame_idx, (t, cids) in enumerate(plan):
        for cid in cids:
            name = f"cam{cid:03d}_{t:.3f}"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(base / "images" / f"{name}.png")
            captures.append(
                {
                    "image_path": f"images/{name}.png",
                    "camera_id": cid,
                    "frame_id": frame_idx,
                    "timestamp": t,
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 3.0],
                }
            )
    schema = {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": cameras,
        "captures": captures,
    }
    with open(base / "train_dataset.json", "w") as f:
        json.dump(schema, f)

    with pytest.raises(ValueError, match="Non-uniform camera count"):
        MultiViewSequenceDataset(base, split="train")


@pytest.mark.parametrize("raw_ids", [[1, 2, 3], [0, 2, 5]])
def test_bilateral_grid_indexing_safe(tmp_path: Path, raw_ids: list[int]) -> None:
    """Every emitted camera_id is a valid index into an n_cameras-sized grid (no OOB, no waste)."""
    _build_dataset(tmp_path, raw_ids)
    ds = MultiViewSequenceDataset(tmp_path, split="train")

    emitted = [ds[i].camera_id for i in range(len(ds))]
    assert min(emitted) == 0
    assert max(emitted) == ds.n_cameras - 1  # contiguous: max index uses the last grid slot
