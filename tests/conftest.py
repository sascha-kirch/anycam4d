"""Shared pytest fixtures for anycam4d tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anycam4d.cuda.ext import BUILD_ERROR
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail on a GPU machine whose CUDA extension did not build, instead of skipping every kernel test."""
    if torch.cuda.is_available() and not HAS_CUDA_EXT:
        pytest.exit(f"the anycam4d CUDA extension failed to build ({BUILD_ERROR!r}); run `pixi run cuda-build`", 1)


@pytest.fixture()
def random_model() -> FourDGaussians:
    """A random FourDGaussians on CUDA (or CPU if CUDA unavailable)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FourDGaussians.from_random(n_points=1000, sh_degree=3).to(device)


@pytest.fixture()
def make_grads():
    """Helper that fills deterministic ``.grad`` on every ``model.splats`` param, so optimizer tests need no render."""

    def _make(model: FourDGaussians, seed: int = 0) -> None:
        torch.manual_seed(seed)
        for p in model.splats.values():
            p.grad = torch.randn_like(p)

    return _make


def _identity_quat() -> list[float]:
    """Unit quaternion [qw, qx, qy, qz] = identity rotation."""
    return [1.0, 0.0, 0.0, 0.0]


def _normalized_quat(q: list[float]) -> list[float]:
    """Normalize a quaternion to unit length."""
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


@pytest.fixture(scope="session")
def tmp_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic pinhole dataset: 2 frames (0 s, 1/30 s) x 3 cameras, 64x48 RGBA with binary alpha.

    Writes ``train_dataset.json`` (6 captures) and ``eval_dataset.json`` (2 captures).
    """
    base = tmp_path_factory.mktemp("fourdgs")
    (base / "images").mkdir()

    n_cams = 3
    frame_times = [0.0, 1.0 / 30.0]
    rng = np.random.default_rng(42)

    # Camera definition: pinhole, 64×48, fx=fy=50, cx=32, cy=24
    cameras = [
        {
            "camera_id": cam_idx,
            "model": "pinhole",
            "width": 64,
            "height": 48,
            "params": [50.0, 50.0, 32.0, 24.0],
        }
        for cam_idx in range(n_cams)
    ]

    captures: list[dict] = []
    for frame_idx, t in enumerate(frame_times):
        for cam_idx in range(n_cams):
            rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
            alpha = rng.choice(np.array([0, 255], dtype=np.uint8), size=(48, 64, 1))
            rgba = np.concatenate([rgb, alpha], axis=2)
            img = Image.fromarray(rgba, mode="RGBA")
            name = f"cam{cam_idx:02d}_{frame_idx:04d}"
            img.save(base / "images" / f"{name}.png")

            captures.append(
                {
                    "image_path": f"images/{name}.png",
                    "camera_id": cam_idx,
                    "frame_id": frame_idx,
                    "timestamp": t,
                    "rotation": _identity_quat(),
                    "translation": [cam_idx * 0.5, 0.0, 3.0],
                }
            )

    metadata = {
        "world_coordinate_system": "right_handed",
        "camera_coordinate_system": "opencv",
        "transform_convention": "camera_to_world",
    }

    train_schema = {
        "schema_version": 1,
        "metadata": metadata,
        "cameras": cameras,
        "captures": captures,
    }
    with open(base / "train_dataset.json", "w") as f:
        json.dump(train_schema, f)

    # Eval set: first camera of each frame
    eval_captures = [captures[0], captures[n_cams]]
    eval_schema = {
        "schema_version": 1,
        "metadata": metadata,
        "cameras": cameras,
        "captures": eval_captures,
    }
    with open(base / "eval_dataset.json", "w") as f:
        json.dump(eval_schema, f)

    return base


@pytest.fixture(scope="session")
def tmp_dataset_mixed_formats(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """1 frame x 2 cameras: lossless WEBP with alpha, and JPEG without; train split only."""
    base = tmp_path_factory.mktemp("mixed_formats")
    (base / "images").mkdir()

    w, h = 64, 48
    rng = np.random.default_rng(11)

    cameras = [
        {"camera_id": cam_idx, "model": "pinhole", "width": w, "height": h, "params": [50.0, 50.0, 32.0, 24.0]}
        for cam_idx in range(2)
    ]

    rgb0 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha0 = rng.choice(np.array([0, 255], dtype=np.uint8), size=(h, w, 1))
    rgba0 = np.concatenate([rgb0, alpha0], axis=2)
    Image.fromarray(rgba0, mode="RGBA").save(base / "images" / "cam00_0000.webp", lossless=True)

    rgb1 = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(rgb1, mode="RGB").save(base / "images" / "cam01_0000.jpg", quality=95)

    captures = [
        {
            "image_path": "images/cam00_0000.webp",
            "camera_id": 0,
            "frame_id": 0,
            "timestamp": 0.0,
            "rotation": _identity_quat(),
            "translation": [0.0, 0.0, 3.0],
        },
        {
            "image_path": "images/cam01_0000.jpg",
            "camera_id": 1,
            "frame_id": 0,
            "timestamp": 0.0,
            "rotation": _identity_quat(),
            "translation": [0.5, 0.0, 3.0],
        },
    ]

    metadata = {
        "world_coordinate_system": "right_handed",
        "camera_coordinate_system": "opencv",
        "transform_convention": "camera_to_world",
    }
    schema = {"schema_version": 1, "metadata": metadata, "cameras": cameras, "captures": captures}
    with open(base / "train_dataset.json", "w") as f:
        json.dump(schema, f)

    return base
