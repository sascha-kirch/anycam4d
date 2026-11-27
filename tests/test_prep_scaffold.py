"""Prep scaffold: stage composition, view validation, the import boundary."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.build import compose
from anycam4d.enums import CameraModelType


def _make_view(cam_id: int, frame_id: int, timestamp: float) -> CalibratedView:
    h, w = 8, 8
    rng = np.random.default_rng(cam_id * 100 + frame_id)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = rng.choice(np.array([0, 255], dtype=np.uint8), size=(h, w))
    K = np.array([[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = cam_id * 0.5
    c2w[2, 3] = 3.0
    return CalibratedView(
        cam_id=cam_id,
        frame_id=frame_id,
        timestamp=timestamp,
        image=image,
        mask=mask,
        model=CameraModelType.PINHOLE,
        K=K,
        params=[50.0, 50.0, 4.0, 4.0],
        c2w=c2w,
    )


def test_compose_applies_left_to_right() -> None:
    view = _make_view(0, 0, 3.0)

    def add_one(v: CalibratedView) -> CalibratedView:
        return dataclasses.replace(v, timestamp=v.timestamp + 1.0)

    def times_two(v: CalibratedView) -> CalibratedView:
        return dataclasses.replace(v, timestamp=v.timestamp * 2.0)

    result = compose([add_one, times_two])(view)

    assert result.timestamp == pytest.approx(8.0)  # (3+1)*2 -- (3*2)+1==7 would prove wrong order


def test_prep_package_does_not_import_torch() -> None:
    """Prep runs without torch."""
    prep_dir = Path(__file__).resolve().parents[1] / "anycam4d" / "data" / "prep"
    for path in prep_dir.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            assert "torch" not in roots, path
