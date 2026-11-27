"""Dataset manifest validation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from anycam4d.data.schema import CaptureSchema
from anycam4d.data.schema import DatasetSchema


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rotation", [2.0, 0.0, 0.0, 0.0], "unit-norm"),
        ("rotation", [1.0, 0.0, 0.0], "4 components"),
        ("translation", [0.0, 3.0], "3 components"),
    ],
)
def test_capture_rejects_a_malformed_pose(field: str, value: list[float], message: str) -> None:
    capture = {
        "image_path": "images/cam00_0000.png",
        "camera_id": 0,
        "timestamp": 0.0,
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 3.0],
        field: value,
    }
    with pytest.raises(ValidationError, match=message):
        CaptureSchema(**capture)


def _minimal_schema_dict() -> dict:
    return {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": [
            {"camera_id": 0, "model": "pinhole", "width": 64, "height": 48, "params": [50.0, 50.0, 32.0, 24.0]},
        ],
        "captures": [
            {
                "image_path": "images/cam00_0000.png",
                "camera_id": 0,
                "frame_id": 0,
                "timestamp": 0.0,
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 3.0],
            }
        ],
    }


def test_dataset_schema_missing_camera_id_raises() -> None:
    """Capture referencing a non-existent camera_id raises ValidationError."""
    d = _minimal_schema_dict()
    d["captures"][0]["camera_id"] = 99  # doesn't exist
    with pytest.raises(ValidationError, match="camera_id=99 not in cameras"):
        DatasetSchema.model_validate(d)


def test_documentation_full_example_validates() -> None:
    """The full JSON example in ``README.md`` validates and keeps the frame invariants."""
    md = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    blocks = re.findall(r"```json\n(.*?)```", md, re.S)
    full = [b for b in blocks if '"captures"' in b and '"cameras"' in b and "..." not in b]
    assert full, "README.md has no full-document JSON example"

    for block in full:
        doc = DatasetSchema.model_validate(json.loads(block))
        by_frame: dict[int, list] = defaultdict(list)
        for cap in doc.captures:
            by_frame[cap.frame_id].append(cap)
        assert len(by_frame) > 1, "the example should show more than one frame, or frame_id looks vestigial"
        assert len({len(caps) for caps in by_frame.values()}) == 1, "example rig is ragged"
        for frame_id, caps in by_frame.items():
            assert len({c.timestamp for c in caps}) == 1, f"example frame {frame_id} has mixed timestamps"
