"""Per-frame metric curves carry their time axis, and the static-fraction threshold."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.eval.evaluator import _aggregate


def test_aggregate_carries_the_time_axis_into_the_payload() -> None:
    """The per-frame curve is useless without the timestamps it is indexed on."""
    result = _aggregate(
        {"psnr": [30.0, 20.0]},
        num_timesteps=2,
        num_cameras=1,
        num_images=2,
        timestamps=[-0.5, 0.5],
    )
    payload = result.to_dict()
    assert payload["per_timestamp"]["psnr"] == [30.0, 20.0]
    assert payload["timestamps"] == [-0.5, 0.5]
    assert result.mean["psnr"] == pytest.approx(25.0)
