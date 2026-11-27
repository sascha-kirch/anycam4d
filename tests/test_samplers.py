"""Temporal tiled sampler: tiles are one camera's consecutive frames, and an epoch covers every view once."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.samplers import TemporalTiledSampler


@pytest.mark.parametrize("tile_len", [1, 2])
def test_tiles_are_consecutive_frames_of_one_camera_covering_the_epoch(tmp_dataset: Path, tile_len: int) -> None:
    """At ``tile_len=1`` this is a plain shuffle over views, the same distribution as ``uniform``."""
    ds = MultiViewSequenceDataset(tmp_dataset, split="train")
    order = list(TemporalTiledSampler(ds, tile_len=tile_len, shuffle=True))

    assert sorted(order) == list(range(len(ds)))
    frame_rank = {fid: r for r, fid in enumerate(ds.frame_ids)}
    for start in range(0, len(order), tile_len):
        tile = order[start : start + tile_len]
        assert len({ds.camera_ids[i] for i in tile}) == 1
        ranks = [frame_rank[ds.frame_id_for(i)] for i in tile]
        assert ranks == list(range(ranks[0], ranks[0] + tile_len))
        assert ranks[0] % tile_len == 0


def test_rejects_frames_that_do_not_tile(tmp_dataset: Path) -> None:
    ds = MultiViewSequenceDataset(tmp_dataset, split="train")
    with pytest.raises(ValueError, match=r"2 frames do not split into tiles of 3"):
        TemporalTiledSampler(ds, tile_len=3)
