"""Source-level camera exclusion: registry field -> importer -> discovery drops the camera."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.importers.dynerf import DyNeRFImporter
from anycam4d.data.prep.importers.exclusion import drop_excluded
from anycam4d.data.prep.registry import DatasetConfig
from anycam4d.data.prep.registry import LicenseInfo
from anycam4d.data.prep.registry import SceneConfig
from anycam4d.data.prep.registry import build_importer


_CONF = Path(__file__).resolve().parents[1] / "conf" / "datasets"


def _scene_cfg(dataset: str, **kwargs) -> DatasetConfig:
    fields = {"source_subdir": "scene", "fps": 30.0, "eval_cams": [0], "variants": []} | kwargs
    return DatasetConfig(
        dataset=dataset,
        license=LicenseInfo(redistribute=False, source="test"),
        source_url="",
        scenes={"scene": SceneConfig(**fields)},
    )


def test_drop_excluded_removes_ids() -> None:
    assert drop_excluded([0, 1, 2, 9], [9], source="src") == [0, 1, 2]


def test_drop_excluded_ignores_absent_ids() -> None:
    assert drop_excluded([0, 1], [7], source="src") == [0, 1]


def test_drop_excluded_all_raises() -> None:
    with pytest.raises(ValueError, match="nothing left to import"):
        drop_excluded([0, 1], [0, 1], source="src")


def test_eval_camera_cannot_be_excluded() -> None:
    with pytest.raises(ValueError, match="both eval_cams and excluded_cams"):
        _scene_cfg("dynerf", eval_cams=[0], excluded_cams=[0])


def test_registry_forwards_exclusions_to_importer() -> None:
    cfg = _scene_cfg("dynerf", excluded_cams=[9])
    imp = build_importer(cfg, "scene", Path("/src"), Path("/wd"), ColmapOptions())
    assert imp._exclude_cams == (9,)


def test_excluded_camera_never_reaches_calibration_or_extraction(tmp_path: Path) -> None:
    # Discovery is the only place cameras enter the importer, so a dropped id cannot reach
    # COLMAP (frame-0 stills are extracted per discovered camera) nor the ffmpeg extraction plan.
    for cid in (0, 1, 9):
        (tmp_path / f"cam{cid:02d}.mp4").touch()

    imp = DyNeRFImporter(tmp_path, tmp_path / "work", exclude_cams=[9])
    imp._discover_videos()

    assert imp._cam_ids == [0, 1]
