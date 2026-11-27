"""Dataset registry: shipped YAMLs load, importers dispatch, variants map to stages."""

from __future__ import annotations

from pathlib import Path

import pytest

from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.importers.deepview import DeepViewImporter
from anycam4d.data.prep.importers.dynerf import DyNeRFImporter
from anycam4d.data.prep.importers.selfcap import SelfCapImporter
from anycam4d.data.prep.registry import DatasetConfig
from anycam4d.data.prep.registry import build_importer
from anycam4d.data.prep.registry import load_registry
from anycam4d.data.prep.registry import variant_spec
from anycam4d.data.prep.stages.resize import Resize
from anycam4d.data.prep.stages.to_pinhole import ToPinhole


_CONF = Path(__file__).resolve().parents[1] / "conf" / "datasets"


@pytest.mark.parametrize("name", ["dynerf", "selfcap", "deepview", "dynopsis"])
def test_shipped_registry_loads(name: str) -> None:
    cfg = load_registry(_CONF / f"{name}.yaml")
    assert isinstance(cfg, DatasetConfig)
    assert cfg.dataset == name
    assert cfg.source_url
    assert cfg.license.redistribute is (name == "dynopsis")  # only the paper's own dataset is redistributable
    assert cfg.scenes  # at least one scene


@pytest.mark.parametrize(
    ("dataset", "scene", "importer"),
    [
        ("dynerf", "cook_spinach", DyNeRFImporter),
        ("deepview", "welder", DeepViewImporter),
        ("selfcap", "bike", SelfCapImporter),
    ],
)
def test_build_importer_dispatches_on_dataset(dataset: str, scene: str, importer: type) -> None:
    cfg = load_registry(_CONF / f"{dataset}.yaml")
    assert isinstance(build_importer(cfg, scene, Path("/src"), Path("/wd"), ColmapOptions()), importer)


def test_build_importer_selfcap_relative_calib_dir() -> None:
    # Some releases ship calibration as a sibling directory; a relative calib_dir resolves
    # against source_root, not the scene dir.
    cfg = load_registry(_CONF / "selfcap.yaml")
    cfg.scenes["bike"].calib_dir = "bike-calib"
    imp = build_importer(cfg, "bike", Path("/data/self"), Path("/wd"), ColmapOptions())
    assert imp._calib_dir == Path("/data/self/bike-calib")


def test_build_importer_unknown_dataset_raises() -> None:
    cfg = load_registry(_CONF / "dynerf.yaml")
    cfg.dataset = "nope"
    with pytest.raises(ValueError, match="unknown dataset"):
        build_importer(cfg, "cook_spinach", Path("/src"), Path("/wd"), ColmapOptions())


@pytest.mark.parametrize(
    ("variant", "scale", "stages"),
    [
        ("native", 1.0, []),
        ("native", 2.0, [Resize]),
        ("undistorted", 1.0, [ToPinhole]),
        ("undistorted", 2.0, [Resize, ToPinhole]),
    ],
)
def test_variant_spec_stages(variant: str, scale: float, stages: list[type]) -> None:
    """Scale resizes first, so the projection stage derives its grid from the smaller image."""
    spec = variant_spec(variant, scale=scale)
    assert [type(s) for s in spec.stages] == stages
    if stages:
        assert spec.image_format == "jpg"


def test_variant_spec_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        variant_spec("bogus")
