"""``training.batch_size`` auto-resolution."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize_config_dir

from anycam4d.config.experiment import ExperimentConfig
from anycam4d.enums import SamplerType


_TRAIN_PY = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("_train_script", _TRAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _experiment_cfg(sampler: SamplerType, batch_size: int) -> ExperimentConfig:
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        composed = compose(config_name="4dgs", overrides=["recipe=mcmc_spatial", "dataset.data_dir=/nonexistent"])
    composed.training.sampler = sampler.value
    composed.training.batch_size = batch_size
    return ExperimentConfig.from_omegaconf(composed)


@pytest.fixture(scope="module")
def resolve():
    return _load_train_module()._resolve_batch_size


@pytest.mark.parametrize(
    ("sampler", "requested", "expected"),
    [
        (SamplerType.FRAME_ALIGNED, -1, 19),
        (SamplerType.UNIFORM, -1, 1),
        (SamplerType.UNIFORM, 1, 1),
        (SamplerType.FRAME_ALIGNED, 1, 1),
        (SamplerType.FRAME_ALIGNED, 7, 7),
    ],
)
def test_batch_size_resolves_silently(resolve, sampler: SamplerType, requested: int, expected: int) -> None:
    """``-1`` takes the rig under frame-aligned sampling; an explicit value always wins; neither warns."""
    cfg = _experiment_cfg(sampler, requested)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve(cfg, 19)
    assert cfg.training.batch_size == expected
