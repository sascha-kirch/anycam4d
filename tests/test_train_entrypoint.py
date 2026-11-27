"""Every shipped config survives the startup path: compose, run the resolvers, build the strategy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize_config_dir

from anycam4d.config.experiment import ExperimentConfig
from anycam4d.enums import BackendType
from anycam4d.strategy.strategy_factory import StrategyFactory


_ROOT = Path(__file__).resolve().parents[1]
_CONF_DIR = _ROOT / "conf"
# base.yaml is the shared part the top-level configs include, not a config to run.
_TOP_LEVEL_CONFIGS = sorted(p.stem for p in _CONF_DIR.glob("*.yaml") if p.stem != "base")
_RECIPES = sorted(p.stem for p in (_CONF_DIR / "recipe").glob("*.yaml"))


def _train_module():
    spec = importlib.util.spec_from_file_location("_train_script", _ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose(config_name: str, overrides: list[str] | None = None) -> ExperimentConfig:
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return ExperimentConfig.from_omegaconf(cfg)


@pytest.fixture(scope="module")
def train():
    return _train_module()


def test_there_are_top_level_configs_to_check() -> None:
    """Guard against the parametrization silently covering nothing."""
    assert _TOP_LEVEL_CONFIGS
    assert _RECIPES


@pytest.mark.parametrize("config_name", _TOP_LEVEL_CONFIGS)
def test_shipped_config_resolves_and_builds_its_strategy(config_name: str, train) -> None:
    exp_cfg = _compose(config_name)
    train._resolve_temporal_scales(exp_cfg, n_frames=30)
    train._resolve_batch_size(exp_cfg, cameras_per_frame=19)
    StrategyFactory.create(exp_cfg.strategy, exp_cfg.rendering.backend, exp_cfg.training.max_steps)


@pytest.mark.parametrize("recipe", _RECIPES)
def test_every_recipe_composes_and_builds(recipe: str, train) -> None:
    """Recipes are selectable from the CLI, so each must survive composition on its own."""
    exp_cfg = _compose("4dgs", overrides=[f"recipe={recipe}"])
    train._resolve_temporal_scales(exp_cfg, n_frames=30)
    StrategyFactory.create(exp_cfg.strategy, BackendType.EWA, exp_cfg.training.max_steps)
    assert exp_cfg.strategy.type.value == recipe, "a recipe is named after the strategy it selects"


def test_mcmc_recipe_is_regularized() -> None:
    """MCMC never prunes, so without the opacity regularizer it has no dead set to relocate."""
    assert _compose("4dgs", overrides=["recipe=mcmc_spatial"]).training.opacity_reg_lambda > 0


@pytest.mark.parametrize("recipe", [r for r in _RECIPES if r != "og_4dgs"])
def test_our_recipes_bound_mu_t(recipe: str) -> None:
    """Our strategies cannot remove a primitive drifting out of the span; only the clamp keeps it drawable."""
    assert _compose("4dgs", overrides=[f"recipe={recipe}"]).model.mu_t_margin_sigmas < 0


@pytest.mark.parametrize("config_name", _TOP_LEVEL_CONFIGS)
def test_shipped_configs_use_the_visibility_masked_optimizer(config_name: str) -> None:
    """Dense Adam reproduces a known training failure, and differing optimizers confound backend comparisons."""
    assert _compose(config_name).training.selective_adam is True
