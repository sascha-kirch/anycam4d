"""Densification schedule validation, as absolute step counts checked against the run length."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from anycam4d.config.strategy import StrategyConfig
from anycam4d.enums import StrategyType


_STRATEGY_DIR = Path(__file__).parents[1] / "conf" / "strategy"
_STRATEGY_YAMLS = sorted(_STRATEGY_DIR.glob("*.yaml"))


def _mcmc(**overrides: object) -> StrategyConfig:
    """Build an MCMC ``StrategyConfig`` — the only family the noise schedule applies to."""
    return _from_dict(type=StrategyType.MCMC_SPATIAL.value, cap_max=1000, **overrides)


def _from_dict(**overrides: object) -> StrategyConfig:
    """Build a ``StrategyConfig`` through ``from_omegaconf`` from a raw dict of overrides."""
    base: dict[str, object] = {"type": StrategyType.DEFAULT_SPATIAL.value}
    base.update(overrides)
    return StrategyConfig.from_omegaconf(OmegaConf.create(base))


def test_empty_refinement_window_is_rejected() -> None:
    """A stop at or below the start means densification never fires, so it raises at config time."""
    with pytest.raises(ValueError, match="refine_stop_iter"):
        _from_dict(refine_start_iter=5_000, refine_stop_iter=5_000)
    with pytest.raises(ValueError, match="refine_stop_iter"):
        _from_dict(refine_start_iter=5_000, refine_stop_iter=4_000)


def test_a_stop_past_the_run_is_rejected() -> None:
    """The 447-run defect: a 25 000-step stop on a 15 000-step run never stops anything."""
    cfg = _from_dict(refine_start_iter=500, refine_stop_iter=25_000)
    with pytest.raises(ValueError, match="past training.max_steps"):
        cfg.validate_against_run(15_000)


def test_densifying_for_the_whole_run_is_expressible() -> None:
    """Raising rather than warning refuses nothing legitimate: stop == max_steps is allowed."""
    _from_dict(refine_start_iter=500, refine_stop_iter=15_000).validate_against_run(15_000)


def test_a_start_past_the_run_is_rejected() -> None:
    """The 13_freeze part-5 shape: a probe run too short for its own warm-up densifies never."""
    cfg = _from_dict(refine_start_iter=500, refine_stop_iter=1_000)
    with pytest.raises(ValueError, match="densification never starts"):
        cfg.validate_against_run(500)


def test_noise_stop_past_refinement_is_rejected() -> None:
    """Injecting noise into a population densification has already frozen is a config error."""
    with pytest.raises(ValueError, match="noise_injection_stop_iter"):
        _mcmc(refine_stop_iter=10_000, noise_injection_stop_iter=12_000)


def test_noise_stop_past_the_run_is_rejected() -> None:
    cfg = _mcmc(refine_stop_iter=30_000, noise_injection_stop_iter=25_000)
    with pytest.raises(ValueError, match="noise_injection_stop_iter"):
        cfg.validate_against_run(20_000)


def test_the_noise_stop_is_inert_for_the_gradient_families() -> None:
    """They inject no noise, so the field is not checked against their shorter window."""
    assert _from_dict(refine_stop_iter=10_000, noise_injection_stop_iter=12_000).refine_stop_iter == 10_000
