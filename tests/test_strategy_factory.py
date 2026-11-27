"""StrategyFactory covers every strategy type and checks the schedule against the run."""

from __future__ import annotations

import warnings

import pytest

from anycam4d.config.strategy import StrategyConfig
from anycam4d.enums import BackendType
from anycam4d.enums import StrategyType
from anycam4d.strategy.coupled import OGFourDGSStrategy
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.strategy.null_strategy import NullStrategy
from anycam4d.strategy.strategy_factory import StrategyFactory


EXPECTED = {
    StrategyType.DEFAULT_SPATIAL: DefaultSpatialStrategy,
    StrategyType.OG_4DGS: OGFourDGSStrategy,
    StrategyType.MCMC_SPATIAL: MCMCSpatialStrategy,
    StrategyType.NULL_4DGS: NullStrategy,
}


def _build(strategy_type: StrategyType, **overrides: object) -> object:
    cfg = StrategyConfig(type=strategy_type, cap_max=1000, **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StrategyFactory.create(cfg, BackendType.EWA, max_steps=30_000)


def test_factory_threads_min_opacity_to_the_mcmc_strategy() -> None:
    assert _build(StrategyType.MCMC_SPATIAL, min_opacity=0.02).min_opacity == 0.02


@pytest.mark.parametrize("strategy_type", list(EXPECTED))
def test_every_strategy_type_builds(strategy_type: StrategyType) -> None:
    assert isinstance(_build(strategy_type), EXPECTED[strategy_type])


def test_the_factory_covers_the_whole_enum() -> None:
    """A new StrategyType with no factory branch would raise only at run time."""
    assert set(EXPECTED) == set(StrategyType)


def test_the_schedule_is_validated_against_the_run() -> None:
    """The factory is the one place the schedule and the run length are checked to agree."""
    cfg = StrategyConfig(type=StrategyType.MCMC_SPATIAL, cap_max=1000, refine_stop_iter=25_000)
    with pytest.raises(ValueError, match="past training.max_steps"):
        StrategyFactory.create(cfg, BackendType.EWA, max_steps=15_000)
