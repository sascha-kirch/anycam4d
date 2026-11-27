"""Top-level experiment config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.strategy import StrategyConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.config.viewer import ViewerConfig


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class ExperimentConfig:
    """Top-level composed config — one instance per training run."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    strategy: StrategyConfig
    rendering: RenderingConfig
    viewer: ViewerConfig

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> ExperimentConfig:
        """Create ExperimentConfig from OmegaConf DictConfig."""
        return ExperimentConfig(
            dataset=DatasetConfig.from_omegaconf(cfg.dataset),
            model=ModelConfig.from_omegaconf(cfg.model),
            training=TrainingConfig.from_omegaconf(cfg.training),
            strategy=StrategyConfig.from_omegaconf(cfg.strategy),
            rendering=RenderingConfig.from_omegaconf(cfg.rendering),
            viewer=ViewerConfig.from_omegaconf(cfg.viewer),
        )
