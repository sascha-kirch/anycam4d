"""Dataset config dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    """MultiViewSequence dataset (train_dataset.json / eval_dataset.json format)."""

    data_dir: Path
    scale_factor: int = 1

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> DatasetConfig:
        """Create DatasetConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)

        return DatasetConfig(
            data_dir=Path(raw["data_dir"]),
            scale_factor=raw["scale_factor"],
        )
