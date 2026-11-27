"""Viewer config dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import OmegaConf


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class ViewerConfig:
    """Interactive viser viewer settings."""

    enabled: bool = True
    port: int = 8080
    playback_fps: float = 30.0
    update_every_n_steps: int = 50

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> ViewerConfig:
        """Create ViewerConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return ViewerConfig(
            enabled=raw["enabled"],
            port=raw["port"],
            playback_fps=raw["playback_fps"],
            update_every_n_steps=raw["update_every_n_steps"],
        )
