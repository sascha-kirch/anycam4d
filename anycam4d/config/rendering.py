"""Rendering config dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from anycam4d.enums import BackendType
from anycam4d.enums import PostProcessorType
from anycam4d.enums import RasterizeMode
from anycam4d.enums import RenderMode


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class RenderingConfig:
    """Configuration for the gsplat rasterization adapter."""

    backend: BackendType = BackendType.EWA
    near_plane: float = 0.01  # camera z in normalized scene units (world = norm_scale times this)
    far_plane: float = 1e10

    # Screen-space dilation in pixels², read by exactly two backends: EWA adds it to the 2D
    # covariance it integrates (and compensates for it under rasterize_mode=antialiased), and
    # UT adds it to the 2D covariance that sizes its tile bound. The world rasterizer never sees it — it
    # composites the conditioned Σ unmodified, identically in every view.
    eps2d: float = 0.3

    # Opt-in packed projection layout: a multi-camera memory optimization that renders identically
    # to the dense one. Default off; conf/rendering/*.yaml set it explicitly.
    packed: bool = False

    render_mode: RenderMode = RenderMode.RGB
    rasterize_mode: RasterizeMode = RasterizeMode.CLASSIC
    post_processor: PostProcessorType = PostProcessorType.NONE
    bilagrid_shape: tuple[int, int, int] = field(default_factory=lambda: (16, 16, 8))

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> RenderingConfig:
        """Create RenderingConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        return RenderingConfig(
            backend=BackendType(raw["backend"]),
            near_plane=raw["near_plane"],
            far_plane=raw["far_plane"],
            eps2d=raw["eps2d"],
            packed=raw["packed"],
            render_mode=RenderMode(raw["render_mode"]),
            rasterize_mode=RasterizeMode(raw["rasterize_mode"]),
            post_processor=PostProcessorType(raw["post_processor"]),
            bilagrid_shape=tuple(raw["bilagrid_shape"]),
        )
