"""Factory for building RenderBackend instances from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycam4d.console import notice
from anycam4d.enums import BackendType
from anycam4d.enums import PostProcessorType
from anycam4d.renderer.bilateral_grid_post_processor import BilateralGridPostProcessor
from anycam4d.renderer.ewa_fused_backend import FusedEWABackend
from anycam4d.renderer.render_registry import check_render_capability
from anycam4d.renderer.ut_fused_backend import UTFusedBackend


if TYPE_CHECKING:
    import torch

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.renderer.backend import RenderBackend
    from anycam4d.renderer.post_processor import PostProcessor


_BACKENDS: dict[BackendType, type[RenderBackend]] = {
    BackendType.EWA: FusedEWABackend,
    BackendType.ANYCAM4D: UTFusedBackend,
}


class BackendFactory:
    """Creates RenderBackend from config."""

    @staticmethod
    def create(cfg: RenderingConfig) -> RenderBackend:
        """Build a render backend from config.

        Args:
            cfg: Rendering configuration.

        Returns:
            Configured RenderBackend ready for use.
        """
        backend = _BACKENDS.get(cfg.backend)
        if backend is None:
            raise ValueError(f"Unsupported backend: {cfg.backend}")

        check_render_capability(cfg.backend, cfg.rasterize_mode)
        return backend(cfg)


class PostProcessorFactory:
    """Creates PostProcessor from config."""

    @staticmethod
    def create(
        cfg: RenderingConfig,
        n_cameras: int,
        device: torch.device,
    ) -> PostProcessor | None:
        """Build a post-processor from config.

        Args:
            cfg: Rendering configuration (includes post-processor type and parameters).
            n_cameras: Number of cameras in the scene.
            device: Device to place the post-processor on.

        Returns:
            Configured PostProcessor or None if not used.
        """
        post_processor = None

        if cfg.post_processor == PostProcessorType.BILATERAL_GRID:
            post_processor = BilateralGridPostProcessor(
                n_cameras=n_cameras,
                grid_shape=cfg.bilagrid_shape,
            ).to(device)
            notice(f"Bilateral grid post-processor: {n_cameras} cameras, grid_shape={cfg.bilagrid_shape}")

        return post_processor
