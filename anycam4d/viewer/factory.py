"""Factory for creating 4DGS viewers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import viser

from anycam4d.viewer.viewer import Anycam4DViewer


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.config.viewer import ViewerConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.pipeline import RenderPipeline


class ViewerFactory:
    """Factory for creating 4DGS viewers."""

    @staticmethod
    def create(
        viewer_cfg: ViewerConfig,
        model: FourDGaussians,
        pipeline: RenderPipeline,
        num_timesteps: int = 100,
        T_denorm: Float[Tensor, "4 4"] | None = None,
        output_dir: Path | None = None,
    ) -> Anycam4DViewer | None:
        """Create the appropriate viewer for the given pipeline.

        Args:
            viewer_cfg: Viewer configuration.
            model: Trained FourDGaussians model.
            pipeline: Render pipeline (backend + optional post-processor).
            num_timesteps: Number of discrete timesteps on the slider.
            T_denorm: Optional normalised-space-to-world transform.
            output_dir: Directory for saved trajectories and rendered videos.

        Returns:
            A configured Anycam4DViewer, or None if viewer is disabled.
        """
        if not viewer_cfg.enabled:
            return None

        # Hydra logs at INFO; the websocket library would print every browser connection into the progress bar.
        logging.getLogger("websockets").setLevel(logging.WARNING)
        server = viser.ViserServer(port=viewer_cfg.port, verbose=False, label="AnyCam4D Viewer")
        return Anycam4DViewer(
            server,
            model,
            pipeline,
            num_timesteps=num_timesteps,
            fps=viewer_cfg.playback_fps,
            T_denorm=T_denorm,
            update_every_n_steps=viewer_cfg.update_every_n_steps,
            output_dir=output_dir,
        )
