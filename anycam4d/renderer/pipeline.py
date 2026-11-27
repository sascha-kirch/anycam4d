"""RenderPipeline — thin orchestrator: backend + post-processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.renderer.backend import RenderOutput
from anycam4d.renderer.capability import check_camera_capability


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.backend import RenderBackend
    from anycam4d.renderer.post_processor import PostProcessor


class RenderPipeline:
    """Thin orchestrator: backend + post-processor.

    The backend handles the full 4D→pixels path (conditioning, appearance,
    projection, compositing). The pipeline adds optional post-processing
    (e.g. bilateral grid) and delegates everything else to the backend.

    Used identically by the trainer (with gradients), viewer (without), and
    eval script.

    Args:
        backend: Render backend (handles conditioning, appearance, rasterization).
        post_processor: Optional post-processor for rendered colors.
    """

    def __init__(
        self,
        *,
        backend: RenderBackend,
        post_processor: PostProcessor | None = None,
    ) -> None:
        """Initialize pipeline with backend and optional post-processor."""
        self._backend = backend
        self._post_processor = post_processor

    @property
    def backend(self) -> RenderBackend:
        """The render backend."""
        return self._backend

    @property
    def post_processor(self) -> PostProcessor | None:
        """The post-processor, or None if not used."""
        return self._post_processor

    @property
    def render_cfg(self) -> RenderingConfig:
        """Rendering configuration (delegated to backend)."""
        return self._backend.cfg

    def render(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        timestamp: float,
        sh_degree: int,
        *,
        sh_degree_t: int = MAX_TEMPORAL_DEGREE,
        image_ids: list[int] | None = None,
        camera_ids: list[int] | None = None,
        backgrounds: Float[Tensor, "C 3"] | None = None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Full rendering pipeline: backend render + optional post-processing.

        Args:
            model: 4D Gaussian parameter container.
            camera: BatchedCamera carrier (viewmats, Ks, model, width, height, distortion).
            timestamp: Normalized timestamp for conditioning.
            sh_degree: Spatial SH degree to evaluate (supports curriculum).
            sh_degree_t: Temporal degree to evaluate; the model maximum unless a curriculum lowers it.
            image_ids: Optional image IDs for post-processing context.
            camera_ids: Optional camera IDs for post-processing context.
            backgrounds: Optional per-camera background color [C, 3] in [0, 1].
            colors_override: Optional per-Gaussian RGB [N, 3] bypassing appearance model and SH.

        Returns:
            RenderOutput with rendered colors, alphas, and rasterization metadata.
        """
        # Single capability gate: runs exactly once per render, before backend
        # dispatch, so a backend can never silently receive an unsupported camera model.
        check_camera_capability(camera.model, self.render_cfg.backend, camera.distortion)

        out = self._backend.render(
            model,
            camera,
            timestamp,
            sh_degree,
            sh_degree_t=sh_degree_t,
            backgrounds=backgrounds,
            colors_override=colors_override,
        )

        if self._post_processor is not None:
            colors = self._post_processor(out.colors, image_ids=image_ids, camera_ids=camera_ids)
            out = RenderOutput(colors=colors, alphas=out.alphas, meta=out.meta)

        return out
