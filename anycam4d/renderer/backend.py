"""RenderBackend protocol and RenderOutput type for 4D Gaussian rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import runtime_checkable

from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians


@dataclass(frozen=True)
class RenderOutput:
    """Typed wrapper for rasterization output."""

    colors: Float[Tensor, "C H W D"]  # rendered colors (or RGB+depth)
    alphas: Float[Tensor, "C H W 1"]  # rendered alpha
    meta: dict[str, Any]  # full gsplat meta dict


@runtime_checkable
class RenderBackend(Protocol):
    """Full render path: 4D Gaussians → pixels.

    The backend owns conditioning, appearance evaluation (SH/4D-SH),
    projection, and compositing. Hides the choice of rasterization
    method (EWA, UT, ray-tracing) and execution mode (PyTorch, fused CUDA).
    """

    @property
    def cfg(self) -> RenderingConfig:
        """Rendering configuration used by this backend."""
        ...

    def render(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        *,
        sh_degree_t: int = MAX_TEMPORAL_DEGREE,
        backgrounds: Float[Tensor, "C 3"] | None = None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Render 4D Gaussians to image(s).

        Args:
            model: 4D Gaussian parameter container.
            camera: BatchedCamera carrier (viewmats, Ks, model, width, height, distortion).
            tau: Normalized timestamp for conditioning.
            sh_degree: Spatial SH degree for color evaluation (supports curriculum).
            sh_degree_t: Temporal degree; the model maximum unless a curriculum lowers it.
            backgrounds: Optional per-camera background [C, 3].
            colors_override: Bypass appearance model with explicit per-Gaussian RGB [N, 3].

        Returns:
            RenderOutput with colors, alphas, meta.
        """
        ...
