"""PostProcessor protocol for rendered image post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for post-processing rendered images.

    Implementors transform rendered colors after rasterization. The first
    planned implementation is a bilateral grid (appearance compensation).

    Post-processors may be stateful (e.g., learnable parameters in BilagGrid)
    and receive optional image/camera IDs for per-image conditioning.
    """

    def __call__(
        self,
        colors: Float[Tensor, "C H W D"],
        *,
        image_ids: list[int] | None = None,
        camera_ids: list[int] | None = None,
    ) -> Float[Tensor, "C H W D"]:
        """Apply post-processing to rendered colors.

        Args:
            colors: Rendered color tensor from the rasterizer.
            image_ids: Optional per-batch image indices (for per-image conditioning).
            camera_ids: Optional per-batch camera indices (for per-camera conditioning).

        Returns:
            Post-processed color tensor (same shape as input).
        """
        ...
