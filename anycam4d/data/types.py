"""Core data types for training and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.enums import CameraModelType


@dataclass(frozen=True)
class TrainingSample:
    """A single training sample: one rendered view with camera parameters."""

    image: Float[Tensor, "H W 3"]
    K: Float[Tensor, "3 3"]
    viewmat: Float[Tensor, "4 4"]
    timestamp: float  # normalized to [-0.5, 0.5]; matches original 4DGS time_duration convention
    image_id: int
    camera_id: int
    camera_model: CameraModelType
    # Fixed-width zero-filled canonical distortion vector (always present): [12] pinhole-family
    # (all-zero for PINHOLE), [4] fisheye. Width = ProjectionFamily slot count.
    distortion_params: Float[Tensor, "P"]
    mask: Float[Tensor, "H W 1"] | None = None
    # Observation validity: which pixels are real observations at all (static per camera), as
    # opposed to ``mask``, which says which *observed* pixels are foreground (per frame). ``None``
    # means every camera in the dataset observed every pixel, so consumers take a dense path.
    validity: Bool[Tensor, "H W 1"] | None = None
