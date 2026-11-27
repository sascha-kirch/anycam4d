"""The single camera capability gate.

``check_camera_capability`` is the one place that decides whether a ``(model, backend)``
pair is renderable. It is called exactly once per render, in ``RenderPipeline.render``,
before backend dispatch — so an unsupported model can never silently reach a backend.

Derives entirely from the ``CAMERA_REGISTRY`` (model → supported backends) and
``distortion_width`` (model → fixed family slot count). No CUDA import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import distortion_width


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.enums import BackendType
    from anycam4d.enums import CameraModelType


def check_camera_capability(
    model: CameraModelType,
    backend: BackendType,
    distortion: Float[Tensor, "..."] | None,
) -> None:
    """Reject a ``(model, backend)`` pair the registry does not support.

    Args:
        model: Camera projection model for the batch.
        backend: Active render backend.
        distortion: Per-camera distortion ``[..., P]`` or None. When present, its trailing
            width must equal the model's family slot count.

    Raises:
        NotImplementedError: If ``backend`` does not support ``model``.
        ValueError: If ``distortion`` is present with the wrong trailing width.
    """
    spec = CAMERA_REGISTRY[model]

    if backend not in spec.backends:
        raise NotImplementedError(
            f"{backend.value} backend does not support camera model {model.value}; "
            f"supported by: {sorted(b.value for b in spec.backends)}"
        )

    if distortion is not None and distortion.shape[-1] != distortion_width(model):
        raise ValueError(
            f"{model.value}: distortion must be [...,{distortion_width(model)}], got [...,{distortion.shape[-1]}]"
        )
