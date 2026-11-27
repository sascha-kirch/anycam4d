"""Rebuild ``(model, pipeline)`` from a checkpoint's parameters and resolved config snapshot.

No post-processor is rebuilt: the bilateral grid only knows training images, so eval compensates
appearance with per-image colour correction (``cc_*``), as gsplat does.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import OmegaConf

from anycam4d.config.model import ModelConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.primitives.factory import ModelFactory
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


if TYPE_CHECKING:
    from jaxtyping import Float
    from numpy import ndarray
    from torch import Tensor

    from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def build_pipeline_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
    rendering_override: RenderingConfig | None = None,
    *,
    source: str = "checkpoint",
) -> tuple[FourDGaussians, RenderPipeline]:
    """Rebuild the model and a post-processor-free pipeline; scalar metadata stays on ``checkpoint``.

    Args:
        checkpoint: Dict with ``model_state`` and a ``config`` snapshot.
        device: Model device.
        rendering_override: Rendering config to score through instead of the stored one; the caller
            must check the dataset's scene normalization matches the checkpoint's.
        source: File named in error messages.

    Raises:
        ValueError: If the checkpoint has no ``config`` snapshot.
    """
    model_state = checkpoint["model_state"]
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError(
            "Checkpoint has no 'config' snapshot; cannot reconstruct the render backend. "
            "Re-train with config persistence."
        )

    model_cfg = ModelConfig.from_omegaconf(OmegaConf.create(cfg["model"]))
    if rendering_override is None:
        render_cfg = RenderingConfig.from_omegaconf(OmegaConf.create(cfg["rendering"]))
    else:
        render_cfg = rendering_override

    model = ModelFactory.from_checkpoint(model_state, model_cfg.model_type, device, source=source)
    backend = BackendFactory.create(render_cfg)

    pipeline = RenderPipeline(backend=backend)
    return model, pipeline


def scene_normalization(checkpoint: dict) -> tuple[float, Float[ndarray, "3"]] | None:
    """Training's ``(norm_scale, norm_centre [3])`` with ``p_norm = (p_world - centre) / scale``, or ``None``."""
    if "norm_scale" not in checkpoint or "norm_center" not in checkpoint:
        return None
    return float(checkpoint["norm_scale"]), np.asarray(checkpoint["norm_center"], dtype=np.float64)


def denorm_transform(checkpoint: dict) -> Float[Tensor, "4 4"] | None:
    """``[4, 4]`` normalized-to-world matrix for ``viewmat @ T_denorm``, or ``None`` with a warning."""
    norm = scene_normalization(checkpoint)
    if norm is None:
        warnings.warn(
            "Checkpoint has no norm_scale/norm_center; falling back to an identity denormalization. "
            "Every metric computed against it is meaningless, not merely approximate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    scale, centre = norm
    t_denorm = torch.eye(4, dtype=torch.float32)
    t_denorm[:3, :3] *= scale
    t_denorm[:3, 3] = torch.from_numpy(centre).float()
    return t_denorm
