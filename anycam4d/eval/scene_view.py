"""One trained model and one eval view, loaded for the forward-only render scripts (any camera, RS, DoF, lens morph)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.valid_mask import pack_validity
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import BackendType
from anycam4d.enums import RasterizeMode
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.eval.checkpoint import denorm_transform


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.data.types import TrainingSample
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.backend import RenderBackend


@dataclass(frozen=True)
class EvalView:
    """A model, a UT rendering config, the backend it was trained with, the eval split and one sample with its camera."""

    model: FourDGaussians
    cfg: RenderingConfig
    trained_backend: RenderBackend
    dataset: MultiViewSequenceDataset
    sample: TrainingSample
    camera: BatchedCamera


def load_eval_view(checkpoint: str | Path, data_dir: str | Path, index: int) -> EvalView:
    """Load ``checkpoint`` and eval sample ``index`` of ``data_dir`` in the checkpoint's normalized frame.

    The config is switched to the UT backend with classic rasterization, so an EWA-trained checkpoint renders through
    any lens; ``camera`` is the sample's own lens with its validity mask.
    """
    device = torch.device("cuda")
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model, pipeline = build_pipeline_from_checkpoint(state, device, source=str(checkpoint))
    model.eval()
    dataset = MultiViewSequenceDataset(str(data_dir), split="eval")
    t_denorm = denorm_transform(state)
    if t_denorm is not None:
        dataset.apply_viewmat_transform(t_denorm)
    sample = dataset[index]
    validity = dataset.validity_for(sample.camera_id)
    camera = BatchedCamera(
        viewmats=sample.viewmat.to(device)[None],
        Ks=sample.K.to(device)[None],
        model=sample.camera_model,
        width=sample.image.shape[1],
        height=sample.image.shape[0],
        distortion=sample.distortion_params.to(device)[None],
        valid_mask=None if validity is None else pack_validity(validity.to(device))[None].contiguous(),
    )
    cfg = dataclasses.replace(pipeline.render_cfg, backend=BackendType.ANYCAM4D, rasterize_mode=RasterizeMode.CLASSIC)
    return EvalView(
        model=model, cfg=cfg, trained_backend=pipeline.backend, dataset=dataset, sample=sample, camera=camera
    )


def to_bgr8(colors: Float[Tensor, "1 H W 3"], label: str = "") -> np.ndarray:
    """Render → BGR uint8 for OpenCV, with an optional white label top left."""
    image = (colors[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)[..., ::-1].copy()
    if label:
        cv2.putText(image, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image
