"""RenderPipeline hands the render to the post-processor."""

from __future__ import annotations

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")

_HEIGHT = 64
_WIDTH = 64
_FOCAL = 50.0


def _make_viewmat(device: torch.device) -> torch.Tensor:
    v = torch.eye(4, device=device, dtype=torch.float32)
    v[2, 3] = 3.0
    return v


def _make_K(device: torch.device) -> torch.Tensor:
    cx, cy = _WIDTH / 2.0, _HEIGHT / 2.0
    return torch.tensor(
        [[_FOCAL, 0.0, cx], [0.0, _FOCAL, cy], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )


def _make_camera(device: torch.device) -> BatchedCamera:
    return BatchedCamera.from_single_view(
        _make_viewmat(device), _make_K(device), CameraModelType.PINHOLE, _WIDTH, _HEIGHT
    )


@requires_cuda
def test_pipeline_post_processor_called(random_model: FourDGaussians) -> None:
    """PostProcessor is invoked and its output replaces the rendered colors."""
    device = next(random_model.parameters()).device
    sentinel = torch.ones(1, _HEIGHT, _WIDTH, 3, device=device) * 0.42
    called_with: dict = {}

    class MockPostProcessor:
        def __call__(self, colors, *, image_ids=None, camera_ids=None):
            called_with["colors_shape"] = colors.shape
            called_with["image_ids"] = image_ids
            called_with["camera_ids"] = camera_ids
            return sentinel

    pipeline = RenderPipeline(
        backend=BackendFactory.create(RenderingConfig()),
        post_processor=MockPostProcessor(),
    )

    with torch.no_grad():
        out = pipeline.render(
            random_model,
            _make_camera(device),
            timestamp=0.0,
            sh_degree=random_model.sh_degree,
            image_ids=[7],
            camera_ids=[2],
        )

    assert called_with["colors_shape"] == (1, _HEIGHT, _WIDTH, 3)
    assert called_with["image_ids"] == [7]
    assert called_with["camera_ids"] == [2]
    assert torch.allclose(out.colors, sentinel)
