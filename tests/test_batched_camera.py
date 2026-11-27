"""BatchedCamera validates its fields and moves every tensor it carries."""

from __future__ import annotations

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.enums import CameraModelType


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")

_HEIGHT = 64
_WIDTH = 64
_FOCAL = 50.0


def _viewmat(device: torch.device) -> torch.Tensor:
    v = torch.eye(4, device=device, dtype=torch.float32)
    v[2, 3] = 3.0
    return v


def _K(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[_FOCAL, 0.0, _WIDTH / 2.0], [0.0, _FOCAL, _HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )


def test_batchedcamera_postinit_raises() -> None:
    """Mismatched leading dim, wrong rank, or mixed devices raise AssertionError."""
    good_v = torch.eye(4).unsqueeze(0)
    good_k = torch.eye(3).unsqueeze(0)

    with pytest.raises(AssertionError):  # Ks leading dim != viewmats
        BatchedCamera(good_v, torch.eye(3).expand(2, 3, 3).contiguous(), CameraModelType.PINHOLE, _WIDTH, _HEIGHT)

    with pytest.raises(AssertionError):  # distortion wrong rank
        BatchedCamera(good_v, good_k, CameraModelType.FISHEYE_OPENCV, _WIDTH, _HEIGHT, torch.zeros(4))

    with pytest.raises(AssertionError):  # distortion leading dim != C
        BatchedCamera(good_v, good_k, CameraModelType.FISHEYE_OPENCV, _WIDTH, _HEIGHT, torch.zeros(2, 4))

    if torch.cuda.is_available():
        with pytest.raises(AssertionError):  # mixed devices
            BatchedCamera(good_v, good_k.cuda(), CameraModelType.PINHOLE, _WIDTH, _HEIGHT)


def test_to_device_moves_all() -> None:
    """.to() moves viewmats, Ks, distortion; None distortion stays None."""
    cam = BatchedCamera.from_single_view(
        _viewmat(torch.device("cpu")),
        _K(torch.device("cpu")),
        CameraModelType.FISHEYE_OPENCV,
        _WIDTH,
        _HEIGHT,
        torch.zeros(4),
    )
    moved = cam.to("cpu")
    assert moved.viewmats.device.type == "cpu"
    assert moved.Ks.device.type == "cpu"
    assert moved.distortion.device.type == "cpu"

    cam_none = BatchedCamera.from_single_view(
        _viewmat(torch.device("cpu")), _K(torch.device("cpu")), CameraModelType.PINHOLE, _WIDTH, _HEIGHT
    )
    assert cam_none.to("cpu").distortion is None

    if torch.cuda.is_available():
        cuda_cam = cam.to("cuda")
        assert cuda_cam.viewmats.is_cuda and cuda_cam.Ks.is_cuda and cuda_cam.distortion.is_cuda
