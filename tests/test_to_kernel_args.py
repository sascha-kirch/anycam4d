"""``to_kernel_args``: contiguous camera tensors and zero-materialized distortion for the kernels."""

from __future__ import annotations

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.registry import distortion_width
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import CameraModelType


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="BatchedCamera path exercised against the CUDA build")


def test_to_kernel_args_contiguity() -> None:
    """Non-contiguous viewmats/Ks in the carrier → contiguous tensors out."""
    from anycam4d.cuda.camera import to_kernel_args

    viewmats = torch.randn(2, 4, 4).transpose(1, 2)  # non-contiguous view
    Ks = torch.randn(2, 3, 3).transpose(1, 2)  # non-contiguous view
    assert not viewmats.is_contiguous() and not Ks.is_contiguous()

    cam = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, 16, 16)
    args = to_kernel_args(cam)

    assert args.viewmats.is_contiguous()
    assert args.Ks.is_contiguous()
    assert args.distortion.is_contiguous()
    assert torch.equal(args.viewmats, viewmats)
    assert torch.equal(args.Ks, Ks)


def test_to_kernel_args_materialises_zeros() -> None:
    """distortion=None → [C, n_slots] zeros; correct dist_stride and family dispatch int."""
    from anycam4d.camera.registry import family_cuda_id
    from anycam4d.cuda.camera import to_kernel_args

    cam = BatchedCamera(
        torch.eye(4).expand(3, 4, 4), torch.eye(3).expand(3, 3, 3), CameraModelType.FISHEYE_OPENCV, 16, 16
    )
    args = to_kernel_args(cam)

    n_slots = distortion_width(CameraModelType.FISHEYE_OPENCV)
    assert args.dist_stride == n_slots
    assert args.distortion.shape == (3, n_slots)
    assert torch.count_nonzero(args.distortion) == 0
    assert args.family == family_cuda_id(CameraModelType.FISHEYE_OPENCV) == 1
    assert args.width == 16 and args.height == 16
