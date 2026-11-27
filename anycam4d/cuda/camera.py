"""The one Python-to-kernel camera conversion.

``to_kernel_args`` makes the camera tensors contiguous (the C++ wrappers only check), fills a fixed-width zero
distortion buffer when the camera has none, and derives the ``ProjectionFamily`` dispatch id from the registry. A
missing ``valid_mask`` stays ``None`` and reaches the kernel as a null pointer, which selects the dense path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from anycam4d.camera.registry import distortion_width
from anycam4d.camera.registry import family_cuda_id


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.camera.batched import BatchedCamera


@dataclass(frozen=True)
class KernelCameraArgs:
    """Flat, contiguous camera inputs ready for the CUDA bindings.

    Attributes:
        viewmats: World-to-camera transforms [C, 4, 4] (contiguous).
        Ks: Camera intrinsics [C, 3, 3] (contiguous).
        distortion: Fixed-width per-camera distortion [C, P] (contiguous; zeros if the
            carrier had None).
        family: Projection-family dispatch int (PINHOLE_FAMILY=0 / FISHEYE_FAMILY=1);
            selects the ``kernel<F>`` template instantiation host-side.
        dist_stride: Family slot count P (4 fisheye / 12 pinhole-family).
        width: Image width in pixels.
        height: Image height in pixels.
        valid_mask: Bitpacked observation mask [C, H, (W+31)//32] int32 (contiguous), or None
            when every pixel was observed — passed through unmaterialised, so the kernel sees a
            null pointer and takes its dense path.
    """

    viewmats: Float[Tensor, "C 4 4"]
    Ks: Float[Tensor, "C 3 3"]
    distortion: Float[Tensor, "C P"]
    family: int
    dist_stride: int
    width: int
    height: int
    valid_mask: Int[Tensor, "C H W_words"] | None = None


def to_kernel_args(cam: BatchedCamera) -> KernelCameraArgs:
    """Convert a ``BatchedCamera`` to flat, contiguous kernel inputs.

    The single boundary conversion: enforces contiguity, materialises a zero distortion
    buffer when the carrier had None, and derives the projection-family dispatch int.

    Args:
        cam: Source camera carrier.

    Returns:
        ``KernelCameraArgs`` with contiguous tensors and the projection-family int.
    """
    n_slots = distortion_width(cam.model)
    dist = cam.distortion
    if dist is None:
        dist = torch.zeros((cam.num_cameras, n_slots), device=cam.device, dtype=torch.float32)
    return KernelCameraArgs(
        viewmats=cam.viewmats.contiguous(),
        Ks=cam.Ks.contiguous(),
        distortion=dist.contiguous(),
        family=family_cuda_id(cam.model),
        dist_stride=n_slots,
        width=cam.width,
        height=cam.height,
        valid_mask=None if cam.valid_mask is None else cam.valid_mask.contiguous(),
    )
