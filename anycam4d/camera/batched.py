"""BatchedCamera — struct-of-arrays camera carrier.

The single camera object that trainer / eval / viewer build and hand to the
render pipeline. One camera *model* per batch (warp-uniform); the batch
dimension ``C`` carries per-view extrinsics/intrinsics/distortion.

"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING

from anycam4d.camera.registry import distortion_width


if TYPE_CHECKING:
    import torch
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.enums import CameraModelType


@dataclass(frozen=True)
class BatchedCamera:
    """Struct-of-arrays carrier for a batch of `C` cameras sharing one model, height and width.

    Args:
        viewmats: World-to-camera transforms [C, 4, 4].
        Ks: Camera intrinsic matrices [C, 3, 3].
        model: Camera projection model — one model for the whole batch.
        width: Image width in pixels.
        height: Image height in pixels.
        distortion: Fixed-width per-camera distortion coefficients [C, P] (P = family slot
            count: 12 pinhole-family / 4 fisheye), or None (synthetic/viewer → UT zeros).
        valid_mask: Bitpacked per-camera observation mask [C, H, (W+31)//32] int32, a set bit
            meaning the pixel was observed (see :mod:`anycam4d.camera.valid_mask`), or None when
            every pixel of every camera was observed — the dense render path.
    """

    viewmats: Float[Tensor, "C 4 4"]
    Ks: Float[Tensor, "C 3 3"]
    model: CameraModelType
    width: int
    height: int
    distortion: Float[Tensor, "C P"] | None = None
    valid_mask: Int[Tensor, "C H W_words"] | None = None

    def __post_init__(self) -> None:
        """Cheap, loud, O(1) shape/device checks (no registry/capability checks)."""
        c = self.viewmats.shape[0]

        assert self.viewmats.shape == (c, 4, 4), f"viewmats must be [C,4,4], got {tuple(self.viewmats.shape)}"  # noqa: S101
        assert self.Ks.shape == (c, 3, 3), f"Ks must be [C,3,3], got {tuple(self.Ks.shape)}"  # noqa: S101
        assert self.Ks.device == self.viewmats.device, "viewmats and Ks must share one device"  # noqa: S101

        if self.distortion is not None:
            assert self.distortion.ndim == 2, f"distortion must be [C,P], got rank {self.distortion.ndim}"  # noqa: S101
            assert self.distortion.shape[0] == c, f"distortion dim0 {self.distortion.shape[0]} != C={c}"  # noqa: S101
            assert self.distortion.device == self.viewmats.device, "distortion must share viewmats device"  # noqa: S101
            n_slots = distortion_width(self.model)
            assert (  # noqa: S101
                self.distortion.shape[1] == n_slots
            ), f"distortion width {self.distortion.shape[1]} != family slots {n_slots} for {self.model}"

        if self.valid_mask is not None:
            n_words = (self.width + 31) // 32
            assert self.valid_mask.shape == (  # noqa: S101
                c,
                self.height,
                n_words,
            ), f"valid_mask must be [C,H,(W+31)//32] = {(c, self.height, n_words)}, got {tuple(self.valid_mask.shape)}"
            assert self.valid_mask.device == self.viewmats.device, "valid_mask must share viewmats device"  # noqa: S101

    @property
    def num_cameras(self) -> int:
        """Number of cameras in the batch (``C``)."""
        return self.viewmats.shape[0]

    @property
    def device(self) -> torch.device:
        """Device shared by all carried tensors."""
        return self.viewmats.device

    def select(self, i: int) -> BatchedCamera:
        """Camera ``i`` as a batch of one."""
        return replace(
            self,
            viewmats=self.viewmats[i : i + 1],
            Ks=self.Ks[i : i + 1],
            distortion=None if self.distortion is None else self.distortion[i : i + 1],
            valid_mask=None if self.valid_mask is None else self.valid_mask[i : i + 1],
        )

    def to(self, device: torch.device | str) -> BatchedCamera:
        """Move all carried tensors to ``device``; returns a new frozen carrier.

        Intended for small, off-hot-path moves (eval/viewer setup). The carried tensors are
        KB-scale, so a blocking transfer is fine — the trainer hot path instead moves the
        large image tensors with ``non_blocking=True`` and builds the carrier on-device,
        bypassing this method entirely.
        """
        return BatchedCamera(
            viewmats=self.viewmats.to(device),
            Ks=self.Ks.to(device),
            model=self.model,
            width=self.width,
            height=self.height,
            distortion=None if self.distortion is None else self.distortion.to(device),
            valid_mask=None if self.valid_mask is None else self.valid_mask.to(device),
        )

    @classmethod
    def from_single_view(
        cls,
        viewmat: Float[Tensor, "4 4"],
        K: Float[Tensor, "3 3"],
        model: CameraModelType,
        width: int,
        height: int,
        distortion: Float[Tensor, "D"] | None = None,
        valid_mask: Int[Tensor, "H W_words"] | None = None,
    ) -> BatchedCamera:
        """Build a ``C=1`` carrier from a single view by ``unsqueeze(0)``."""
        return cls(
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            model=model,
            width=width,
            height=height,
            distortion=None if distortion is None else distortion.unsqueeze(0),
            valid_mask=None if valid_mask is None else valid_mask.unsqueeze(0),
        )
