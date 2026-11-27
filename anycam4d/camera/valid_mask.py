"""Bitpacked observation masks for the world rasterizer.

The host mask from :mod:`anycam4d.camera.validity` gates compositing, so the rendered footprint and the
loss mask are the same object. Layout ``[C, H, (W + 31) // 32]`` int32: bit ``px & 31`` of word
``px >> 5`` is pixel ``px``. ``None`` means every camera observed every pixel (dense path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Sequence

    from jaxtyping import Bool
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset


_BITS_PER_WORD = 32


def pack_validity(mask: Bool[Tensor, "H W"]) -> Int[Tensor, "H W_words"]:
    """Pack a bool ``[H, W]`` mask into row-aligned int32 words; padding bits are 0."""
    if mask.ndim != 2:
        raise ValueError(f"validity mask must be [H, W], got shape {tuple(mask.shape)}")
    height, width = mask.shape
    n_words = (width + _BITS_PER_WORD - 1) // _BITS_PER_WORD

    padded = torch.zeros((height, n_words * _BITS_PER_WORD), dtype=torch.bool, device=mask.device)
    padded[:, :width] = mask
    bits = padded.view(height, n_words, _BITS_PER_WORD).to(torch.int64)
    shifts = torch.arange(_BITS_PER_WORD, device=mask.device, dtype=torch.int64)
    weights = torch.ones_like(shifts) << shifts
    words = (bits * weights).sum(dim=-1)
    # int32 is two's complement, so the top bit of a full word makes the value negative.
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32)


class ValidMaskCache:
    """Per-camera packed masks, built once on device; only partially observed cameras hold one.

    Cameras may differ in resolution (a mixed-lens rig), so the shape is checked per camera.
    """

    def __init__(self, dataset: MultiViewSequenceDataset, device: torch.device) -> None:
        """Pack the mask of every partially observed camera onto ``device``."""
        self._device = device
        self._packed: dict[int, Int[Tensor, "H W_words"]] = {}
        self._shapes: dict[int, tuple[int, int]] = {}
        self._batches: dict[tuple[int, ...], Int[Tensor, "C H W_words"]] = {}
        for camera_id in sorted(set(dataset.camera_ids)):
            mask = dataset.validity_for(camera_id)
            if mask is None or bool(mask.all()):
                continue
            self._shapes[camera_id] = (int(mask.shape[0]), int(mask.shape[1]))
            self._packed[camera_id] = pack_validity(mask.to(device))

    def for_batch(
        self,
        camera_ids: Sequence[int],
        width: int,
        height: int,
    ) -> Int[Tensor, "C H W_words"] | None:
        """Packed masks for ``camera_ids`` in batch order, or ``None`` when every camera in it is fully observed.

        A fully observed camera batched with a masked one gets an all-true mask.

        Raises:
            ValueError: If a camera's mask was built at a different resolution than the render.
        """
        key = tuple(int(c) for c in camera_ids)
        if not any(c in self._packed for c in key):
            return None
        for c in key:
            if c in self._shapes and self._shapes[c] != (height, width):
                raise ValueError(
                    f"validity mask resolution {self._shapes[c]} of camera {c} does not match the render "
                    f"resolution {(height, width)}"
                )
        cached = self._batches.get(key)
        if cached is None:
            full = pack_validity(torch.ones((height, width), dtype=torch.bool, device=self._device))
            cached = torch.stack([self._packed.get(c, full) for c in key], dim=0).contiguous()
            self._batches[key] = cached
        return cached
