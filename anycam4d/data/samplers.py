"""Samplers and collate function for FourDGS temporal training."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING
from typing import Iterator

import torch
from torch.utils.data import Sampler

from anycam4d.enums import CameraModelType  # noqa: TC001 — used at runtime in anycam4d_collate


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
    from anycam4d.data.types import TrainingSample


class FrameAlignedSampler(Sampler[list[int]]):
    """Yields one frame's camera indices per batch, for ``DataLoader(batch_sampler=sampler)``.

    Args:
        dataset: MultiViewSequenceDataset to sample from.
        shuffle: Shuffle frame order each epoch (default True).
    """

    def __init__(self, dataset: MultiViewSequenceDataset, shuffle: bool = True) -> None:
        """Initialise sampler from dataset frame groups."""
        self._groups: list[list[int]] = [dataset.indices_at_frame(i) for i in range(dataset.num_timesteps)]
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self._groups)))

        if self._shuffle:
            random.shuffle(order)

        for i in order:
            yield self._groups[i]

    def __len__(self) -> int:
        return len(self._groups)


class TemporalTiledSampler(Sampler[int]):
    """Yields one camera's consecutive frames, one fixed tile of ``tile_len`` frames after another.

    The sequence is cut into tiles ``{0..N-1}, {N..2N-1}, ...``; each epoch visits every (camera, tile) once,
    in shuffled order. With ``batch_size=1`` and ``accum_renders = tile_len`` an optimizer step renders
    exactly one tile — which holds across epochs only because every tile is full, hence the divisibility check.

    Args:
        dataset: MultiViewSequenceDataset to sample from.
        tile_len: Frames per tile; must divide the frame count.
        shuffle: Shuffle tile order each epoch (default True).

    Raises:
        ValueError: When ``tile_len`` does not divide the number of frames.
    """

    def __init__(self, dataset: MultiViewSequenceDataset, tile_len: int, shuffle: bool = True) -> None:
        """Build the (camera, tile) index lists from the dataset's frame groups."""
        n_frames = dataset.num_timesteps
        if tile_len < 1 or n_frames % tile_len != 0:
            raise ValueError(
                f"temporal_tiled needs tiles that divide the sequence: {n_frames} frames do not split into "
                f"tiles of {tile_len} (training.accum_renders). A short tile would shift every later step."
            )

        index_of: dict[tuple[int, int], int] = {}
        for frame in range(n_frames):
            for idx in dataset.indices_at_frame(frame):
                index_of[(dataset.camera_ids[idx], frame)] = idx

        self._tiles = [
            [index_of[(cam, frame)] for frame in range(start, start + tile_len)]
            for cam in sorted(set(dataset.camera_ids))
            for start in range(0, n_frames, tile_len)
        ]
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self._tiles)))

        if self._shuffle:
            random.shuffle(order)

        for i in order:
            yield from self._tiles[i]

    def __len__(self) -> int:
        return sum(len(tile) for tile in self._tiles)


def anycam4d_collate(samples: list[TrainingSample]) -> dict[str, Tensor | list[int] | list[float] | None]:
    """Collate a list of TrainingSample into a batched dict.

    Stacks images, K, viewmat tensors. Collects scalar fields into lists.
    Validates that all samples share the same camera model (mixed batches are not supported).

    Args:
        samples: List of TrainingSample from __getitem__.

    Returns:
        Dict with keys: images, Ks, viewmats, timestamps, image_ids, camera_ids,
        camera_model, distortion_params, masks, validity.

        ``validity`` is ``[B, H, W, 1]`` bool whenever any camera in the dataset has unobserved
        pixels, and ``None`` when every camera observed its full frame. The dataset decides that
        for all its samples at once, so the all-or-nothing stacking below can never drop one
        camera's real mask because another camera's was trivial.

    Raises:
        ValueError: If samples contain mixed camera models.
    """
    models = {s.camera_model for s in samples}
    if len(models) != 1:
        raise ValueError(f"anycam4d_collate: batch has mixed camera models: {models}")
    camera_model: CameraModelType = next(iter(models))

    # Validate uniform resolution before stacking (torch.stack otherwise throws an opaque
    # shape error). Same fail-fast spirit as the mixed-camera-model guard above.
    hw = {(s.image.shape[0], s.image.shape[1]) for s in samples}
    if len(hw) != 1:
        raise ValueError(f"anycam4d_collate: batch has mixed image resolutions (H, W): {hw}")
    mask_hw = {(s.mask.shape[0], s.mask.shape[1]) for s in samples if s.mask is not None}
    if len(mask_hw) > 1:
        raise ValueError(f"anycam4d_collate: batch has mixed mask resolutions (H, W): {mask_hw}")

    images = torch.stack([s.image for s in samples])  # [B, H, W, 3]
    Ks = torch.stack([s.K for s in samples])  # [B, 3, 3]
    viewmats = torch.stack([s.viewmat for s in samples])  # [B, 4, 4]

    timestamps: list[float] = [s.timestamp for s in samples]
    image_ids: list[int] = [s.image_id for s in samples]
    camera_ids: list[int] = [s.camera_id for s in samples]

    # distortion_params is always present (fixed-width zero-filled per family), so the batch
    # always stacks to [B, P].
    distortion_params: Float[Tensor, "B P"] = torch.stack([s.distortion_params for s in samples])

    masks_list = [s.mask for s in samples]
    if all(m is not None for m in masks_list):
        masks: Float[Tensor, "B H W 1"] | None = torch.stack(masks_list)  # type: ignore[arg-type]
    else:
        masks = None

    validity_list = [s.validity for s in samples]
    if all(v is not None for v in validity_list):
        validity: Bool[Tensor, "B H W 1"] | None = torch.stack(validity_list)  # type: ignore[arg-type]
    else:
        validity = None

    return {
        "images": images,
        "Ks": Ks,
        "viewmats": viewmats,
        "timestamps": timestamps,
        "image_ids": image_ids,
        "camera_ids": camera_ids,
        "camera_model": camera_model,
        "distortion_params": distortion_params,
        "masks": masks,
        "validity": validity,
    }
