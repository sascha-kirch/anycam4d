"""Factory for building datasets and dataloaders from config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import TemporalTiledSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import SamplerType


# Image decoding runs in worker processes; two batches each in flight keep the GPU fed.
_NUM_WORKERS = 4
_PREFETCH_FACTOR = 2


if TYPE_CHECKING:
    from anycam4d.config.dataset import DatasetConfig
    from anycam4d.config.training import TrainingConfig


class DatasetFactory:
    """Creates datasets and dataloaders from config."""

    @staticmethod
    def create_datasets(cfg: DatasetConfig) -> tuple[MultiViewSequenceDataset, MultiViewSequenceDataset | None]:
        """Create the train and (optional) eval MultiViewSequenceDatasets.

        The eval split is optional: when ``eval_dataset.json`` is absent the eval
        dataset is ``None`` and in-loop evaluation is skipped. The existence check
        lives here so the dataset class keeps its strict "split must exist" contract.

        Args:
            cfg: Dataset configuration.

        Returns:
            Tuple of (train_dataset, eval_dataset or None).
        """
        train_ds = MultiViewSequenceDataset(cfg.data_dir, split="train", scale_factor=cfg.scale_factor)

        eval_json = Path(cfg.data_dir) / "eval_dataset.json"
        eval_ds = None
        if eval_json.exists():
            eval_ds = MultiViewSequenceDataset(cfg.data_dir, split="eval", scale_factor=cfg.scale_factor)

        return train_ds, eval_ds

    @staticmethod
    def create_dataloader(
        dataset: MultiViewSequenceDataset,
        training_cfg: TrainingConfig,
    ) -> DataLoader:
        """Build the training DataLoader with the configured sampler.

        Args:
            dataset: Training dataset.
            training_cfg: Training config with sampler type.

        Returns:
            Configured DataLoader.
        """
        common = {
            "num_workers": _NUM_WORKERS,
            "persistent_workers": True,
            "pin_memory": True,
            "prefetch_factor": _PREFETCH_FACTOR,
            "collate_fn": anycam4d_collate,
        }
        sampler_type = training_cfg.sampler

        if sampler_type == SamplerType.FRAME_ALIGNED:
            # C>1 path: each step renders a whole frame's cameras in one pass, so they must share a timestamp.
            # FrameAlignedSampler is a batch_sampler — it yields one frame's full index list,
            # so the leading batch dim C is the camera count of that frame.
            if not dataset.frame_synchronous:
                raise ValueError(
                    "sampler=frame_aligned renders a frame's cameras in one pass at one timestamp, but this "
                    "dataset's cameras are offset in time within a frame; use sampler=uniform or temporal_tiled."
                )
            sampler = FrameAlignedSampler(dataset, shuffle=True)
            return DataLoader(dataset, batch_sampler=sampler, **common)

        # UNIFORM and TEMPORAL_TILED give each render its own timestamp, so they cannot share the single
        # render timestamp the C>1 path needs — they are C=1-only.
        if training_cfg.batch_size != 1:
            raise ValueError(
                f"Sampler {sampler_type.value} mixes timestamps across a batch and is C=1-only; "
                f"set training.batch_size=1 (got {training_cfg.batch_size}). Use sampler=frame_aligned for C>1."
            )

        if sampler_type == SamplerType.TEMPORAL_TILED:
            # One tile of consecutive frames per optimizer step: the tile length is the step's render count.
            sampler = TemporalTiledSampler(dataset, tile_len=training_cfg.accum_renders, shuffle=True)
            return DataLoader(dataset, sampler=sampler, batch_size=1, **common)

        # SamplerType.UNIFORM — default shuffle
        return DataLoader(dataset, shuffle=True, batch_size=1, **common)
