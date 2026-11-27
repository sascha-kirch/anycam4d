"""Training entry point — Hydra-managed config."""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from anycam4d.config.experiment import ExperimentConfig
from anycam4d.data.factory import DatasetFactory
from anycam4d.data.normalization import compute_normalization
from anycam4d.enums import SamplerType
from anycam4d.primitives.factory import ModelFactory
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.factory import PostProcessorFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.strategy.strategy_factory import StrategyFactory
from anycam4d.trainer.trainer import Trainer
from anycam4d.utils import seed_everything
from anycam4d.viewer.factory import ViewerFactory


def _resolve_temporal_scales(exp_cfg: ExperimentConfig, n_frames: int) -> None:
    """Resolve ``-1.0`` temporal sentinels as multiples of ``1/n_frames``: ``temporal_spread_init = 5/n_frames``, ``sigma_t_floor = 1/n_frames``.

    The init spread is clamped up to the floor if an override would start below it.
    """
    frame_spacing = 1.0 / max(n_frames, 1)
    resolved = []

    if exp_cfg.model.temporal_spread_init < 0:
        exp_cfg.model.temporal_spread_init = 5.0 * frame_spacing
        resolved.append(f"temporal_spread_init = 5/{n_frames} = {exp_cfg.model.temporal_spread_init:.4f}")

    if exp_cfg.strategy.sigma_t_floor < 0:
        exp_cfg.strategy.sigma_t_floor = 1.0 * frame_spacing
        resolved.append(f"sigma_t_floor = 1/{n_frames} = {exp_cfg.strategy.sigma_t_floor:.4f}")

    # Init consistency: a user override forcing init scale_t below the floor would seed primitives
    # beneath the very bound the floor is meant to hold — clamp init up to the floor and warn.
    floor = exp_cfg.strategy.sigma_t_floor
    if floor > 0 and exp_cfg.model.temporal_spread_init < floor:
        print(
            f"  [warn] temporal_spread_init ({exp_cfg.model.temporal_spread_init:.4f}) is below "
            f"sigma_t_floor ({floor:.4f}); clamping init up to the floor."
        )
        exp_cfg.model.temporal_spread_init = floor

    if resolved:
        print(f"  [auto temporal scale] n_frames={n_frames}: " + ", ".join(resolved))


def _resolve_batch_size(exp_cfg: ExperimentConfig, cameras_per_frame: int) -> None:
    """Resolve ``training.batch_size = -1`` (camera count under ``frame_aligned``, else 1).

    Neither the batch size nor ``accum_renders`` scales learning rates; together they only report the effective batch.
    """
    t = exp_cfg.training
    frame_aligned = t.sampler is SamplerType.FRAME_ALIGNED

    if t.batch_size < 0:
        t.batch_size = cameras_per_frame if frame_aligned else 1
        print(f"  batch_size={t.batch_size} (auto: {'cameras per frame' if frame_aligned else 'C=1 sampler'})")

    if t.accum_renders < 1:
        raise ValueError(f"training.accum_renders must be >= 1, got {t.accum_renders}")

    print(f"  effective batch = {t.batch_size} x {t.accum_renders} = {t.batch_size * t.accum_renders} views/step")


@hydra.main(version_base=None, config_path="../conf", config_name="4dgs")
def main(cfg: ExperimentConfig) -> None:  # type: ignore[type-arg]
    """Build all components, then train."""
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    exp_cfg = ExperimentConfig.from_omegaconf(cfg)
    # Seeded here rather than under __main__: the seed is a config field, so it cannot be applied
    # before Hydra has composed the config. This is also upstream of every RNG consumer -- dataset
    # construction, parameter init, the sampler and the DataLoader workers are all built below.
    seed_everything(exp_cfg.training.seed)
    print(f"  seed={exp_cfg.training.seed}")
    # Resolved plain dict — persisted into each checkpoint so the standalone eval
    # can rebuild the exact render pipeline (backend, projection, post-processor).
    config_snapshot = OmegaConf.to_container(cfg, resolve=True)

    train_ds, eval_ds = DatasetFactory.create_datasets(exp_cfg.dataset)
    eval_str = f", eval samples: {len(eval_ds)}" if eval_ds is not None else ", eval: disabled"
    print(f"Train samples: {len(train_ds)}{eval_str}")

    _resolve_temporal_scales(exp_cfg, train_ds.num_timesteps)
    _resolve_batch_size(exp_cfg, train_ds.cameras_per_timestep)

    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)
    if eval_ds is not None:
        eval_ds.apply_viewmat_transform(norm.T_denorm)
    print(f"  scene_scale={norm.scene_scale:.4f} (camera-based, normalized space)")

    model = ModelFactory.create_model(train_ds.point_cloud_path, norm, exp_cfg.model, device)

    backend = BackendFactory.create(exp_cfg.rendering)

    post_processor = PostProcessorFactory.create(exp_cfg.rendering, n_cameras=train_ds.n_cameras, device=device)

    pipeline = RenderPipeline(backend=backend, post_processor=post_processor)

    strategy = StrategyFactory.create(exp_cfg.strategy, exp_cfg.rendering.backend, exp_cfg.training.max_steps)

    train_dataloader = DatasetFactory.create_dataloader(train_ds, exp_cfg.training)

    viewer = ViewerFactory.create(
        exp_cfg.viewer,
        model,
        pipeline,
        num_timesteps=train_ds.num_timesteps,
        T_denorm=norm.T_denorm,
    )

    trainer = Trainer(
        model,
        train_ds,
        eval_ds,
        train_dataloader,
        strategy,
        norm,
        training_cfg=exp_cfg.training,
        pipeline=pipeline,
        result_dir=result_dir,
        viewer=viewer,
        config_snapshot=config_snapshot,
    )
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
