"""Evaluate a checkpoint on the eval split: whole-frame metrics, native-space region split and incidence bands.

The model and pipeline are rebuilt from the checkpoint's config snapshot. ``--data_dir`` scores it on another
variant of its capture and ``--rendering``/``--set`` through another backend; both require the scene normalization
the model was trained in (checked). Region split and bands are also written to the run's TensorBoard.

Usage::

    python scripts/eval.py --checkpoint runs/<run>/checkpoints/best.pt
        [--data_dir DIR] [--output FILE] [--no_mask] [--tb_dir DIR]
        [--rendering NAME] [--set k=v ...] [--band_frames] [--central_ref DIR]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

from anycam4d.config.dataset import DatasetConfig
from anycam4d.config.rendering import RenderingConfig
from anycam4d.config.training import TrainingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.enums import LpipsNet
from anycam4d.enums import PostProcessorType
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.eval.checkpoint import denorm_transform
from anycam4d.eval.checkpoint import scene_normalization
from anycam4d.eval.evaluator import Evaluator
from anycam4d.eval.native_evaluator import NativeRegionEvaluator
from anycam4d.eval.native_evaluator import load_central_fields


if TYPE_CHECKING:
    from jaxtyping import Float
    from numpy import ndarray

    from anycam4d.eval.native_evaluator import NativeRegionReport
    from anycam4d.eval.result import EvalResult
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.pipeline import RenderPipeline


_RENDERING_GROUP_DIR = Path(__file__).parents[1] / "conf" / "rendering"


def resolve_rendering(stored: dict, group: str | None, overrides: list[str]) -> tuple[RenderingConfig, bool]:
    """The stored rendering block or a ``conf/rendering`` preset, with ``key=value`` overrides; returns ``(config, is_override)``."""
    if group is None:
        base = OmegaConf.create(stored)
    else:
        path = Path(group) if group.endswith((".yaml", ".yml")) else _RENDERING_GROUP_DIR / f"{group}.yaml"
        if not path.is_file():
            available = ", ".join(sorted(p.stem for p in _RENDERING_GROUP_DIR.glob("*.yaml")))
            raise SystemExit(f"No rendering config at {path}. Available presets: {available}.")
        base = OmegaConf.load(path)
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides)) if overrides else base
    return RenderingConfig.from_omegaconf(merged), bool(group) or bool(overrides)


def _existing_tb_dir(checkpoint: Path) -> Path | None:
    """The run's TensorBoard directory (``<run>/tb``) if it exists, else ``None``.

    Checkpoints live at ``<run>/checkpoints/best.pt``, so the run root is two levels up.
    """
    tb = checkpoint.parent.parent / "tb"
    return tb if tb.is_dir() else None


def write_frame_curves(writer: SummaryWriter, result: EvalResult, prefix: str) -> None:
    """Write per-frame metric curves to TensorBoard stepped by frame index, with timestamp and frame id alongside."""
    for metric, values in result.per_timestamp.items():
        for frame, value in enumerate(values):
            writer.add_scalar(f"{prefix}/{metric}", value, frame)
    for frame, tau in enumerate(result.timestamps):
        writer.add_scalar(f"{prefix}/timestamp", tau, frame)
    for frame, frame_id in enumerate(result.frame_ids):
        writer.add_scalar(f"{prefix}/frame_id", frame_id, frame)


def write_band_curves(writer: SummaryWriter, report: NativeRegionReport, prefix: str) -> None:
    """Write the region split and non-empty incidence bands to TensorBoard, stepped by band index."""
    table = report.bands.table()
    for index, (label, values) in enumerate(table.items()):
        if values["px"] == 0:
            continue
        writer.add_scalar(f"{prefix}/band_psnr", values["psnr"], index)
        writer.add_scalar(f"{prefix}/band_ssim", values["ssim"], index)
        writer.add_scalar(f"{prefix}/band_px", values["px"], index)
        writer.add_scalar(f"{prefix}/band_lo_deg", float(label.split("-")[0]), index)

    lines = ["### Native-space region split", "", "| metric | mean | std |", "|---|---|---|"]
    lines += [f"| {key} | {value:.4f} | {report.result.std[key]:.4f} |" for key, value in report.result.mean.items()]
    lines += ["", "### Incidence bands", "", "| band (deg) | PSNR | SSIM | px |", "|---|---|---|---|"]
    lines += [f"| {label} | {v['psnr']:.3f} | {v['ssim']:.4f} | {v['px']} |" for label, v in table.items() if v["px"]]
    lines += ["", f"_{report.bands.to_dict()['aggregation']}_"]
    writer.add_text(f"{prefix}/summary", "\n".join(lines), global_step=0)


@dataclass
class EvalConfig:
    """Resolved evaluation settings: CLI overrides merged onto the checkpoint config snapshot."""

    checkpoint: Path
    output: Path
    data_dir: Path
    scale_factor: int
    lpips_net: LpipsNet
    compute_cc: bool
    apply_mask: bool
    device: torch.device
    tb_dir: Path | None
    rendering: RenderingConfig
    rendering_is_override: bool
    band_frames: bool
    central_ref: Path | None

    @staticmethod
    def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        """Parse the (minimal) CLI. Only ``--checkpoint`` is required."""
        # fmt: off
        parser = argparse.ArgumentParser(description="Evaluate a 4DGS checkpoint on the eval split.")
        parser.add_argument("-c", "--checkpoint", type=Path, required=True, help="Checkpoint written by training (best.pt or last.pt).")
        parser.add_argument("-d", "--data_dir", type=Path, default=None, help="Dataset root (overrides checkpoint config).")
        parser.add_argument("-o", "--output", type=Path, default=None, help="JSON output path (default: <ckpt_dir>/../eval/metrics.json).")
        parser.add_argument("--no_mask", action="store_true", help="Evaluate full RGB instead of masked foreground.")
        parser.add_argument("--tb_dir", type=Path, default=None, help="TensorBoard dir for the per-frame curves (default: the run's own tb/, when it exists).")
        parser.add_argument("--rendering", type=str, default=None, help="Render through a different backend: a conf/rendering preset name (anycam4d, ewa) or a path to a YAML. Needs a --data_dir in the matching camera parameterization.")
        parser.add_argument("--set", dest="rendering_set", action="append", default=[], metavar="KEY=VALUE", help="Dotlist override on the rendering config (repeatable), e.g. --set packed=false.")
        parser.add_argument("--lpips_net", type=LpipsNet, default=None, help="LPIPS backbone (alex, vgg, squeeze); default: the checkpoint's training.lpips_net.")
        parser.add_argument("--band_frames", action="store_true", help="Also emit the incidence-band table separately for every frame (bands x frames; off by default because the per-frame central/peripheral curves already answer whether the periphery degrades faster over time).")
        parser.add_argument("--central_ref", type=Path, default=None, help="Undistorted dataset whose held-out pinholes define the central region (the shared target). Without it no central/peripheral split is scored.")
        # fmt: on
        return parser.parse_args(argv)

    @classmethod
    def from_args(cls, checkpoint: dict, args: argparse.Namespace) -> EvalConfig:
        """Merge CLI args with the checkpoint's config snapshot; exits without one."""
        cfg = checkpoint.get("config")
        if cfg is None:
            raise SystemExit("Checkpoint lacks a config snapshot (re-train with config persistence).")

        dataset_cfg = DatasetConfig.from_omegaconf(OmegaConf.create(cfg["dataset"]))
        training_cfg = TrainingConfig.from_omegaconf(OmegaConf.create(cfg["training"]))
        stored_rendering = RenderingConfig.from_omegaconf(OmegaConf.create(cfg["rendering"]))
        rendering, is_override = resolve_rendering(cfg["rendering"], args.rendering, args.rendering_set)

        output = args.output or args.checkpoint.parent.parent / "eval" / "metrics.json"
        return cls(
            checkpoint=args.checkpoint,
            output=output,
            data_dir=args.data_dir or dataset_cfg.data_dir,
            scale_factor=dataset_cfg.scale_factor,
            lpips_net=args.lpips_net or training_cfg.lpips_net,
            # cc_* metrics when the run trained a bilateral grid; a --rendering override does not change that.
            compute_cc=stored_rendering.post_processor is PostProcessorType.BILATERAL_GRID,
            apply_mask=not args.no_mask,
            device=torch.device("cuda"),
            # Per-frame curves go into the run's own event file, next to its training scalars; a
            # checkpoint without a tb/ directory gets none rather than a new, phantom run directory.
            tb_dir=args.tb_dir or _existing_tb_dir(args.checkpoint),
            rendering=rendering,
            rendering_is_override=is_override,
            band_frames=args.band_frames,
            central_ref=args.central_ref,
        )

    def print_config(self) -> None:
        """Pretty-print the resolved configuration as a rich table."""
        table = Table(title="AnyCam4D Evaluation", style="bold blue", title_style="bold blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_row("Checkpoint", str(self.checkpoint))
        table.add_row("Output", str(self.output))
        table.add_row("Data Dir", str(self.data_dir))
        table.add_row("Scale Factor", str(self.scale_factor))
        table.add_row("LPIPS Net", self.lpips_net.value)
        table.add_row("Color Correct", str(self.compute_cc))
        table.add_row("Apply Mask", str(self.apply_mask))
        backend = f"{self.rendering.backend.value} (packed={self.rendering.packed})"
        table.add_row("Backend", backend + ("  [CROSS-RENDER]" if self.rendering_is_override else ""))
        table.add_row("Per-frame Bands", str(self.band_frames))
        table.add_row("Central Reference", str(self.central_ref) if self.central_ref else "-")
        table.add_row("TensorBoard", str(self.tb_dir) if self.tb_dir else "-")
        table.add_row("Device", str(self.device))
        Console().print(table)


def run_eval(argv: list[str] | None = None) -> EvalResult:
    """Load a checkpoint, evaluate the eval split, print + persist metrics, return the result."""
    args = EvalConfig.parse_args(argv)
    # Fail early without a GPU: loading onto CUDA would otherwise fail deep in unpickling, like a corrupt file.
    if not torch.cuda.is_available():
        raise SystemExit("eval needs a GPU (the render backends are CUDA-only) and torch.cuda.is_available() is False.")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    config = EvalConfig.from_args(checkpoint, args)
    config.print_config()

    if config.rendering_is_override or args.data_dir is not None:
        _check_shared_normalization(config, checkpoint)

    dataset = MultiViewSequenceDataset(config.data_dir, split="eval", scale_factor=config.scale_factor)
    t_denorm = denorm_transform(checkpoint)
    if t_denorm is not None:
        dataset.apply_viewmat_transform(t_denorm)

    override = config.rendering if config.rendering_is_override else None
    model, pipeline = build_pipeline_from_checkpoint(
        checkpoint, config.device, rendering_override=override, source=str(config.checkpoint)
    )
    model.eval()

    evaluator = Evaluator(config.device, lpips_net=config.lpips_net, compute_cc=config.compute_cc)
    result = evaluator.evaluate(model, pipeline, dataset, apply_mask=config.apply_mask)

    print(result.format())
    payload = {
        "checkpoint": str(config.checkpoint),
        "step": checkpoint.get("step"),
        "data_dir": str(config.data_dir),
        "lpips_net": config.lpips_net.value,
        "backend": config.rendering.backend.value,
        "cross_render": config.rendering_is_override,
        "central_ref": str(config.central_ref) if config.central_ref else None,
        **result.to_dict(),
    }

    native_report = _native_region_eval(config, model, pipeline, dataset)
    native_result = native_report.result
    payload["native_region"] = native_result.to_dict()
    payload["native_bands"] = native_report.to_dict()

    if config.tb_dir is not None:
        with SummaryWriter(log_dir=str(config.tb_dir)) as writer:
            write_frame_curves(writer, result, "eval_frame")
            write_frame_curves(writer, native_result, "eval_frame_native")
            write_band_curves(writer, native_report, "eval_native")
        print(f"Wrote per-frame curves and the region split to {config.tb_dir}")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote metrics to {config.output}")
    return result


def _native_region_eval(
    config: EvalConfig,
    model: FourDGaussians,
    pipeline: RenderPipeline,
    dataset: MultiViewSequenceDataset,
) -> NativeRegionReport:
    """Run and print the native-space region split, bands and per-frame curve."""
    fields = load_central_fields(config.central_ref) if config.central_ref else None
    report = NativeRegionEvaluator(
        config.device, lpips_net=config.lpips_net, per_frame_bands=config.band_frames, central_fields=fields
    ).evaluate(model, pipeline, dataset, apply_mask=config.apply_mask)
    print("\nNative-space region split:")
    print(report.result.format())
    print("\nNative-space incidence bands:")
    print(report.bands.format())
    return report


def _same_normalization(
    a: tuple[float, Float[ndarray, "3"]] | None,
    b: tuple[float, Float[ndarray, "3"]] | None,
) -> bool:
    """Whether two checkpoints' ``(scale, centre)`` normalizations agree to float tolerance."""
    if a is None or b is None:
        return a is b
    return bool(np.isclose(a[0], b[0]) and np.allclose(a[1], b[1]))


def _check_shared_normalization(config: EvalConfig, checkpoint: dict) -> None:
    """Exit unless the eval dataset shares the checkpoint's scene normalization."""
    stored = scene_normalization(checkpoint)
    if stored is None:
        raise SystemExit(
            f"{config.checkpoint} has no norm_scale/norm_center, so it cannot be cross-rendered "
            "against another dataset: there is no way to place it in that dataset's space."
        )
    train = MultiViewSequenceDataset(config.data_dir, split="train", scale_factor=config.scale_factor)
    norm = compute_normalization(train)
    if not _same_normalization((norm.norm_scale, norm.norm_center.cpu().numpy()), stored):
        raise SystemExit(
            f"{config.data_dir} normalizes to scale={norm.norm_scale:.8f} centre={norm.norm_center.tolist()}, "
            f"but {config.checkpoint} was trained at scale={stored[0]:.8f} centre={stored[1].tolist()}. "
            "These are different captures, not two parameterizations of one — every metric would be "
            "computed from a model rendered with the wrong cameras."
        )


def main() -> None:
    """CLI entry point."""
    run_eval()


if __name__ == "__main__":
    main()
