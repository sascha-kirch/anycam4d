"""Build each registry scene's base dataset, then its variants; resumable.

Stages: ``base`` (ffmpeg extract + COLMAP/MVS + writer, full resolution) and one ``variant`` per
``(variant, scale)``, named ``{scene}_{variant}[_r{scale}]``. Native at full resolution and undistorted
on a pinhole base are skipped as duplicates. A stage is skipped when its marker is set and its input
hash is unchanged. The raw source is fetched first if absent.

Flags: ``--force`` redo all; ``--from <stage>`` redo it and later ones; ``--only <stage>`` run just it
(add ``--force`` to redo). ``<stage>`` is ``base``, ``variants`` or a variant name.

Usage::

    python scripts/prepare_datasets/build_variants.py --registry conf/datasets/dynerf.yaml \
        --source-root ~/datasets/DyNeRF --out-root data/4dgs [--scene NAME] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pycolmap

from anycam4d.console import notice
from anycam4d.console import rule
from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.provenance import build_manifest
from anycam4d.data.prep.registry import build_importer
from anycam4d.data.prep.registry import check_scene
from anycam4d.data.prep.registry import load_registry
from anycam4d.data.prep.registry import manifest_source
from anycam4d.data.prep.registry import variant_spec
from anycam4d.data.prep.staging import input_hash
from anycam4d.data.prep.staging import mark_done
from anycam4d.data.prep.staging import stage_done
from anycam4d.data.prep.variants import build_variant
from anycam4d.enums import CameraModelType


if TYPE_CHECKING:
    from anycam4d.data.prep.registry import DatasetConfig


# Load the sibling ``download.py`` (scripts/ is not an importable package) for its fetch seam.
_DOWNLOAD_PATH = Path(__file__).resolve().with_name("download.py")
_dl_spec = importlib.util.spec_from_file_location("prep_download", _DOWNLOAD_PATH)
if _dl_spec is None or _dl_spec.loader is None:
    raise ImportError(f"could not load sibling download.py at {_DOWNLOAD_PATH}")
download = importlib.util.module_from_spec(_dl_spec)
sys.modules[_dl_spec.name] = download
_dl_spec.loader.exec_module(download)


# Stage ordering for ``--from``; every variant shares the last slot.
_GROUP_INDEX = {"base": 0, "variants": 1}


@dataclass
class ScenePaths:
    """Resolved on-disk locations for one scene's stages."""

    source_root: Path
    source_dir: Path
    out_root: Path
    base_dir: Path
    work_dir: Path
    state_path: Path


@dataclass
class VariantJob:
    """A single ``(variant, scale)`` build queued for the variants stage."""

    base_dir: Path
    out_dir: Path
    name: str
    scale: float
    state_path: Path
    stage_key: str
    input_hash: str
    tool_versions: dict
    forced: bool


def _scene_paths(cfg: DatasetConfig, scene: str, args: argparse.Namespace) -> ScenePaths:
    """Resolve every path a scene's stages read/write from the CLI args + registry."""
    out_root = args.out_root
    work_dir = args.work_dir if args.work_dir is not None else out_root / f"{scene}_work"
    return ScenePaths(
        source_root=args.source_root,
        source_dir=args.source_root / cfg.scenes[scene].source_subdir,
        out_root=out_root,
        base_dir=out_root / scene,
        work_dir=work_dir,
        state_path=out_root / f"{scene}.prep_state.json",
    )


def _scale_suffix(scale: float) -> str:
    """``"_r2"`` for a 2x downscale, ``""`` for full resolution (``scale == 1``)."""
    return "" if scale == 1.0 else f"_r{scale:g}"


def _variant_label(name: str, scale: float) -> str:
    """Stable ``(variant, scale)`` identifier for the resume marker (e.g. ``undistorted``, ``undistorted_r2``)."""
    return f"{name}{_scale_suffix(scale)}"


def _variant_dirname(scene: str, name: str, scale: float) -> str:
    """Output directory ``{scene}_{variant}[_r{scale}]``; ``native`` keeps the bare stem (``{scene}_r2``)."""
    stem = scene if name == "native" else f"{scene}_{name}"
    return f"{stem}{_scale_suffix(scale)}"


def _base_is_pinhole(base_dir: Path) -> bool:
    """Whether every base camera is an undistorted pinhole; ``False`` when the base JSON is absent."""
    train = base_dir / "train_dataset.json"
    if not train.exists():
        return False
    cameras = json.loads(train.read_text()).get("cameras", [])
    return bool(cameras) and all(cam.get("model") == CameraModelType.PINHOLE.value for cam in cameras)


def _matches(group: str, stage_key: str, sel: str) -> bool:
    """Whether a stage matches an ``--only`` selector; a bare variant name matches every scale."""
    return sel == group or stage_key == f"variant:{sel}" or stage_key.startswith(f"variant:{sel}_r")


def _forced(group: str, args: argparse.Namespace) -> bool:
    """Is this stage's group force-rebuilt (``--force``, or ``--from`` reached it)?"""
    return args.force or (args.from_stage is not None and _GROUP_INDEX[group] >= _GROUP_INDEX[args.from_stage])


def _selected(group: str, stage_key: str, args: argparse.Namespace) -> bool:
    """Whether a stage is in ``--only`` scope; reported differently from a stage that is already done."""
    return args.only is None or _matches(group, stage_key, args.only)


def _should_run(
    state_path: Path, stage_key: str, group: str, ihash: str, artifact: Path, args: argparse.Namespace
) -> bool:
    """Whether an in-scope stage is forced or not yet done."""
    return _forced(group, args) or not stage_done(state_path, stage_key, ihash, artifact)


def _ensure_source(cfg: DatasetConfig, scene: str, ctx: ScenePaths, args: argparse.Namespace, plan: list) -> str:
    """Return the source ``sha256:`` hash, fetching the raw data first if it is not already present."""
    if ctx.source_dir.exists():
        return download.source_hash(ctx.source_dir)

    plan.append((scene, "download"))
    if args.dry_run:
        return "unknown"

    _, digest = download.fetch_source(cfg, scene, ctx.source_root)
    return digest


def _base_stage(
    cfg: DatasetConfig, scene: str, ctx: ScenePaths, src_hash: str, args: argparse.Namespace, plan: list
) -> None:
    """Run (or skip) the base dataset build: ffmpeg extract + COLMAP/MVS + writer."""
    if not _selected("base", "base", args):
        notice(f"[dim]base {scene}: out of scope for --only {args.only}.[/]")
        return

    ihash = input_hash(ctx.source_dir, cfg.scenes[scene].model_dump())

    if not _should_run(ctx.state_path, "base", "base", ihash, ctx.base_dir, args):
        notice(f"[yellow]base {scene}: already done — skipping (use --force to rebuild).[/]")
        return

    plan.append((scene, "base"))
    if args.dry_run:
        return

    colmap = ColmapOptions(
        gpu=args.colmap_gpu,
        threads=args.colmap_threads,
        max_image_size=args.colmap_max_image_size,
        dense=args.dense,
        dense_max_image_size=args.dense_max_image_size,
    )
    _do_base(cfg, scene, ctx, src_hash, ihash, colmap, forced=_forced("base", args))


def _do_base(
    cfg: DatasetConfig,
    scene: str,
    ctx: ScenePaths,
    src_hash: str,
    ihash: str,
    colmap: ColmapOptions,
    *,
    forced: bool,
) -> None:
    """Build the base dataset in place, resumably; the marker is written only on success.

    ``forced`` wipes both directories first; the scratch workdir is removed on success and kept on failure.
    """
    sc = cfg.scenes[scene]

    manifest = build_manifest(
        source=manifest_source(cfg, scene),
        source_url=cfg.source_url,
        source_hash=src_hash,
        importer=f"registry:{cfg.dataset}",
        cli_args={"scene": scene, "fps": sc.fps, "eval_cams": sc.eval_cams, "excluded_cams": sc.excluded_cams},
        extra_tools={"pycolmap": pycolmap.__version__},
    )

    if forced:
        shutil.rmtree(ctx.base_dir, ignore_errors=True)
        shutil.rmtree(ctx.work_dir, ignore_errors=True)
    ctx.base_dir.mkdir(parents=True, exist_ok=True)

    importer = build_importer(cfg, scene, ctx.source_root, ctx.work_dir, colmap)
    build_dataset(importer, ctx.base_dir, eval_cams=sc.eval_cams, manifest=manifest)

    mark_done(ctx.state_path, "base", ihash, {"pycolmap": pycolmap.__version__})
    shutil.rmtree(ctx.work_dir, ignore_errors=True)  # scratch no longer needed; kept only on interrupt
    notice(f"[green]base {scene} built[/]")


def _collect_variants(
    cfg: DatasetConfig, scene: str, ctx: ScenePaths, args: argparse.Namespace, plan: list
) -> list[VariantJob]:
    """Every ``(variant, scale)`` to build, except native at full resolution and undistorted on a pinhole base."""
    sc = cfg.scenes[scene]
    base_pinhole = _base_is_pinhole(ctx.base_dir)
    jobs: list[VariantJob] = []

    for name in sc.variants:
        if name == "undistorted" and base_pinhole:
            continue
        for scale in sc.scales:
            if name == "native" and scale == 1.0:
                continue

            stage_key = f"variant:{_variant_label(name, scale)}"
            if not _selected("variants", stage_key, args):
                continue

            out_dir = ctx.out_root / _variant_dirname(scene, name, scale)
            ihash = input_hash(ctx.base_dir, name, scale)

            if not _should_run(ctx.state_path, stage_key, "variants", ihash, out_dir, args):
                notice(
                    f"[yellow]variant {out_dir.name} ({scene}): already done — skipping (use --force to rebuild).[/]"
                )
                continue

            plan.append((scene, stage_key))
            if args.dry_run:
                continue

            forced = _forced("variants", args)
            jobs.append(VariantJob(ctx.base_dir, out_dir, name, scale, ctx.state_path, stage_key, ihash, {}, forced))
    return jobs


def _execute_variant(job: VariantJob, jobs: int) -> None:
    """Build one variant in place, resumably, and mark it done; ``forced`` wipes ``out_dir`` first."""
    spec = variant_spec(job.name, job.scale)

    if job.forced:
        shutil.rmtree(job.out_dir, ignore_errors=True)
    build_variant(job.base_dir, job.out_dir, spec, jobs=jobs)

    mark_done(job.state_path, job.stage_key, job.input_hash, job.tool_versions)


def _resolve_jobs(requested: int | None) -> int:
    """Renderer worker count: ``--jobs``, else CPU count minus 2 to keep the machine responsive."""
    if requested is not None:
        return max(1, requested)
    return max(1, (os.cpu_count() or 4) - 2)


def _run_variant_jobs(jobs: list[VariantJob], args: argparse.Namespace) -> None:
    """Execute queued variant builds serially; each one parallelizes internally across its cameras."""
    if not jobs:
        return

    workers = _resolve_jobs(args.jobs)
    for job in jobs:
        _execute_variant(job, workers)

    for job in jobs:
        notice(f"[green]variant {job.out_dir.name} built[/]")


def run(args: argparse.Namespace) -> None:
    """Load the registry and drive base (serial) then variant (pooled) builds for each scene."""
    cfg = load_registry(args.registry)
    if args.scene is not None:
        check_scene(cfg, args.scene)
    scenes = [args.scene] if args.scene is not None else list(cfg.scenes)
    plan: list = []
    variant_jobs: list[VariantJob] = []

    for scene in scenes:
        if not args.dry_run:
            rule(f"{cfg.dataset} · {scene}")
        ctx = _scene_paths(cfg, scene, args)
        # Only the base stage consumes the raw source (and its hash, for the manifest). Fetching it
        # for a variant-only run would download the whole raw corpus to build nothing.
        src_hash = _ensure_source(cfg, scene, ctx, args, plan) if _selected("base", "base", args) else "unused"
        _base_stage(cfg, scene, ctx, src_hash, args, plan)
        variant_jobs += _collect_variants(cfg, scene, ctx, args, plan)

    if args.dry_run:
        _print_plan(plan)
        return

    _run_variant_jobs(variant_jobs, args)


def _print_plan(plan: list) -> None:
    """List the planned ``(scene, stage)`` work for ``--dry-run`` (writes nothing)."""
    if not plan:
        notice("[dim]nothing to do — all stages up to date[/]")
        return

    notice("[bold]planned work (dry-run):[/]")
    for scene, stage in plan:
        notice(f"  {scene}: {stage}")


def _parse_args() -> argparse.Namespace:
    """Parse the orchestrator CLI."""
    # fmt: off
    parser = argparse.ArgumentParser(description="Registry-driven base + variant builder with resumable staging.")
    parser.add_argument("--registry", type=Path, required=True, help="Dataset registry YAML (conf/datasets/*.yaml).")
    parser.add_argument("--scene", type=str, default=None, help="Single scene to build (default: every scene).")
    parser.add_argument("--source-root", type=Path, required=True, help="Root holding the raw source subdirs.")
    parser.add_argument("--out-root", type=Path, required=True, help="Root for base + variant datasets.")
    parser.add_argument("--work-dir", type=Path, default=None, help="COLMAP scratch dir (default: <out>/<scene>_work).")
    parser.add_argument("--jobs", type=int, default=None, help="Per-camera renderer worker threads (default: CPU count - 2).")
    parser.add_argument("--force", action="store_true", help="Redo all stages, ignoring done markers.")
    parser.add_argument("--from", dest="from_stage", choices=list(_GROUP_INDEX), default=None, help="Redo from a stage onward.")
    parser.add_argument("--only", type=str, default=None, help="Run only one stage/variant (base/variants/<variant>); add --force to redo it.")
    parser.add_argument("--dry-run", action="store_true", help="List planned (scene, stage) work and exit.")
    parser.add_argument("--colmap-threads", type=int, default=4, help="COLMAP threads (lower = less RAM, slower).")
    parser.add_argument("--colmap-max-image-size", type=int, default=3200, help="SIFT feature-detection downscale cap.")
    parser.add_argument("--no-gpu", dest="colmap_gpu", action="store_false", help="Force CPU COLMAP (default: GPU).")
    parser.add_argument("--no-dense", dest="dense", action="store_false", help="Skip dense MVS; use sparse cloud.")
    parser.add_argument("--dense-max-image-size", type=int, default=1600, help="MVS resolution cap (GPU-VRAM lever).")
    return parser.parse_args()
    # fmt: on


def main() -> None:
    """Parse args and run the orchestrator."""
    run(_parse_args())


def _run_and_exit() -> None:
    """Run ``main`` then hard-exit, dodging pycolmap/glog's static-destructor segfault at teardown."""
    code = 0

    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _run_and_exit()
