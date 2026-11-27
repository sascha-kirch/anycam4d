"""Absolute cost of one render and of render + loss + backward, AnyCam4D against EWA, same model and same views.

Quantity: end-to-end latency per render on one GPU, with a sync before and after every view, as a training loop that
reads the loss per render has. Excluded: image transfer (views stay on the GPU), optimizer step, densification.
Both backends draw one trained model through its native lens, so only the renderer differs and both draw the same
active Gaussians; they alternate per round so drift hits both. Every round starts on a cooled GPU, every pass is
warmed up, and a pass during which the GPU reported thermal or hardware slowdown is repeated and left out of the
statistics. One process gives a distribution over views; run it three times for between-process spread.

    pack (on the machine with data):  python scripts/reproduce/benchmark_cost.py pack --out bench_bundle
    run  (on the GPU under test):     python scripts/reproduce/benchmark_cost.py run --bundle bench_bundle --out r1.json
    report:                           python scripts/reproduce/benchmark_cost.py report r1.json r2.json r3.json

Lock the GPU clock below its throttling point for stable absolute times (needs admin rights; laptops may refuse):
    sudo nvidia-smi -lgc 1500,1500   ...   sudo nvidia-smi -rgc
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import BackendType
from anycam4d.eval.checkpoint import build_pipeline_from_checkpoint
from anycam4d.eval.checkpoint import denorm_transform
from anycam4d.losses.photometric import l1_ssim_loss
from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.data.types import TrainingSample
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

# scene: (run under runs/, data dir under the data root). One per real dataset; primitive counts differ.
SCENES = {
    "flame_steak": ("main_tables/ut_nat_fs_1", "dynerf/flame_steak_r2"),
    "corgi1": ("main_tables/ut_nat_fco1_1", "selfcap/ftgs_corgi1_r2"),
    "flames": ("main_tables/ut_nat_flam_1", "deepview/flames_r2"),
}
BACKENDS = (BackendType.ANYCAM4D, BackendType.EWA)
MODES = ("forward", "backward")
N_VIEWS = 40
N_WARMUP = 4  # untimed views before every pass: the clock ramps up after the cool-down idle
N_ROUNDS = 6
N_RETRIES = 2
SSIM_LAMBDA = 0.2
MIN_PSNR = 20.0  # a bundle or backend mismatch, not a quality statement
# nvidia-smi clocks_throttle_reasons bits that slow the GPU down for heat or hardware protection
SLOWDOWN_BITS = {0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
POWER_CAP_BIT = 0x4
SMI_FIELDS = "temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active"
CHUNK = 1_000_000


def pack(out: Path, runs: Path, data: Path) -> None:
    """Copy each checkpoint and save N_VIEWS fixed training views, so the run needs neither dataset nor loader."""
    for scene, (run, data_dir) in SCENES.items():
        target = out / scene
        target.mkdir(parents=True, exist_ok=True)
        source = runs / run / "checkpoints" / "last.pt"
        shutil.copyfile(source, target / "last.pt")
        state = torch.load(source, map_location="cpu", weights_only=False)
        dataset = MultiViewSequenceDataset(str(data / data_dir), split="train")
        transform = denorm_transform(state)
        if transform is not None:
            dataset.apply_viewmat_transform(transform)
        picks = np.random.default_rng(0).choice(len(dataset), size=N_VIEWS, replace=False)
        torch.save([view_of(dataset[int(i)]) for i in picks], target / "views.pt")
        print(scene, "packed", flush=True)


def view_of(sample: TrainingSample) -> dict:
    """One view as CPU tensors, the image as uint8."""
    image = sample.image[..., :3]
    if image.is_floating_point():
        image = (image.clamp(0, 1) * 255).round().to(torch.uint8)
    if sample.validity is not None:
        raise ValueError("benchmark scenes have no validity mask")
    return {
        "viewmat": sample.viewmat,
        "K": sample.K,
        "model": sample.camera_model,
        "distortion": sample.distortion_params,
        "timestamp": float(sample.timestamp),
        "image": image,
    }


def to_device(view: dict, device: torch.device) -> tuple[BatchedCamera, torch.Tensor, float]:
    """Camera, image [1 H W 3] in [0, 1] and timestamp on the GPU."""
    image = view["image"].to(device).float()[None] / 255.0
    camera = BatchedCamera(
        viewmats=view["viewmat"].to(device)[None],
        Ks=view["K"].to(device)[None],
        model=view["model"],
        width=image.shape[2],
        height=image.shape[1],
        distortion=view["distortion"].to(device)[None],
    )
    return camera, image, view["timestamp"]


class GpuMonitor:
    """Samples nvidia-smi every 100 ms in the background; answers what the GPU did during a time window."""

    def __init__(self) -> None:
        """Start the sampler."""
        self.samples: list[tuple[float, float, float, float, int]] = []
        self._proc = subprocess.Popen(  # noqa: S603
            ["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits", "-lms", "100", "-i", "0"],  # noqa: S607
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        while not self.samples and self._proc.poll() is None:
            time.sleep(0.05)

    def _read(self) -> None:
        """Parse one line per sample, stamped with the host clock on arrival."""
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            try:
                self.samples.append(
                    (time.perf_counter(), float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3], 16))
                )
            except (ValueError, IndexError):
                continue

    def temperature(self) -> float:
        """Latest temperature in degrees C."""
        return self.samples[-1][1]

    def window(self, start: float, end: float) -> dict:
        """Clock, temperature and slowdown reasons over samples in [start, end], plus the one just after."""
        inside = [s for s in self.samples if start <= s[0] <= end + 0.15]
        if not inside:
            return {"n": 0, "slowdown": [], "power_capped": False}
        reasons = 0
        for sample in inside:
            reasons |= sample[4]
        return {
            "n": len(inside),
            "sm_mhz": [min(s[2] for s in inside), max(s[2] for s in inside)],
            "temp_max": max(s[1] for s in inside),
            "power_max": max(s[3] for s in inside),
            "slowdown": [name for bit, name in SLOWDOWN_BITS.items() if reasons & bit],
            "power_capped": bool(reasons & POWER_CAP_BIT),
        }

    def stop(self) -> None:
        """Stop the sampler."""
        self._proc.terminate()


def cool_down(monitor: GpuMonitor, max_temp: float, timeout: float) -> float:
    """Wait until the GPU is at most ``max_temp``; returns the seconds waited."""
    start = time.perf_counter()
    torch.cuda.synchronize()
    while monitor.temperature() > max_temp and time.perf_counter() - start < timeout:
        time.sleep(1.0)
    return time.perf_counter() - start


def forward(
    model: FourDGaussians, pipe: RenderPipeline, camera: BatchedCamera, tau: float, _: Float[Tensor, "1 H W 3"]
) -> None:
    """Render only."""
    with torch.no_grad():
        pipe.render(model, camera, tau, model.sh_degree)


def backward(
    model: FourDGaussians, pipe: RenderPipeline, camera: BatchedCamera, tau: float, image: torch.Tensor
) -> None:
    """Render, L1 + SSIM, backward, gradients dropped."""
    out = pipe.render(model, camera, tau, model.sh_degree)
    l1_ssim_loss(out.colors, image, SSIM_LAMBDA)[0].backward()
    for param in model.splats.values():
        param.grad = None


def timed_pass(step: Callable, model: FourDGaussians, pipe: RenderPipeline, views: list) -> list[float]:
    """Milliseconds per view, after N_WARMUP untimed views."""
    for camera, image, tau in views[:N_WARMUP]:
        step(model, pipe, camera, tau, image)
    samples = []
    for camera, image, tau in views:
        torch.cuda.synchronize()
        start = time.perf_counter()
        step(model, pipe, camera, tau, image)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def gpu_state() -> str:
    """Clocks, temperature, power, throttle reasons and memory now."""
    query = (
        "name,driver_version,clocks.gr,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used"
    )
    return subprocess.run(  # noqa: S603
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def render_psnr(model: FourDGaussians, pipe: RenderPipeline, views: list) -> float:
    """Check that bundle and backend agree: mean PSNR of the clamped render on the benchmark views."""
    values = []
    with torch.no_grad():
        for camera, image, tau in views[:8]:
            colors = pipe.render(model, camera, tau, model.sh_degree).colors.clamp(0, 1)
            values.append(-10 * torch.log10(((colors - image) ** 2).mean()).item())
    return float(np.mean(values))


def active_count(model: FourDGaussians, tau: float) -> int:
    """Gaussians whose temporal weight passes the prefilter at ``tau``; chunked so large models fit."""
    splats = model.splats
    count = 0
    with torch.no_grad():
        for i in range(0, len(model), CHUNK):
            part = slice(i, i + CHUNK)
            var_t = build_cov4d_rot(splats["quats_l"][part], splats["quats_r"][part], splats["scales_4d_log"][part])[
                :, 3, 3
            ]
            dt = tau - splats["mu_t"][part].reshape(-1)
            weight = torch.exp(-0.5 * dt * dt / (var_t + TEMPORAL_VARIANCE_EPS))
            count += int((weight >= model.prefilter_threshold).sum())
    return count


def drawn_count(model: FourDGaussians, pipe: RenderPipeline, camera: BatchedCamera, tau: float) -> int:
    """Gaussians the renderer projects to a non-empty footprint in the image."""
    with torch.no_grad():
        radii = pipe.render(model, camera, tau, model.sh_degree).meta["radii"]
    return int((radii > 0).reshape(-1, radii.shape[-1]).any(-1).sum())


def counts(model: FourDGaussians, pipes: dict, views: list) -> dict:
    """Per view: temporally active and drawn Gaussians, the work every renderer shares."""
    return {
        "active": [active_count(model, tau) for _, _, tau in views],
        "drawn": {b.value: [drawn_count(model, pipes[b], camera, tau) for camera, _, tau in views] for b in BACKENDS},
    }


@dataclasses.dataclass
class Settings:
    """Run settings shared by every scene."""

    rounds: int
    max_temp: float
    cooldown_timeout: float


def measure_mode(
    mode: str, model: FourDGaussians, pipes: dict, views: list, monitor: GpuMonitor, settings: Settings
) -> dict:
    """Alternate the backends over the rounds; a pass with GPU slowdown is repeated and recorded as excluded."""
    step = forward if mode == "forward" else backward
    passes: list[dict] = []
    for round_index in range(settings.rounds):
        waited = cool_down(monitor, settings.max_temp, settings.cooldown_timeout)
        order = BACKENDS if round_index % 2 == 0 else BACKENDS[::-1]
        for backend in order:
            for attempt in range(N_RETRIES + 1):
                start = time.perf_counter()
                values = timed_pass(step, model, pipes[backend], views)
                state = monitor.window(start, time.perf_counter())
                valid = not state["slowdown"]
                passes.append(
                    {
                        "backend": backend.value,
                        "round": round_index,
                        "attempt": attempt,
                        "valid": valid,
                        "cooldown_s": waited,
                        "gpu": state,
                        "ms": values,
                    }
                )
                if valid:
                    break
                waited = cool_down(monitor, settings.max_temp, settings.cooldown_timeout)
    peak = {}
    for backend in BACKENDS:
        torch.cuda.reset_peak_memory_stats()
        timed_pass(step, model, pipes[backend], views[:4])
        peak[backend.value] = torch.cuda.max_memory_allocated() / 2**20
    ms = {b.value: [v for p in passes if p["backend"] == b.value and p["valid"] for v in p["ms"]] for b in BACKENDS}
    return {"ms": ms, "peak_mib": peak, "passes": passes}


def measure_scene(scene: str, bundle: Path, device: torch.device, monitor: GpuMonitor, settings: Settings) -> dict:
    """All modes of one scene; a mode that runs out of memory is recorded, not fatal."""
    state = torch.load(bundle / scene / "last.pt", map_location="cpu", weights_only=False)
    model, base = build_pipeline_from_checkpoint(state, device)
    del state
    config: RenderingConfig = base.render_cfg
    pipes = {b: RenderPipeline(backend=BackendFactory.create(dataclasses.replace(config, backend=b))) for b in BACKENDS}
    views = [to_device(v, device) for v in torch.load(bundle / scene / "views.pt", weights_only=False)]
    result: dict = {
        "gaussians": len(model),
        "height": views[0][1].shape[1],
        "width": views[0][1].shape[2],
        "gpu_before": gpu_state(),
        "psnr": {b.value: render_psnr(model, pipes[b], views) for b in BACKENDS},
        "counts": counts(model, pipes, views),
    }
    if min(result["psnr"].values()) < MIN_PSNR:
        print(f"WARNING {scene}: render PSNR {result['psnr']} below {MIN_PSNR}, bundle or backend mismatch", flush=True)
    for mode in MODES:
        try:
            result[mode] = measure_mode(mode, model, pipes, views, monitor, settings)
        except torch.cuda.OutOfMemoryError:
            print(f"{scene} {mode}: out of GPU memory", flush=True)
            result[mode] = {"oom": True}
            torch.cuda.empty_cache()
    result["gpu_after"] = gpu_state()
    return result


def git_state() -> dict:
    """Commit and modified tracked files under the code that runs."""

    def run(*args: str) -> str:
        """Stdout of one git command."""
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603, S607

    return {"sha": run("rev-parse", "HEAD"), "modified": run("status", "--porcelain", "--", "anycam4d", "cuda")}


def run_all(bundle: Path, out: Path, scenes: list[str], settings: Settings) -> None:
    """Measure the scenes in order and write the JSON after each."""
    device = torch.device("cuda")
    monitor = GpuMonitor()
    results: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "host": platform.node(),
        "git": git_state(),
        "gpu_start": gpu_state(),
        "config": {"views": N_VIEWS, "warmup": N_WARMUP, "retries": N_RETRIES, **dataclasses.asdict(settings)},
        "scenes": {},
    }
    for scene in scenes:
        try:
            results["scenes"][scene] = measure_scene(scene, bundle, device, monitor, settings)
        except torch.cuda.OutOfMemoryError:
            print(f"{scene}: out of GPU memory while loading, skipped", flush=True)
        torch.cuda.empty_cache()
        out.write_text(json.dumps(results))  # keep what finished if a later scene dies
        print(scene, "done", flush=True)
    monitor.stop()
    report([out])


def percentile(values: list[float], q: float) -> float:
    """Percentile as a float."""
    return float(np.percentile(values, q))


def pass_health(runs: list[dict], scene: str, mode: str) -> str:
    """Excluded passes, SM clock range and peak temperature over the valid passes of all processes."""
    passes = [p for run in runs for p in run["scenes"][scene][mode]["passes"]]
    valid = [p for p in passes if p["valid"] and p["gpu"]["n"]]
    excluded = len(passes) - sum(p["valid"] for p in passes)
    if not valid:
        return f"{excluded} excl."
    low = min(p["gpu"]["sm_mhz"][0] for p in valid)
    high = max(p["gpu"]["sm_mhz"][1] for p in valid)
    hot = max(p["gpu"]["temp_max"] for p in valid)
    capped = sum(p["gpu"]["power_capped"] for p in valid)
    return f"{excluded} excl., {low:.0f}-{high:.0f} MHz, max {hot:.0f} C, power-capped {capped}/{len(valid)}"


def scene_rows(scene: str, runs: list[dict]) -> list[str]:
    """One table row per mode, pooling the samples of all processes."""
    first = runs[0]["scenes"][scene]
    active = statistics.median(first["counts"]["active"]) / 1e6
    drawn = statistics.median(first["counts"]["drawn"]["anycam4d"]) / 1e6
    label = f"{scene} ({first['gaussians'] / 1e6:.2f} M, active {active:.2f} M, drawn {drawn:.2f} M, {first['width']}x{first['height']})"
    rows = []
    for mode in MODES:
        if any(run["scenes"][scene][mode].get("oom") for run in runs):
            rows.append(f"| {label} | {mode} | OOM | | | | |")
            continue
        ms = {b.value: [v for run in runs for v in run["scenes"][scene][mode]["ms"][b.value]] for b in BACKENDS}
        medians = {b: statistics.median(values) for b, values in ms.items()}
        per_process = [
            statistics.median(run["scenes"][scene][mode]["ms"]["anycam4d"])
            / statistics.median(run["scenes"][scene][mode]["ms"]["ewa"])
            for run in runs
        ]
        peak = first[mode]["peak_mib"]
        rows.append(
            f"| {label} | {mode} "
            f"| {medians['anycam4d']:.1f} ({percentile(ms['anycam4d'], 10):.1f}-{percentile(ms['anycam4d'], 90):.1f}) "
            f"| {medians['ewa']:.1f} ({percentile(ms['ewa'], 10):.1f}-{percentile(ms['ewa'], 90):.1f}) "
            f"| {statistics.median(per_process):.3f} ({min(per_process):.3f}-{max(per_process):.3f}) "
            f"| {peak['anycam4d']:.0f} / {peak['ewa']:.0f} | {pass_health(runs, scene, mode)} |"
        )
    return rows


def report(paths: list[Path]) -> None:
    """Print the table of one or more run files."""
    runs = [json.loads(path.read_text()) for path in paths]
    print(
        f"\nGPU {runs[0]['gpu']}, torch {runs[0]['torch']}, CUDA {runs[0]['cuda']}, git {runs[0]['git']['sha'][:8]}, {len(runs)} process(es)"
    )
    if runs[0]["git"]["modified"]:
        print("modified tracked files:\n" + runs[0]["git"]["modified"])
    print(
        "| scene | mode | ours ms median (p10-p90) | EWA ms median (p10-p90) | ours / EWA (range over processes) | peak MiB ours / EWA | GPU over valid passes |"
    )
    print("|---|---|---|---|---|---|---|")
    for scene in runs[0]["scenes"]:
        print("\n".join(scene_rows(scene, [r for r in runs if scene in r["scenes"]])))


def main() -> None:
    """Dispatch pack, run or report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--data", type=Path, default=Path("data"), help="DATA_ROOT of scripts/reproduce/main_tables.sh")
    r = sub.add_parser("run")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    r.add_argument("--rounds", type=int, default=N_ROUNDS)
    r.add_argument("--max-temp", type=float, default=65.0, help="start a round only at or below this GPU temperature")
    r.add_argument("--cooldown-timeout", type=float, default=300.0, help="seconds to wait for the GPU to cool")
    t = sub.add_parser("report")
    t.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.out, args.runs, args.data)
    elif args.command == "run":
        run_all(args.bundle, args.out, args.scenes, Settings(args.rounds, args.max_temp, args.cooldown_timeout))
    else:
        report(args.paths)


if __name__ == "__main__":
    main()
