"""Trainer — full training loop for 4D Gaussian Splatting."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.valid_mask import ValidMaskCache
from anycam4d.console import notice
from anycam4d.eval.evaluator import Evaluator
from anycam4d.eval.result import lower_is_better
from anycam4d.losses.mask import background_alpha_loss
from anycam4d.losses.photometric import l1_ssim_loss
from anycam4d.losses.regularization import opacity_reg
from anycam4d.losses.regularization import scale_reg
from anycam4d.primitives.appearance import active_degrees
from anycam4d.provenance import code_provenance
from anycam4d.renderer.bilateral_grid_post_processor import BilateralGridPostProcessor
from anycam4d.strategy.mcmc import MCMCSpatialStrategy
from anycam4d.trainer import diagnostics
from anycam4d.trainer.activity import build_activity_mask
from anycam4d.trainer.setup import build_bilagrid_optimizer
from anycam4d.trainer.setup import build_optimizers
from anycam4d.trainer.setup import build_schedulers


if TYPE_CHECKING:
    from gsplat.strategy.base import Strategy
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor
    from torch.utils.data import DataLoader

    from anycam4d.config.training import TrainingConfig
    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
    from anycam4d.data.normalization import SceneNormalization
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.backend import RenderOutput
    from anycam4d.renderer.pipeline import RenderPipeline
    from anycam4d.viewer.viewer import Anycam4DViewer


# Parameters whose per-render gradient the strategies read, as ``meta["<name>_grad"]``.
_RENDER_GRAD_PARAMS = ("means",)


def _get_gsplat_version() -> str:
    """Return the installed gsplat version string, or 'unknown' if not available."""
    try:
        return importlib.metadata.version("gsplat")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class MicroBatch:
    """One render's inputs, already on the device: the unit gradient accumulation repeats."""

    images: Float[Tensor, "C H W 3"]
    masks: Float[Tensor, "C H W 1"] | None
    validity: Bool[Tensor, "C H W 1"] | None
    camera: BatchedCamera
    timestamp: float
    image_ids: list[int]
    camera_ids: list[int]


def shared_timestamp(timestamps: list[float]) -> float:
    """The one timestamp a render is conditioned at; raises when a batch mixes several.

    One scalar ``tau`` reaches the kernels, so a *render* cannot span timestamps. A *step* can:
    ``training.accum_renders`` renders separately and accumulates.
    """
    timestamp = timestamps[0]
    if any(abs(t - timestamp) > 1e-8 for t in timestamps):
        raise ValueError(
            f"a render requires a single shared timestamp across its cameras (frame-aligned); "
            f"got mixed timestamps {timestamps}. Only sampler=frame_aligned is C>1-capable."
        )
    return timestamp


def step_resolution(seen: tuple[int, int] | None, images: Float[Tensor, "C H W 3"]) -> tuple[int, int]:
    """Return this step's render resolution, rejecting a step whose renders disagree on it.

    Accumulation itself is resolution-agnostic — each render is independent and the primitives never see
    the image grid — but two quantities downstream are not: the per-render loss is a per-pixel mean, so
    averaging over renders would weight a large and a small image equally, and the screen-space growth
    statistic is scaled into pixels per render, so one ``grow_grad2d`` would pool two different units.
    """
    shape = (int(images.shape[1]), int(images.shape[2]))
    if seen is not None and shape != seen:
        raise ValueError(
            f"a step's renders must share one resolution; got {seen} and {shape}. Mixed resolutions need "
            f"a per-pixel loss weighting and a resolution-free growth statistic first."
        )
    return shape


def add_losses(acc: dict[str, Float[Tensor, ""]], losses: dict[str, Float[Tensor, ""]]) -> None:
    """Sum detached loss scalars across the renders of one optimizer step."""
    for name, value in losses.items():
        detached = value.detach()
        acc[name] = detached if name not in acc else acc[name] + detached


def add_scalars(acc: dict[str, float], scalars: dict[str, float]) -> None:
    """Sum logged scalars across the renders of one optimizer step."""
    for name, value in scalars.items():
        acc[name] = value if name not in acc else acc[name] + value


def warn_if_mcmc_misregularized(strategy: Strategy, training_cfg: TrainingConfig) -> None:
    """Warn when an MCMC run has no opacity regularization: nothing creates the dead set relocation needs."""
    if not isinstance(strategy, MCMCSpatialStrategy):
        return
    if training_cfg.opacity_reg_lambda == 0:
        warnings.warn(
            "MCMC strategy with opacity_reg_lambda=0: relocation manufactures its dead set "
            "from driven-down opacities — with none it starves and the run may stall. Set "
            "training.opacity_reg_lambda > 0 (the shipped MCMC preset uses 0.002).",
            RuntimeWarning,
            stacklevel=2,
        )


def raise_on_non_finite_loss(step: int, total: float, parts: dict[str, float]) -> None:
    """Stop the run on a non-finite loss.

    NaN primitives are culled by the rasterizer, so image metrics keep looking healthy after the
    parameters are gone.

    Raises:
        RuntimeError: If any term is NaN or infinite, naming the ones that are.
    """
    nonfinite = [name for name, value in (("total", total), *parts.items()) if not math.isfinite(value)]
    if nonfinite:
        raise RuntimeError(f"Step {step}: loss is not finite ({', '.join(nonfinite)}); training cannot recover.")


class Trainer:
    """Training loop over pre-built model, datasets, strategy and pipeline; owns optimizers and logging.

    Args:
        model: Initialized FourDGaussians on target device.
        train_dataset: Training MultiViewSequenceDataset (viewmats already transformed).
        eval_dataset: Eval MultiViewSequenceDataset (viewmats already transformed), None to disable in-loop evaluation.
        train_dataloader: DataLoader wrapping train_dataset with configured sampler.
        strategy: Densification strategy (e.g. MCMCSpatialStrategy).
        normalization: Scene normalization parameters.
        training_cfg: Training schedule, learning rates, loss weights, and optimizer settings.
        pipeline: Render pipeline (conditioning -> appearance -> rasterization -> post-processing).
        result_dir: Directory for checkpoints and TensorBoard logs.
        viewer: Pre-built viewer, or None to disable.
        config_snapshot: Resolved config stored in every checkpoint; eval rebuilds the pipeline from it.
    """

    def __init__(  # noqa: PLR0913 — primitive type, datasets, strategy, configs each demand a slot
        self,
        model: FourDGaussians,
        train_dataset: MultiViewSequenceDataset,
        eval_dataset: MultiViewSequenceDataset | None,
        train_dataloader: DataLoader,
        strategy: Strategy,
        normalization: SceneNormalization,
        *,
        training_cfg: TrainingConfig,
        pipeline: RenderPipeline,
        result_dir: str | Path,
        viewer: Anycam4DViewer | None = None,
        config_snapshot: dict | None = None,
    ) -> None:
        """Store the pre-built components; ``setup`` builds the optimizers."""
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._trainloader = train_dataloader
        self.strategy = strategy
        self.normalization = normalization

        self.training_cfg = training_cfg
        self.pipeline = pipeline
        self.result_dir = Path(result_dir)
        self._config_snapshot = config_snapshot

        self.device = next(model.parameters()).device
        # Packed once here: the mask is static per camera, so the render path only gathers it.
        self._valid_masks = ValidMaskCache(train_dataset, self.device)
        self._viewer = viewer
        self._step = 0
        self._step_timestamps: list[float] = []

    def setup(self) -> None:
        """Build optimizers, schedulers, strategy state, dataloader iterator, and writer."""
        if self.training_cfg.accum_renders < 1:
            raise ValueError(f"training.accum_renders must be >= 1, got {self.training_cfg.accum_renders}")

        self.optimizers = build_optimizers(self.model, self.training_cfg, scene_scale=self.normalization.scene_scale)
        self.schedulers = build_schedulers(self.optimizers, self.training_cfg)

        # The bilateral grid is not per-Gaussian, so it lives outside the (sparse) optimizer group.
        self.bilagrid_optimizer: torch.optim.Optimizer | None = None
        post_processor = self.pipeline.post_processor

        if post_processor is not None and isinstance(post_processor, BilateralGridPostProcessor):
            opt, sched = build_bilagrid_optimizer(post_processor, self.training_cfg)
            self.bilagrid_optimizer = opt
            self.schedulers.append(sched)

        # Without the prefilter, temporally gated primitives still project and STSO stops freezing them.
        if self.training_cfg.selective_adam and self.model.prefilter_threshold == 0:
            warnings.warn(
                "training.selective_adam=true with model.prefilter_threshold == 0: the activity mask collapses to "
                "spatial-only visibility (temporal gate lost). Set prefilter_threshold > 0.",
                RuntimeWarning,
                stacklevel=2,
            )

        warn_if_mcmc_misregularized(self.strategy, self.training_cfg)

        self.strategy_state = self.strategy.initialize_state(
            scene_scale=self.normalization.scene_scale,
            time_extent=1.0,  # mu_t is normalized to [-0.5, 0.5]; extent = 1.0
        )
        self._trainloader_iter = iter(self._trainloader)
        self._init_writer()
        self._init_evaluator()

    def _init_evaluator(self) -> None:
        """Build the Evaluator when an eval set exists; color-corrected metrics when a bilateral grid is trained."""
        self._evaluator: Evaluator | None = None
        if self.eval_dataset is None:
            return
        compute_cc = isinstance(self.pipeline.post_processor, BilateralGridPostProcessor)
        self._evaluator = Evaluator(self.device, lpips_net=self.training_cfg.lpips_net, compute_cc=compute_cc)
        self._best_metric_value = math.inf if lower_is_better(self.training_cfg.best_metric) else -math.inf

    def _init_writer(self) -> None:
        """Initialize TensorBoard SummaryWriter in result_dir/tb/."""
        tb_dir = self.result_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tb_dir))
        self._log_provenance()

    def _log_provenance(self) -> None:
        """Record the producing commit to ``run_provenance.json`` and to TensorBoard (the copy that travels with the logs)."""
        provenance = code_provenance()
        (self.result_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        lines = "\n".join(f"- **{key}**: `{value}`" for key, value in provenance.items())
        self.writer.add_text("provenance/code", lines, global_step=0)
        # Only a tracked modification invalidates the SHA.
        dirty = " (tracked files MODIFIED — SHA does not describe this run)" if provenance["git_dirty"] else ""
        untracked = " (+untracked files present)" if provenance["git_untracked"] else ""
        notice(f"Code provenance: {provenance['git_sha']} on {provenance['git_branch']}{dirty}{untracked}")

    def train(self) -> None:
        """Run the full training loop."""
        max_steps = self.training_cfg.max_steps
        self._last_log_str = ""

        # The bar is redrawn only where its numbers change — on the logging cadence — so a redirected
        # log file gets one line per logged step instead of one per step. `mininterval=inf` switches
        # off tqdm's own refresh; the postfix below is the only thing that draws, plus `close()` at the end.
        log_every = self.training_cfg.log_every
        with tqdm(total=max_steps, unit="step", dynamic_ncols=True, colour="blue", mininterval=float("inf")) as pbar:
            for step in range(max_steps):
                self._train_step(step)
                pbar.update(1)

                if step % log_every == 0:
                    pbar.set_postfix_str(self._last_log_str)

        final_step = max_steps - 1
        self._run_final_eval(final_step)
        self.save_checkpoint(final_step, name="last" if self.training_cfg.save_best else None)
        self.writer.close()

        if self._viewer is not None:
            notice("Training complete. Viewer running... Ctrl+C to exit.")
            time.sleep(1_000_000)

    def _train_step(self, step: int) -> None:
        """Single training iteration with viewer lock."""
        if self._viewer is not None:
            self._viewer.lock.acquire()
        try:
            self._train_step_body(step)
        finally:
            if self._viewer is not None:
                self._viewer.lock.release()

    def _train_step_body(self, step: int) -> None:
        """One optimizer step over ``accum_renders`` separately rendered micro-batches.

        Each render contributes ``loss / accum_renders``; the optimizer, the schedulers and the
        densification step run once, on the accumulated gradient.
        """
        t0 = time.perf_counter()
        n_renders = self.training_cfg.accum_renders
        active = torch.zeros(len(self.model), dtype=torch.bool, device=self.device)
        grad_running: dict[str, Tensor] = {}
        loss_acc: dict[str, Float[Tensor, ""]] = {}
        metric_acc: dict[str, float] = {}
        packed = self.pipeline.render_cfg.packed
        resolution: tuple[int, int] | None = None
        logging_step = step % self.training_cfg.log_every == 0

        self._step = step
        self._step_timestamps = []
        for _ in range(n_renders):
            micro = self._next_micro_batch()
            self._step_timestamps.append(micro.timestamp)
            resolution = step_resolution(resolution, micro.images)
            out = self._render_micro_batch(step, micro)
            total, parts = self._compute_loss(out, micro.images, micro.masks, step, micro.timestamp, micro.validity)
            (total / n_renders).backward()
            self._hand_render_grads_to(out.meta, grad_running, n_renders)
            packed = out.meta.get("packed", packed)
            self.strategy.accumulate(self.model.splats, self.strategy_state, step, out.meta, packed=packed)
            active |= self._render_activity(out)
            add_losses(loss_acc, {"total": total, **parts})
            if logging_step:
                add_scalars(
                    metric_acc,
                    diagnostics.train_psnr(micro.images, out.colors, micro.validity),
                )

        losses = {name: value / n_renders for name, value in loss_acc.items()}
        total_loss = losses.pop("total")
        metrics = {name: value / n_renders for name, value in metric_acc.items()}

        self._optimizer_step(active)

        for sched in self.schedulers:
            sched.step()

        # Before zero_grad: the strategy accumulates parameter gradients.
        self.strategy.step_post_backward(
            self.model.splats,
            self.optimizers,
            self.strategy_state,
            step,
            out.meta,
            packed=packed,
            lr=self._noise_reference_lr(),
        )

        self._zero_grad()

        step_ms = (time.perf_counter() - t0) * 1e3
        # Losses and metrics are the step's mean over its renders; the images kept for TensorBoard are the
        # last micro-batch's, since an image cannot be averaged into something a human should look at.
        self._post_step(step, total_loss, losses, metrics, micro.images, out, step_ms)

    def _next_micro_batch(self) -> MicroBatch:
        """Pull one loader batch onto the device (the collate dict: leading-C tensors + scalar lists)."""
        try:
            batch = next(self._trainloader_iter)
        except StopIteration:
            self._trainloader_iter = iter(self._trainloader)
            batch = next(self._trainloader_iter)

        images = batch["images"].to(self.device, non_blocking=True)  # [C, H, W, 3]
        masks = None
        if batch["masks"] is not None:
            masks = batch["masks"].to(self.device, non_blocking=True)  # [C, H, W, 1]
            images = images * masks

        # Validity weights the loss reduction; None means every pixel was observed.
        v_cpu = batch["validity"]
        validity = v_cpu.to(self.device, non_blocking=True) if v_cpu is not None else None

        camera = BatchedCamera(
            viewmats=batch["viewmats"].to(self.device, non_blocking=True),  # [C, 4, 4]
            Ks=batch["Ks"].to(self.device, non_blocking=True),  # [C, 3, 3]
            model=batch["camera_model"],
            width=images.shape[2],
            height=images.shape[1],
            distortion=batch["distortion_params"].to(self.device, non_blocking=True),  # [C, P]
            valid_mask=self._valid_masks.for_batch(batch["camera_ids"], images.shape[2], images.shape[1]),
        )

        return MicroBatch(
            images=images,
            masks=masks,
            validity=validity,
            camera=camera,
            timestamp=shared_timestamp(batch["timestamps"]),
            image_ids=batch["image_ids"],
            camera_ids=batch["camera_ids"],
        )

    def _render_micro_batch(self, step: int, micro: MicroBatch) -> RenderOutput:
        """Render one micro-batch and let the strategy retain whatever it reads after backward."""
        sh_degree_now, sh_degree_t_now = active_degrees(
            step, self.training_cfg.sh_degree_interval, self.model.sh_degree
        )

        out = self.pipeline.render(
            self.model,
            micro.camera,
            micro.timestamp,
            sh_degree_now,
            sh_degree_t=sh_degree_t_now,
            image_ids=micro.image_ids,
            camera_ids=micro.camera_ids,
        )

        self.strategy.step_pre_backward(
            self.model.splats,
            self.optimizers,
            self.strategy_state,
            step,
            out.meta,
        )
        return out

    def _hand_render_grads_to(self, meta: dict, running: dict[str, Tensor], n_renders: int) -> None:
        """Put this render's own ``means`` gradient and the loss rescale into ``meta``.

        ``.grad`` is the running sum over a step's renders, so a render's own signal is its increment;
        reading the accumulator directly would weight the first render ``n_renders`` times. ``running``
        holds the previous sums. ``grad_scale`` undoes ``loss / n_renders``, keeping the growth thresholds
        at their batch-1 meaning.
        """
        meta["grad_scale"] = float(n_renders)
        for name in _RENDER_GRAD_PARAMS:
            grad = self.model.splats[name].grad if name in self.model.splats else None
            if grad is None:
                continue
            prev = running.get(name)
            meta[f"{name}_grad"] = grad if prev is None else grad - prev
            running[name] = grad.detach().clone() if n_renders > 1 else grad

    def _render_activity(self, out: RenderOutput) -> Bool[Tensor, "N"]:
        """Primitives this render updated; the step steps the union over its renders."""
        return build_activity_mask(out.meta, len(self.model), self.device)

    def _noise_reference_lr(self) -> float:
        """The scheduled means LR MCMC noise anneals with."""
        return self.optimizers.optimizers["means"].param_groups[0]["lr"]

    def _zero_grad(self) -> None:
        """Zero the Gaussian optimizer group and the separate dense bilagrid optimizer."""
        self.optimizers.zero_grad(set_to_none=True)
        if self.bilagrid_optimizer is not None:
            self.bilagrid_optimizer.zero_grad(set_to_none=True)

    def _optimizer_step(self, active: Bool[Tensor, "N"]) -> None:
        """Step the Gaussian optimizers over the active set (dense Adam ignores it) and the bilateral grid.

        ``active`` is the **union** over the step's renders: a primitive that received gradient from any
        of them must be stepped, or that contribution is dropped with no error.
        """
        self._raise_on_non_finite("before the optimizer step")
        self.optimizers.step(visibility=active)
        self.model.clamp_temporal_mean(self.model.mu_t_margin_sigmas, self.model.prefilter_threshold)
        self._raise_on_non_finite("after the optimizer step")
        if self.bilagrid_optimizer is not None:
            self.bilagrid_optimizer.step()

    def _raise_on_non_finite(self, phase: str) -> None:
        """Dump the first non-finite primitives and stop; later steps would only spread the NaN.

        Raises:
            RuntimeError: If any parameter value or gradient is non-finite.
        """
        if not diagnostics.any_nonfinite(self.model.splats):
            return
        bad = diagnostics.nonfinite_rows(self.model.splats)
        step = self._step
        context = {"phase": phase, "timestamps": self._step_timestamps}
        path = self.result_dir / f"nonfinite_step{step:06d}.pt"
        summary = diagnostics.dump_nonfinite(path, step, self.model.splats, self.optimizers.optimizers, bad, context)
        self.writer.add_text("nonfinite", summary, step)
        self.writer.flush()
        raise RuntimeError(summary)

    def _post_step(
        self,
        step: int,
        total_loss: Float[Tensor, ""],
        losses: dict[str, Float[Tensor, ""]],
        metrics: dict[str, float],
        images: Float[Tensor, "C H W 3"],
        out: RenderOutput,
        step_ms: float,
    ) -> None:
        """Handle per-step side effects: logging, checkpointing, viewer update."""
        if step % self.training_cfg.log_every == 0:
            self._log(step, total_loss, losses, metrics, images, out.colors, step_ms)

        if self.training_cfg.save_every > 0 and step > 0 and step % self.training_cfg.save_every == 0:
            self.save_checkpoint(step)

        if self._viewer is not None and step % self._viewer.update_every_n_steps == 0:
            self._viewer.update_model(self.model)

        self._maybe_eval(step)

    def _maybe_eval(self, step: int) -> None:
        """Eval at ``(step + 1) % eval_every == 0``, off the refine and opacity-reset steps (as gsplat does)."""
        if self._evaluator is None or self.training_cfg.eval_every <= 0:
            return
        if step != self.training_cfg.max_steps - 1 and (step + 1) % self.training_cfg.eval_every == 0:
            self._run_eval(step)

    def _run_final_eval(self, step: int) -> None:
        """Run one eval after the loop (outside the per-step viewer-lock scope)."""
        if self._evaluator is None or self.training_cfg.eval_every <= 0:
            return
        if self._viewer is not None:
            self._viewer.lock.acquire()
        try:
            self._run_eval(step)
        finally:
            if self._viewer is not None:
                self._viewer.lock.release()

    def _run_eval(self, step: int) -> None:
        """Evaluate on the eval set and log mean, std and the worst frame/camera."""
        assert self._evaluator is not None  # noqa: S101
        assert self.eval_dataset is not None  # noqa: S101
        self.model.eval()
        result = self._evaluator.evaluate(self.model, self.pipeline, self.eval_dataset)
        self.model.train()
        for k, v in result.mean.items():
            self.writer.add_scalar(f"eval/{k}", v, step)
            self.writer.add_scalar(f"eval/{k}_std", result.std[k], step)
        diagnostics.log_eval_marginals(self.writer, result, step)
        notice(f"[bold yellow] (eval @ {step}) " + result.format().replace("\n", "  ") + "[/]")
        self._last_eval_metrics = result.mean.keys()
        self._maybe_save_best(step, result.mean.get(self.training_cfg.best_metric))

    def _maybe_save_best(self, step: int, value: float | None) -> None:
        """Keep one ``best.pt`` for the best eval metric so far, alongside the final ``last.pt``.

        Selection reads the held-out eval split, so a number quoted from ``best.pt`` is selected on that
        split — state it wherever such a number is reported.
        """
        if not self.training_cfg.save_best:
            return
        if value is None:
            raise KeyError(
                f"training.best_metric={self.training_cfg.best_metric!r} is not produced by the evaluator "
                f"(available: {sorted(self._last_eval_metrics)})"
            )
        better = (
            value < self._best_metric_value
            if lower_is_better(self.training_cfg.best_metric)
            else value > self._best_metric_value
        )
        if not better:
            return
        self._best_metric_value = value
        self.save_checkpoint(step, name="best")

    def _compute_loss(
        self,
        out: RenderOutput,
        images: Float[Tensor, "C H W 3"],
        masks: Float[Tensor, "C H W 1"] | None,
        step: int,
        timestamp: float,
        validity: Bool[Tensor, "C H W 1"] | None = None,
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        """Weighted total loss; image terms reduce over observed pixels, regularizers act on primitives."""
        ssim_enabled = step >= self.training_cfg.ssim_start_step
        total_loss, l1_loss, ssim_loss = l1_ssim_loss(
            out.colors, images, self.training_cfg.ssim_lambda, ssim_enabled=ssim_enabled, validity=validity
        )
        losses: dict[str, Float[Tensor, ""]] = {"l1": l1_loss, "ssim": ssim_loss}

        if masks is not None and self.training_cfg.mask_bg_alpha_lambda > 0:
            bg_alpha = background_alpha_loss(out.alphas, masks, validity=validity)
            total_loss = total_loss + self.training_cfg.mask_bg_alpha_lambda * bg_alpha
            losses["bg_alpha"] = bg_alpha

        total_loss = self._add_regularization_losses(total_loss, losses, timestamp)

        post_processor = self.pipeline.post_processor
        if (
            post_processor is not None
            and isinstance(post_processor, BilateralGridPostProcessor)
            and self.training_cfg.bilagrid_tv_lambda > 0
        ):
            tv = post_processor.tv_loss()
            total_loss = total_loss + self.training_cfg.bilagrid_tv_lambda * tv
            losses["bilagrid_tv"] = tv

        return total_loss, losses

    def _add_regularization_losses(
        self, total_loss: Float[Tensor, ""], losses: dict[str, Float[Tensor, ""]], tau: float
    ) -> Float[Tensor, ""]:
        """Add ``scale_reg`` and the timestamp-dependent ``opacity_reg`` when enabled."""
        cfg = self.training_cfg
        if cfg.scale_reg_lambda > 0:
            s_reg = scale_reg(self.model.intrinsic_spatial_scales())
            total_loss = total_loss + cfg.scale_reg_lambda * s_reg
            losses["scale_reg"] = s_reg

        if cfg.opacity_reg_lambda > 0:
            o_reg = opacity_reg(self.model.splats, tau, detach_gate=cfg.opacity_reg_detach_gate)
            total_loss = total_loss + cfg.opacity_reg_lambda * o_reg
            losses["opacity_reg"] = o_reg

        return total_loss

    def _log(
        self,
        step: int,
        total_loss: Float[Tensor, ""],
        losses: dict[str, Float[Tensor, ""]],
        metrics: dict[str, float],
        gt_images: Float[Tensor, "C H W 3"],
        rendered_images: Float[Tensor, "C H W 3"],
        step_ms: float,
    ) -> None:
        """Log progress to the bar and TensorBoard.

        ``losses`` and ``metrics`` are already the step's mean over its renders; the images are the last
        render's, and are written only under ``save_images``.
        """
        psnr = metrics.get("train/psnr", float("nan"))

        vram_mb = torch.cuda.max_memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
        n_prim = len(self.model)
        self._last_log_str = f"loss={total_loss.item():.4f} psnr={psnr:.2f} N={n_prim} vram={vram_mb:.0f}MB"

        total = total_loss.item()
        parts = {name: val.item() for name, val in losses.items()}
        raise_on_non_finite_loss(step, total, parts)

        self.writer.add_scalar("loss/total", total, step)
        for name, val in parts.items():
            self.writer.add_scalar(f"loss/{name}", val, step)
        self.writer.add_scalar("timing/step_ms", step_ms, step)
        self.writer.add_scalar("gaussians/count", n_prim, step)

        # train/psnr against eval/psnr: overfitting vs optimization failure.
        diagnostics.log_scalars(self.writer, metrics, step)

        if self.device.type == "cuda":
            self.writer.add_scalar("memory/gpu_peak_mb", vram_mb, step)
            torch.cuda.reset_peak_memory_stats()

        if self.training_cfg.save_images:
            self.writer.add_image("render/GT", gt_images[0], step, dataformats="HWC")
            self.writer.add_image("render/rendered", rendered_images[0], step, dataformats="HWC")

        diagnostics.log_refine_events(self.writer, self.strategy_state, step)

        opts = self.optimizers.optimizers
        if "means" in opts:
            self.writer.add_scalar("lr/means", opts["means"].param_groups[0]["lr"], step)

        if "mu_t" in opts:
            self.writer.add_scalar("lr/temporal", opts["mu_t"].param_groups[0]["lr"], step)

        if self.bilagrid_optimizer is not None:
            self.writer.add_scalar("lr/bilagrid", self.bilagrid_optimizer.param_groups[0]["lr"], step)

    def save_checkpoint(self, step: int, name: str | None = None) -> Path:
        """Save the model for eval and viewing to ``result_dir/checkpoints/step_{step:06d}.pt``.

        Args:
            step: Training step the weights come from.
            name: File stem to write instead of ``step_{step:06d}``; overwritten in place, which is how
                ``best`` and ``last`` stay at two files per run.

        Training cannot be resumed from it: no optimizer or strategy state is written.
        """
        ckpt_dir = self.result_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{name or f'step_{step:06d}'}.pt"

        checkpoint = {
            "step": step,
            "model_state": self.model.state_dict_with_meta(
                self.normalization.norm_scale,
                self.normalization.norm_center,
                self.train_dataset.t_min,
                self.train_dataset.t_max,
            ),
            "norm_scale": self.normalization.norm_scale,
            "norm_center": self.normalization.norm_center.tolist(),
            "scene_scale": self.normalization.scene_scale,
            "t_min": self.train_dataset.t_min,
            "t_max": self.train_dataset.t_max,
            "gsplat_version": _get_gsplat_version(),
            "provenance": code_provenance(),
            "config": self._config_snapshot,
        }

        torch.save(checkpoint, ckpt_path)
        notice(f"Checkpoint saved: {ckpt_path}")
        return ckpt_path
