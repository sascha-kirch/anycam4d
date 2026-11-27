"""TensorBoard diagnostics for a training run.

Tag names are stable: renaming one splits its series across runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch

from anycam4d.eval.metrics import mean_view_psnr
from anycam4d.strategy.events import EVENTS_KEY


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter

    from anycam4d.eval.result import EvalResult


# Rows kept in a non-finite dump; enough to characterize the population without a multi-GB file.
_NONFINITE_DUMP_ROWS = 4096


@torch.no_grad()
def any_nonfinite(params: dict[str, nn.Parameter]) -> bool:
    """Whether any parameter value or gradient is non-finite; one reduction per tensor, one sync."""
    sums = [p.sum() for p in params.values()] + [p.grad.sum() for p in params.values() if p.grad is not None]
    return not bool(torch.isfinite(torch.stack(sums)).all())


@torch.no_grad()
def nonfinite_rows(params: dict[str, nn.Parameter]) -> Bool[Tensor, "N"]:
    """Primitives with a non-finite value or gradient in any parameter."""
    n = len(params["means"])
    bad = torch.zeros(n, dtype=torch.bool, device=params["means"].device)
    for p in params.values():
        bad |= ~torch.isfinite(p.reshape(n, -1)).all(dim=1)
        if p.grad is not None:
            bad |= ~torch.isfinite(p.grad.reshape(n, -1)).all(dim=1)
    return bad


@torch.no_grad()
def dump_nonfinite(
    path: Path,
    step: int,
    params: dict[str, nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    bad: Bool[Tensor, "N"],
    context: dict[str, Any],
) -> str:
    """Save the offending rows (values, gradients, optimizer state) and return a text summary for TensorBoard."""
    n = len(bad)
    ids = bad.nonzero().flatten()[:_NONFINITE_DUMP_ROWS]
    record: dict[str, Any] = {"step": step, "n": n, "n_bad": int(bad.sum()), "ids": ids.cpu(), "context": context}
    record["params"] = {k: p[ids].cpu() for k, p in params.items()}
    record["grads"] = {k: p.grad[ids].cpu() for k, p in params.items() if p.grad is not None}
    record["optimizer_state"] = {
        k: {sk: v[ids].cpu() for sk, v in opt.state[params[k]].items() if torch.is_tensor(v) and v.shape[:1] == (n,)}
        for k, opt in optimizers.items()
        if k in params
    }
    torch.save(record, path)

    lines = [f"step {step}: {record['n_bad']} of {n} primitives non-finite; context {context}; dump {path}"]
    for k, p in params.items():
        n_val = int((~torch.isfinite(p.reshape(n, -1)).all(dim=1)).sum())
        n_grad = int((~torch.isfinite(p.grad.reshape(n, -1)).all(dim=1)).sum()) if p.grad is not None else 0
        if n_val or n_grad:
            lines.append(f"- {k}: {n_val} non-finite values, {n_grad} non-finite gradients")
    return "\n".join(lines)


def log_scalars(writer: SummaryWriter, values: dict[str, float], step: int) -> None:
    """Write a tag -> value dict."""
    for tag, value in values.items():
        writer.add_scalar(tag, value, step)


def log_eval_marginals(writer: SummaryWriter, result: EvalResult, step: int) -> None:
    """Worst frame and worst camera per metric, by id, plus the per-frame distribution.

    The camera marginal is skipped for a single-camera split, where it would equal the mean.
    """
    for k in result.mean:
        if result.per_timestamp.get(k):
            frame_id, value = result.worst_timestamp(k)
            writer.add_scalar(f"eval/{k}_worst_frame", value, step)
            writer.add_scalar(f"eval/{k}_worst_frame_id", frame_id, step)
            finite = np.asarray(result.per_timestamp[k], dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:  # add_histogram raises on an empty series
                writer.add_histogram(f"eval/{k}_per_frame", finite, step)
        if len(result.per_camera.get(k, ())) > 1:
            camera_id, value = result.worst_camera(k)
            writer.add_scalar(f"eval/{k}_worst_camera", value, step)
            writer.add_scalar(f"eval/{k}_worst_camera_id", camera_id, step)


def log_refine_events(writer: SummaryWriter, state: dict[str, Any], step: int) -> None:
    """Cumulative densification counters (``refine/cloned``, ``refine/relocated``, ...)."""
    events = state.get(EVENTS_KEY)
    if events:
        log_scalars(writer, {f"refine/{name}": total for name, total in events.items()}, step)


@torch.no_grad()
def train_psnr(
    gt_images: Float[Tensor, "C H W 3"],
    rendered_images: Float[Tensor, "C H W 3"],
    validity: Bool[Tensor, "C H W 1"] | None,
) -> dict[str, float]:
    """``train/psnr`` of one render: the clamped render over observed pixels, as eval scores it.

    Returned rather than written: a step accumulates several renders and the series is their mean.
    """
    return {"train/psnr": mean_view_psnr(rendered_images.clamp(0, 1), gt_images, validity)}
