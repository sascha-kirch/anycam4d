"""Photometric loss functions for 4D Gaussian Splatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

try:
    from fused_ssim import fused_ssim as _fused_ssim

    _FUSED_SSIM_AVAILABLE = True
except ImportError:
    _FUSED_SSIM_AVAILABLE = False


def _compute_ssim(
    renders: Float[Tensor, "C 3 H W"],
    targets: Float[Tensor, "C 3 H W"],
) -> Float[Tensor, ""]:
    """Mean SSIM: fused_ssim on CUDA tensors, a uniform-window approximation on CPU tensors."""
    if not renders.is_cuda:
        return _ssim_cpu(renders, targets)
    if not _FUSED_SSIM_AVAILABLE:
        raise RuntimeError("fused_ssim is not installed; run `pixi run setup`")
    return _fused_ssim(renders, targets)


def _ssim_cpu(
    x: Float[Tensor, "C 3 H W"],
    y: Float[Tensor, "C 3 H W"],
    window_size: int = 11,
) -> Float[Tensor, ""]:
    """Simple CPU SSIM approximation using uniform window filtering."""
    C1 = 0.01**2
    C2 = 0.03**2
    # merge C and channel dims for grouped conv
    N, C, H, W = x.shape
    x = x.reshape(N * C, 1, H, W)
    y = y.reshape(N * C, 1, H, W)

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device, dtype=x.dtype) / (window_size**2)

    mu_x = F.conv2d(x, kernel, padding=pad, groups=1)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=1)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=1) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=1) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=1) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()


def l1_ssim_loss(
    renders: Float[Tensor, "C H W 3"],
    targets: Float[Tensor, "C H W 3"],
    ssim_lambda: float,
    *,
    ssim_enabled: bool = True,
    validity: Bool[Tensor, "C H W 1"] | None = None,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    """L1 + SSIM loss over observed pixels; returns ``(total, l1, ssim)`` scalars.

    Unobserved pixels carry no ground truth, so they give neither loss nor gradient:

        L1 = sum(v · |render - target|) / (sum(v) · n_channels)

    SSIM zeroes both inputs outside ``v`` and rescales by ``numel / sum(v)``; the residual error is a
    one-window ring at the boundary. ``validity=None`` is the dense path.
    """
    if validity is None:
        l1 = F.l1_loss(renders, targets)
        weight = None
    else:
        weight = validity.to(renders.dtype)  # [C, H, W, 1], 0/1
        # clamp_min guards an all-invalid batch (division by an empty valid set).
        denom = (weight.sum() * renders.shape[-1]).clamp_min(1.0)
        l1 = (weight * (renders - targets).abs()).sum() / denom

    if ssim_enabled:
        # SSIM operates on [C, 3, H, W]
        renders_ssim = renders if weight is None else renders * weight
        targets_ssim = targets if weight is None else targets * weight
        renders_nchw = renders_ssim.permute(0, 3, 1, 2)
        targets_nchw = targets_ssim.permute(0, 3, 1, 2)
        ssim_val = 1.0 - _compute_ssim(renders_nchw, targets_nchw)
        if weight is not None:
            # Zeroed pixels score SSIM 1 (zero loss); rescaling recovers the mean over observed pixels.
            ssim_val = ssim_val * (weight.numel() / weight.sum().clamp_min(1.0))
        total = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_val
    else:
        ssim_val = torch.zeros_like(l1)
        total = l1

    return total, l1, ssim_val
