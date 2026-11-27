"""Mask-weighted PSNR/SSIM: a metric reduced over ``mask == 1`` pixels only."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


def masked_psnr(
    render: Float[Tensor, "H W 3"],
    gt: Float[Tensor, "H W 3"],
    mask: Bool[Tensor, "H W"],
    data_range: float = 1.0,
) -> float:
    """PSNR restricted to ``mask``. ``inf`` for a zero masked MSE; ``nan`` for an empty mask."""
    m = mask.to(render.dtype)
    n = m.sum() * render.shape[-1]
    if n <= 0:
        return float("nan")
    mse = (((render - gt) ** 2) * m.unsqueeze(-1)).sum() / n
    if mse <= 0:
        return float("inf")
    return (10.0 * torch.log10(mse.new_tensor(data_range**2) / mse)).item()


def masked_ssim(
    ssim_map: Float[Tensor, "1 C H W"],
    mask: Bool[Tensor, "H W"],
) -> float:
    """Mask-weighted average of a per-pixel SSIM map.

    Args:
        ssim_map: The full SSIM map from
            ``StructuralSimilarityIndexMeasure(return_full_image=True)``, ``[1, C, H, W]``.
        mask: Region mask, ``[H, W]``.

    Returns:
        ``(ssim_map * mask).sum() / (mask.sum() * C)``; ``nan`` for an empty mask.

    Note:
        SSIM is computed with a Gaussian window over the whole frame *before* masking, so pixels
        within a window's half-width of a region boundary carry a value contaminated by
        out-of-region content. This is the accepted region-split approximation (unlike ``masked_psnr``,
        which is exactly pixelwise); the bias is confined to a thin boundary band.
    """
    m = mask.to(ssim_map.dtype)
    n = m.sum() * ssim_map.shape[1]
    if n <= 0:
        return float("nan")
    return ((ssim_map.squeeze(0) * m.unsqueeze(0)).sum() / n).item()


_SKIMAGE_WIN = 7


def ssim_map_3dgs(p: Float[Tensor, "1 C H W"], t: Float[Tensor, "1 C H W"]) -> Float[Tensor, "1 C H W"]:
    """SSIM₁ map as 3DGS ``loss_utils.ssim``: 11x11 Gaussian (sigma 1.5), zero padding, data range 1."""
    x = torch.arange(11, dtype=p.dtype, device=p.device) - 5
    g = torch.exp(-(x**2) / (2 * 1.5**2))
    g = g / g.sum()
    channels = p.shape[1]
    window = (g[:, None] * g[None, :]).expand(channels, 1, 11, 11)

    def blur(img: Float[Tensor, "1 C H W"]) -> Float[Tensor, "1 C H W"]:
        return F.conv2d(img, window, padding=5, groups=channels)

    return _ssim_from_moments(blur, p, t, data_range=1.0, cov_norm=1.0)


def ssim_map_skimage(p: Float[Tensor, "1 C H W"], t: Float[Tensor, "1 C H W"]) -> Float[Tensor, "1 C h w"]:
    """SSIM₂ map as skimage's default on float images: 7x7 uniform, sample covariance, data range 2.

    Only the border-free interior skimage averages is returned: ``h, w = H - 6, W - 6``.
    """

    def blur(img: Float[Tensor, "1 C H W"]) -> Float[Tensor, "1 C h w"]:
        return F.avg_pool2d(img, _SKIMAGE_WIN, stride=1)

    n = _SKIMAGE_WIN**2
    return _ssim_from_moments(blur, p, t, data_range=2.0, cov_norm=n / (n - 1))


def _ssim_from_moments(
    blur: Callable[[Tensor], Tensor],
    p: Float[Tensor, "1 C H W"],
    t: Float[Tensor, "1 C H W"],
    data_range: float,
    cov_norm: float,
) -> Float[Tensor, "1 C h w"]:
    """Per-pixel SSIM from windowed first and second moments (h, w: what ``blur`` leaves)."""
    mu_p, mu_t = blur(p), blur(t)
    var_p = cov_norm * (blur(p * p) - mu_p**2)
    var_t = cov_norm * (blur(t * t) - mu_t**2)
    cov = cov_norm * (blur(p * t) - mu_p * mu_t)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))


def literature_dssim(
    p: Float[Tensor, "1 3 H W"],
    t: Float[Tensor, "1 3 H W"],
    validity: Bool[Tensor, "H W"] | None,
) -> dict[str, float]:
    """``dssim1`` and ``dssim2`` = (1 - SSIM)/2, the two DSSIM families published 4D tables report."""
    if validity is None:
        validity = torch.ones(p.shape[-2:], dtype=torch.bool, device=p.device)
    r = _SKIMAGE_WIN // 2
    ssim1 = masked_ssim(ssim_map_3dgs(p, t), validity)
    ssim2 = masked_ssim(ssim_map_skimage(p, t), validity[r:-r, r:-r])
    return {"dssim1": (1.0 - ssim1) / 2.0, "dssim2": (1.0 - ssim2) / 2.0}


def mean_view_psnr(
    pred: Float[Tensor, "C H W 3"],
    target: Float[Tensor, "C H W 3"],
    validity: Bool[Tensor, "C H W 1"] | None = None,
) -> float:
    """Mean of per-view PSNRs over observed pixels (all pixels when ``validity`` is None).

    Views with no observed pixel are skipped; ``nan`` if no view has one.
    """
    values = []
    for c in range(pred.shape[0]):
        mask = torch.ones_like(pred[c, ..., 0], dtype=torch.bool) if validity is None else validity[c, ..., 0]
        value = masked_psnr(pred[c].float(), target[c].float(), mask)
        if not math.isnan(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")
