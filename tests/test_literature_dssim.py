"""DSSIM₁/DSSIM₂ match the implementations published 4D tables were scored with (STGS ``test.py``)."""

from __future__ import annotations

from math import exp

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from anycam4d.eval.metrics import literature_dssim

try:
    from fused_ssim import fused_ssim
except ImportError:
    fused_ssim = None


def _3dgs_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Gaussian-window SSIM of pytorch-ssim (MIT), the form 3DGS and STGS evaluate with (DSSIM₁)."""
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * 1.5**2)) for x in range(window_size)])
    _1d = (gauss / gauss.sum()).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    channel = img1.size(-3)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous().type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    s11 = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1.pow(2)
    s22 = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2.pow(2)
    s12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1.pow(2) + mu2.pow(2) + c1) * (s11 + s22 + c2))
    return m.mean().item()


def _pair(noise: float) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:48, 0:64] / 16.0
    gt = np.stack([np.sin(xx), np.cos(yy), np.sin(xx + yy)], axis=0) * 0.4 + 0.5
    render = np.clip(gt + rng.normal(0, noise, gt.shape), 0, 1)
    return torch.tensor(render, dtype=torch.float32)[None], torch.tensor(gt, dtype=torch.float32)[None]


@pytest.mark.parametrize("noise", [0.01, 0.1])
def test_dssim_matches_published_implementations(noise: float) -> None:
    p, t = _pair(noise)
    got = literature_dssim(p, t, None)

    ref2 = structural_similarity(
        p[0].permute(1, 2, 0).double().numpy(), t[0].permute(1, 2, 0).double().numpy(), channel_axis=-1, data_range=2.0
    )
    assert got["dssim2"] == pytest.approx((1 - ref2) / 2, abs=1e-6)
    assert got["dssim1"] == pytest.approx((1 - _3dgs_ssim(p, t)) / 2, abs=1e-6)
    if fused_ssim is not None and torch.cuda.is_available():
        ref1 = fused_ssim(p.cuda(), t.cuda()).item()
        assert got["dssim1"] == pytest.approx((1 - ref1) / 2, abs=1e-5)
