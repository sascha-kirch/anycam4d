"""Tests for mask-weighted PSNR/SSIM (`anycam4d.eval.metrics`)."""

from __future__ import annotations

import math

import pytest
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from anycam4d.eval.metrics import masked_psnr
from anycam4d.eval.metrics import mean_view_psnr
from anycam4d.eval.metrics import masked_ssim


def _images(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    render = torch.rand(16, 16, 3, generator=g)
    gt = torch.rand(16, 16, 3, generator=g)
    return render, gt


def _ssim_full_map(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)
    _, full = ssim(render.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0))
    return full


def test_masked_psnr_all_ones_matches_unmasked() -> None:
    render, gt = _images()
    mask = torch.ones(16, 16, dtype=torch.bool)
    got = masked_psnr(render, gt, mask)

    psnr = PeakSignalNoiseRatio(data_range=1.0)
    expected = psnr(render.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0)).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_masked_ssim_all_ones_matches_unmasked() -> None:
    render, gt = _images()
    mask = torch.ones(16, 16, dtype=torch.bool)
    full_map = _ssim_full_map(render, gt)
    got = masked_ssim(full_map, mask)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    expected = ssim(render.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0)).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_masked_psnr_restricts_to_mask() -> None:
    render, gt = _images()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:8, :] = True  # top half only

    got = masked_psnr(render, gt, mask)

    sub_render, sub_gt = render[:8], gt[:8]
    mse = ((sub_render - sub_gt) ** 2).mean()
    expected = (10.0 * torch.log10(torch.tensor(1.0) / mse)).item()
    assert got == pytest.approx(expected, rel=1e-5)

    # Corrupting pixels outside the mask must not move the masked score.
    corrupted_gt = gt.clone()
    corrupted_gt[8:] = 1.0 - corrupted_gt[8:]
    assert masked_psnr(render, corrupted_gt, mask) == pytest.approx(got, rel=1e-5)


def test_masked_ssim_restricts_to_mask() -> None:
    render, gt = _images()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    mask[:, :8] = True  # left half only

    full_map = _ssim_full_map(render, gt)
    got = masked_ssim(full_map, mask)
    expected = (full_map.squeeze(0)[:, :, :8].sum() / (mask.sum() * full_map.shape[1])).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_masked_psnr_empty_mask_is_nan() -> None:
    render, gt = _images()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    assert masked_psnr(render, gt, mask) != masked_psnr(render, gt, mask)  # nan != nan


def test_masked_ssim_empty_mask_is_nan() -> None:
    render, gt = _images()
    mask = torch.zeros(16, 16, dtype=torch.bool)
    full_map = _ssim_full_map(render, gt)
    result = masked_ssim(full_map, mask)
    assert result != result  # nan != nan


def test_masked_psnr_identical_images_is_inf() -> None:
    render, _ = _images()
    mask = torch.ones(16, 16, dtype=torch.bool)
    assert masked_psnr(render, render.clone(), mask) == float("inf")


def test_mean_view_psnr_averages_views_over_observed_pixels() -> None:
    torch.manual_seed(0)
    target = torch.rand(2, 8, 8, 3)
    pred = torch.rand(2, 8, 8, 3)
    validity = torch.zeros(2, 8, 8, 1, dtype=torch.bool)
    validity[:, :, :4] = True
    pred_masked_out = pred.clone()
    pred_masked_out[:, :, 4:] = 0.0  # differences outside the mask must not move the metric

    per_view = [masked_psnr(pred[c], target[c], validity[c, ..., 0]) for c in range(2)]
    assert math.isclose(mean_view_psnr(pred_masked_out, target, validity), sum(per_view) / 2, rel_tol=1e-6)
    assert mean_view_psnr(target, target) == float("inf")
