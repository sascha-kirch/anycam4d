"""Tests for the color_correct port."""

from __future__ import annotations

import torch

from anycam4d.eval.color_correct import color_correct


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0.0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def test_color_correct_identity() -> None:
    """Correcting an image against itself is (near) a no-op."""
    torch.manual_seed(0)
    x = torch.rand(16, 16, 3)
    out = color_correct(x, x)
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-3)


def test_color_correct_recovers_scaled() -> None:
    """A globally scaled render is brought closer to the reference after correction."""
    torch.manual_seed(1)
    ref = torch.rand(32, 32, 3) * 0.8 + 0.1  # keep inside the unclipped band
    render = (ref * 0.6).clamp(0, 1)
    corrected = color_correct(render, ref)
    assert _psnr(corrected, ref) > _psnr(render, ref)
