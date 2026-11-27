"""Tests for anycam4d loss functions."""

from __future__ import annotations

import torch

from anycam4d.losses.mask import background_alpha_loss
from anycam4d.losses.photometric import l1_ssim_loss


def test_l1_ssim_loss_shape() -> None:
    """All three returned values are scalar tensors."""
    renders = torch.rand(2, 64, 64, 3)
    targets = torch.rand(2, 64, 64, 3)
    total, l1, ssim = l1_ssim_loss(renders, targets, ssim_lambda=0.2, ssim_enabled=False)
    assert total.shape == (), f"total should be scalar, got shape {total.shape}"
    assert l1.shape == (), f"l1 should be scalar, got shape {l1.shape}"
    assert ssim.shape == (), f"ssim should be scalar, got shape {ssim.shape}"


def test_l1_loss_only_when_ssim_disabled() -> None:
    """With ssim_enabled=False, total loss equals l1 and ssim component is zero."""
    torch.manual_seed(42)
    renders = torch.rand(2, 32, 32, 3)
    targets = torch.rand(2, 32, 32, 3)
    total, l1, ssim_val = l1_ssim_loss(renders, targets, ssim_lambda=0.3, ssim_enabled=False)
    assert torch.allclose(total, l1), f"total={total.item():.6f}, l1={l1.item():.6f} — must be equal when SSIM disabled"
    assert ssim_val.item() == 0.0, f"ssim component should be 0.0 when disabled, got {ssim_val.item()}"


def test_validity_all_true_matches_dense_path() -> None:
    """v == 1 everywhere reproduces the dense (validity=None) loss exactly."""
    torch.manual_seed(7)
    renders = torch.rand(2, 32, 32, 3)
    targets = torch.rand(2, 32, 32, 3)
    valid = torch.ones(2, 32, 32, 1, dtype=torch.bool)

    dense = l1_ssim_loss(renders, targets, ssim_lambda=0.2)
    masked = l1_ssim_loss(renders, targets, ssim_lambda=0.2, validity=valid)
    for a, b, name in zip(dense, masked, ("total", "l1", "ssim"), strict=True):
        assert torch.allclose(a, b, atol=1e-6), f"{name}: dense={a.item():.8f} masked={b.item():.8f}"

    alphas = torch.rand(2, 32, 32, 1)
    fg = (torch.rand(2, 32, 32, 1) > 0.5).float()
    dense_bg = background_alpha_loss(alphas, fg)
    masked_bg = background_alpha_loss(alphas, fg, validity=valid)
    assert torch.allclose(dense_bg, masked_bg), f"dense={dense_bg.item()} masked={masked_bg.item()}"


def test_masked_l1_ignores_invalid_error() -> None:
    """Error confined to invalid pixels contributes nothing to the masked L1."""
    renders = torch.zeros(1, 8, 8, 3)
    targets = torch.zeros(1, 8, 8, 3)
    valid = torch.ones(1, 8, 8, 1, dtype=torch.bool)
    valid[:, :4] = False
    targets[:, :4] = 100.0  # huge error, all of it unobserved

    _, l1, _ = l1_ssim_loss(renders, targets, ssim_lambda=0.2, ssim_enabled=False, validity=valid)
    assert l1.item() == 0.0, f"invalid-only error must not contribute, got {l1.item()}"


def test_masked_l1_matches_hand_computed_sum() -> None:
    """Masked L1 equals sum(v*|r-t|) / (sum(v)*C) on a random case."""
    torch.manual_seed(3)
    renders = torch.rand(2, 16, 16, 3)
    targets = torch.rand(2, 16, 16, 3)
    valid = torch.rand(2, 16, 16, 1) > 0.4

    _, l1, _ = l1_ssim_loss(renders, targets, ssim_lambda=0.2, ssim_enabled=False, validity=valid)
    w = valid.float()
    expected = (w * (renders - targets).abs()).sum() / (w.sum() * 3)
    assert torch.allclose(l1, expected, atol=1e-6), f"got {l1.item():.8f}, expected {expected.item():.8f}"


def test_invalid_pixels_receive_zero_gradient() -> None:
    """Gradient w.r.t. the render is exactly zero at unobserved pixels."""
    torch.manual_seed(11)
    renders = torch.rand(1, 16, 16, 3, requires_grad=True)
    targets = torch.rand(1, 16, 16, 3)
    valid = torch.ones(1, 16, 16, 1, dtype=torch.bool)
    valid[:, 8:] = False

    _, l1, _ = l1_ssim_loss(renders, targets, ssim_lambda=0.2, ssim_enabled=False, validity=valid)
    l1.backward()
    assert renders.grad is not None, "no gradient computed"
    invalid_grad = renders.grad[~valid.expand_as(renders)]
    valid_grad = renders.grad[valid.expand_as(renders)]
    assert invalid_grad.abs().max().item() == 0.0, f"invalid grad max {invalid_grad.abs().max().item()}"
    assert valid_grad.abs().sum().item() > 0.0, "valid region received no gradient"


def test_masked_l1_normalizes_by_sum_v_not_numel() -> None:
    """Adding more invalid area leaves the masked L1 unchanged (numel() division would shrink it)."""
    torch.manual_seed(5)
    core_r = torch.rand(1, 8, 8, 3)
    core_t = torch.rand(1, 8, 8, 3)

    valid_small = torch.ones(1, 8, 8, 1, dtype=torch.bool)
    _, l1_small, _ = l1_ssim_loss(core_r, core_t, ssim_lambda=0.2, ssim_enabled=False, validity=valid_small)

    # Same valid pixels, padded with a large invalid region.
    big_r = torch.rand(1, 8, 32, 3)
    big_t = torch.rand(1, 8, 32, 3)
    big_r[:, :, :8] = core_r
    big_t[:, :, :8] = core_t
    valid_big = torch.zeros(1, 8, 32, 1, dtype=torch.bool)
    valid_big[:, :, :8] = True
    _, l1_big, _ = l1_ssim_loss(big_r, big_t, ssim_lambda=0.2, ssim_enabled=False, validity=valid_big)

    assert torch.allclose(l1_small, l1_big, atol=1e-6), (
        f"loss must not depend on the invalid fraction: {l1_small.item():.8f} vs {l1_big.item():.8f}"
    )


def test_background_alpha_loss_gated_by_validity() -> None:
    """High alpha in an unobserved background region is only penalised without the validity gate."""
    alphas = torch.zeros(1, 8, 8, 1)
    alphas[:, :4] = 1.0
    fg = torch.zeros(1, 8, 8, 1)  # everything is "background"
    valid = torch.ones(1, 8, 8, 1, dtype=torch.bool)
    valid[:, :4] = False  # ...but the high-alpha half was never observed

    ungated = background_alpha_loss(alphas, fg)
    gated = background_alpha_loss(alphas, fg, validity=valid)
    assert ungated.item() > 0.4, f"ungated loss should penalise the unobserved region, got {ungated.item()}"
    assert gated.item() == 0.0, f"gated loss should be zero, got {gated.item()}"


def test_empty_valid_set_is_finite() -> None:
    """An all-invalid batch returns finite values instead of 0/0."""
    renders = torch.rand(1, 8, 8, 3)
    targets = torch.rand(1, 8, 8, 3)
    valid = torch.zeros(1, 8, 8, 1, dtype=torch.bool)

    total, l1, ssim_val = l1_ssim_loss(renders, targets, ssim_lambda=0.2, validity=valid)
    for name, v in (("total", total), ("l1", l1), ("ssim", ssim_val)):
        assert torch.isfinite(v), f"{name} is not finite: {v.item()}"
    assert l1.item() == 0.0, f"empty valid set should give zero L1, got {l1.item()}"

    alphas = torch.rand(1, 8, 8, 1)
    bg = background_alpha_loss(alphas, torch.zeros(1, 8, 8, 1), validity=valid)
    assert torch.isfinite(bg) and bg.item() == 0.0, f"bg_alpha should be finite zero, got {bg.item()}"


def test_masked_ssim_path_runs_and_is_finite() -> None:
    """The masked SSIM path (inputs zeroed at invalid pixels, then rescaled) is finite."""
    torch.manual_seed(13)
    renders = torch.rand(1, 32, 32, 3)
    targets = torch.rand(1, 32, 32, 3)
    valid = torch.ones(1, 32, 32, 1, dtype=torch.bool)
    valid[:, 16:] = False

    _, _, ssim_masked = l1_ssim_loss(renders, targets, ssim_lambda=0.2, validity=valid)
    assert torch.isfinite(ssim_masked), f"masked ssim not finite: {ssim_masked.item()}"
    assert ssim_masked.item() > 0.0, "a mismatched valid region must still drive the SSIM term"


def test_masked_ssim_magnitude_independent_of_invalid_fraction() -> None:
    """Masked SSIM keeps its scale as the invalid fraction grows (``numel / sum(v)`` rescale)."""
    torch.manual_seed(17)
    h = w = 128
    renders = torch.rand(1, h, w, 3)
    targets = (renders + 0.05 * torch.randn_like(renders)).clamp(0.0, 1.0)

    for cut in (96, 64, 32):  # 25%, 50%, 75% invalid
        valid = torch.ones(1, h, w, 1, dtype=torch.bool)
        valid[:, :, cut:] = False
        _, _, masked = l1_ssim_loss(renders, targets, ssim_lambda=0.2, validity=valid)
        _, _, cropped = l1_ssim_loss(renders[:, :, :cut], targets[:, :, :cut], ssim_lambda=0.2)
        rel_err = abs(masked.item() / cropped.item() - 1.0)
        invalid_frac = 1.0 - cut / w
        assert rel_err < 0.10, (
            f"{invalid_frac:.0%} invalid: masked={masked.item():.6f} vs valid-region "
            f"{cropped.item():.6f} ({rel_err:.1%} off) — the rescale should keep these close"
        )
        # The bug this pins: no rescale => masked == (1 - invalid_frac) * cropped.
        assert rel_err < invalid_frac / 2, f"{invalid_frac:.0%} invalid: looks diluted, not rescaled"
