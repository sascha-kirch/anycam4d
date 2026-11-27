"""The temporal-SH reference: basis layout, reduction to spatial SH, and what it detaches."""

from __future__ import annotations

import math

import pytest
import torch

from tests.oracles.sh4d import Sh4dAppearanceModel
from tests.oracles.sh4d import eval_sh_4d
from tests.oracles.sh4d import temporal_cosine_basis
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def test_temporal_cosine_basis_interleaving() -> None:
    """Verify the interleaving matches [cos(k*pi*tau), sin(k*pi*tau)] for k=1..n."""
    tau = 0.3
    n_freqs = 3
    feat = temporal_cosine_basis(tau, n_freqs)
    for i in range(n_freqs):
        k = i + 1
        expected_cos = math.cos(math.pi * k * tau)
        expected_sin = math.sin(math.pi * k * tau)
        assert abs(feat[2 * i].item() - expected_cos) < 1e-6, f"cos mismatch at k={k}"
        assert abs(feat[2 * i + 1].item() - expected_sin) < 1e-6, f"sin mismatch at k={k}"


_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@_REQUIRES_CUDA
def test_eval_sh_4d_zero_temporal_matches_spatial_sh() -> None:
    """With sh_temporal=0, output equals pure spatial SH evaluation."""
    from gsplat import spherical_harmonics

    torch.manual_seed(2)
    N, K, sh_degree = 15, 16, 3
    T_basis = 8
    device = _CUDA_DEVICE

    sh_spatial = torch.randn(N, K, 3, device=device)
    sh_temporal = torch.zeros(N, T_basis, 3, device=device)
    dirs = torch.randn(N, 3, device=device)
    mu_t = torch.linspace(-0.4, 0.4, N, device=device)

    colors_4d = eval_sh_4d(sh_spatial, sh_temporal, dirs, dir_t=0.7 - mu_t, sh_degree=sh_degree)
    dirs_norm = torch.nn.functional.normalize(dirs, dim=-1)
    # eval_sh_4d adds +0.5 and clamps; match that here for a fair comparison
    colors_3d = torch.clamp(spherical_harmonics(sh_degree, dirs_norm, sh_spatial) + 0.5, min=0.0)

    assert torch.allclose(colors_4d, colors_3d, atol=1e-5), "Mismatch vs pure spatial SH"


@_REQUIRES_CUDA
def test_sh4d_appearance_model_detaches_directions() -> None:
    """Appearance model detaches dirs and dir_t: no gradients to means or mu_t."""
    torch.manual_seed(8)
    N = 10
    device = _CUDA_DEVICE
    model = FourDGaussians.from_random(N, sh_degree=3, appearance_model="sh4d_additive").to(device)
    appearance = Sh4dAppearanceModel(n_temporal_freqs=4)

    means = model.splats["means"]
    viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    colors = appearance.eval_colors(model.splats, means, viewmat, tau=0.5, sh_degree=3)
    colors.sum().backward()

    assert model.splats["means"].grad is None, "means should not receive gradient through SH eval (detached)"
    assert model.splats["mu_t"].grad is None, "mu_t should not receive gradient through SH eval (detached)"
    assert model.splats["sh0"].grad is not None, "sh0 must receive gradient"
    assert model.splats["sh_temporal"].grad is not None, "sh_temporal must receive gradient"


@pytest.mark.parametrize(
    ("appearance", "n_freqs"), [("sh4d_additive", 1), ("sh4d_additive", 4), ("sh4d_multiplicative", 4)]
)
def test_model_carries_only_its_own_temporal_parameter(appearance: str, n_freqs: int) -> None:
    model = FourDGaussians.from_random(10, sh_degree=1, appearance_model=appearance, n_temporal_freqs=n_freqs)
    if appearance == "sh4d_additive":
        assert model.splats["sh_temporal"].shape == (10, 2 * n_freqs, 3)
        assert "sh_spacetime" not in model.splats
    else:
        assert model.splats["sh_spacetime"].shape == (10, 2 * 4, 3)
        assert "sh_temporal" not in model.splats
