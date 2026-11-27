"""The fused kernels read sh0 and shN in place: colour and coefficient gradients against autograd through the oracle.

Degrees below the model's maximum are the case to pin: shN keeps its full stride there, so a kernel that
indexes it as if it were sliced reads (and scatters into) the wrong bases.
"""

from __future__ import annotations

import pytest
import torch

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import AppearanceType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from tests.oracles.sh4d import eval_sh_4d
from tests.oracles.sh4d_multiplicative import eval_shfs_4d


requires_cuda = pytest.mark.skipif(
    not HAS_CUDA_EXT or not torch.cuda.is_available(), reason="the colour leaf runs in the CUDA kernels"
)

_DEVICE = "cuda"
_N = 512
_TAU = 0.2
_W = _H = 256
_TEMPORAL = {AppearanceType.SH4D_ADDITIVE: "sh_temporal", AppearanceType.SH4D_MULTIPLICATIVE: "sh_spacetime"}


def _viewmats(n_cameras: int = 3) -> torch.Tensor:
    """Cameras on a ring around the origin, looking inward."""
    mats = []
    for c in range(n_cameras):
        angle = torch.tensor(2 * torch.pi * c / n_cameras)
        eye = torch.stack([4.0 * torch.cos(angle), torch.tensor(1.0), 4.0 * torch.sin(angle)])
        forward = torch.nn.functional.normalize(-eye, dim=0)
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0])), dim=0)
        rotation = torch.stack([right, torch.linalg.cross(right, forward), forward], dim=0)
        mat = torch.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = -rotation @ eye
        mats.append(mat)
    return torch.stack(mats, dim=0).to(_DEVICE)


def _fused_colors(
    model: FourDGaussians, sh_degree: int, backend: str, packed: bool, viewmats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(camera, Gaussian) colours ``[C, N, 3]`` from one fused kernel, and which pairs it wrote."""
    from anycam4d.cuda.autograd import EWAFusedFunction
    from anycam4d.cuda.autograd import EWAFusedPackedFunction
    from anycam4d.cuda.autograd import UTFusedFunction
    from anycam4d.cuda.autograd import UTFusedPackedFunction

    s = model.splats
    ai = model.appearance_inputs()
    C = viewmats.shape[0]
    Ks = torch.tensor([[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]], device=_DEVICE).expand(C, 3, 3)
    geometry = [s[k].detach() for k in ("quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities")]
    head = (
        *geometry,
        ai.sh0,
        ai.shN,
        ai.coeffs,
        sh_degree,
        ai.kind_id,
        ai.n_terms,
        ai.stride,
        viewmats,
        Ks.contiguous(),
    )
    tail = (_TAU, model.cov_eps, 0.3, 0.01, 1e10, 0.0, _W, _H)
    if backend == "ewa":
        args = (*head, torch.empty(0, device=_DEVICE), 0, *tail, False)
        fn = EWAFusedPackedFunction if packed else EWAFusedFunction
    else:
        args = (*head, torch.zeros(C, 12, device=_DEVICE), *tail, 0)
        fn = UTFusedPackedFunction if packed else UTFusedFunction
    out = fn.apply(*args)

    if not packed:
        colors, radii = (out[2], out[5]) if backend == "ewa" else (out[4], out[3])
        return colors, (radii > 0).all(dim=-1)
    cam, gid = out[0].long(), out[1].long()
    seen = torch.zeros(C, _N, dtype=torch.bool, device=_DEVICE)
    seen[cam, gid] = True
    if backend == "ut":  # the UT packed kernel scatters its colours into a dense [C, N, 3]
        return out[7], seen
    colors = torch.zeros(C, _N, 3, device=_DEVICE).index_put((cam, gid), out[5])
    return colors, seen


def _reference_colors(model: FourDGaussians, sh_degree: int, viewmats: torch.Tensor) -> torch.Tensor:
    """The same colours from the oracle, at the conditioned means the kernel uses."""
    from tests.fused_stages import condition

    s = model.splats
    geometry = [s[k].detach() for k in ("quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities")]
    means3d = condition(*geometry, _TAU, model.cov_eps, 0.0)[0]
    rotations, translations = viewmats[:, :3, :3], viewmats[:, :3, 3]
    campos = -(rotations.transpose(-2, -1) @ translations.unsqueeze(-1)).squeeze(-1)
    dirs = torch.nn.functional.normalize(means3d[None] - campos[:, None], dim=-1)
    dirs_t = _TAU - s["mu_t"].detach()
    ai = model.appearance_inputs()
    sh = torch.cat([s["sh0"], s["shN"]], dim=1)

    colors = []
    for d in dirs:
        if model.appearance_model is AppearanceType.SH4D_MULTIPLICATIVE:
            full = torch.cat([sh, s["sh_spacetime"]], dim=1)
            colors.append(torch.clamp(eval_shfs_4d(sh_degree, ai.n_terms, full, d, dirs_t, sh.shape[1]) + 0.5, min=0.0))
        else:
            temporal = s["sh_temporal"] if "sh_temporal" in s else torch.zeros(_N, 0, 3, device=_DEVICE)
            colors.append(eval_sh_4d(sh, temporal, d, dirs_t, sh_degree, ai.n_terms))
    return torch.stack(colors)


@requires_cuda
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
@pytest.mark.parametrize("appearance", list(AppearanceType))
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("backend", ["ewa", "ut"])
def test_sh_gradients_reach_sh0_and_shN_in_place(
    backend: str, packed: bool, appearance: AppearanceType, sh_degree: int
) -> None:
    """Every fused kernel: colours and d_sh0 / d_shN / d_temporal match autograd through the oracle."""
    torch.manual_seed(5)
    model = FourDGaussians.from_random(_N, sh_degree=3, appearance_model=appearance).to(_DEVICE)
    keys = ["sh0", "shN", *([_TEMPORAL[appearance]] if appearance in _TEMPORAL else [])]
    for k in keys:
        model.splats[k].data.normal_(0.0, 0.3)
    viewmats = _viewmats()
    weights = torch.randn(viewmats.shape[0], _N, 3, device=_DEVICE)

    colors, seen = _fused_colors(model, sh_degree, backend, packed, viewmats)
    assert seen.sum() > 100, "too few visible pairs: the fixture would prove nothing"
    (colors * weights)[seen].sum().backward()
    fused = {k: model.splats[k].grad.clone() for k in keys}

    model.zero_grad(set_to_none=True)
    expected = _reference_colors(model, sh_degree, viewmats)
    (expected * weights)[seen].sum().backward()

    torch.testing.assert_close(colors[seen], expected[seen].detach(), rtol=0, atol=1e-5)
    for k in keys:
        torch.testing.assert_close(fused[k], model.splats[k].grad, rtol=1e-4, atol=1e-5, msg=k)
    inactive = model.splats["shN"].grad[:, (sh_degree + 1) ** 2 - 1 :]
    assert not fused["shN"][:, (sh_degree + 1) ** 2 - 1 :].any() and not inactive.any()
