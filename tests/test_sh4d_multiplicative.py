"""The multiplicative (published 4D SH) colour leaf, against a transcription of eval_shfs_4d."""

from __future__ import annotations

import pytest
import torch

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import AppearanceType
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.primitives.appearance import active_degrees
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from tests.oracles.sh4d_multiplicative import eval_shfs_4d


requires_cuda = pytest.mark.skipif(
    not HAS_CUDA_EXT or not torch.cuda.is_available(), reason="the colour leaf runs in the CUDA kernels"
)

_DEVICE = "cuda"
_N = 512
_TAU = 0.2


def _model(sh_degree: int, seed: int = 7) -> FourDGaussians:
    """A multiplicative-appearance model with all colour coefficients randomized."""
    torch.manual_seed(seed)
    model = FourDGaussians.from_random(_N, sh_degree=sh_degree, appearance_model=AppearanceType.SH4D_MULTIPLICATIVE).to(
        _DEVICE
    )
    for key in ("sh0", "shN", "sh_spacetime"):
        model.splats[key].data.normal_(0.0, 0.5)
    return model


def _fused_colors(
    model: FourDGaussians,
    sh_degree: int,
    viewmats: torch.Tensor,
    tau: float = _TAU,
    sh_degree_t: int = MAX_TEMPORAL_DEGREE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Colours the fused EWA kernel computes, and which (camera, Gaussian) pairs it wrote."""
    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel

    appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)
    splats = model.splats
    Ks = torch.tensor([[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]], device=_DEVICE)
    Ks = Ks.unsqueeze(0).expand(viewmats.shape[0], -1, -1).contiguous()

    _, _, colors, _, _, radii, _, _ = launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
        splats["quats_l"],
        splats["quats_r"],
        splats["scales_4d_log"],
        splats["means"],
        splats["mu_t"],
        splats["opacities"],
        appearance.sh0,
        appearance.shN,
        appearance.coeffs,
        sh_degree,
        appearance.kind_id,
        appearance.n_terms,
        appearance.stride,
        viewmats,
        Ks,
        torch.empty(0, device=Ks.device),
        0,
        tau,
        model.cov_eps,
        0.3,
        0.01,
        1e10,
        0.0,
        256,
        256,
        False,
    )
    return colors, (radii > 0).all(dim=-1)


def _reference_colors(
    model: FourDGaussians, sh_degree: int, deg_t: int, viewmats: torch.Tensor, tau: float = _TAU
) -> torch.Tensor:
    """The same colours from the oracle, at the conditioned means the kernel projects."""
    from tests.fused_stages import condition

    splats = model.splats
    means3d, _, _, _ = condition(
        splats["quats_l"],
        splats["quats_r"],
        splats["scales_4d_log"],
        splats["means"],
        splats["mu_t"],
        splats["opacities"],
        tau,
        model.cov_eps,
        0.0,
    )

    # The conditioned mean is the raw mean plus a correction independent of it, so its Jacobian is the identity.
    means3d = splats["means"] + (means3d - splats["means"]).detach()

    rotations = viewmats[:, :3, :3]
    translations = viewmats[:, :3, 3]
    campos = -(rotations.transpose(-2, -1) @ translations.unsqueeze(-1)).squeeze(-1)
    dirs = torch.nn.functional.normalize(means3d[None, :, :] - campos[:, None, :], dim=-1)
    dirs_t = tau - splats["mu_t"]

    sh = torch.cat([splats["sh0"], splats["shN"], splats["sh_spacetime"]], dim=1)
    k_max = (sh_degree + 1) ** 2
    colors = [eval_shfs_4d(sh_degree, deg_t, sh, dirs[c], dirs_t, k_max) for c in range(viewmats.shape[0])]
    return torch.clamp(torch.stack(colors, dim=0) + 0.5, min=0.0)


def _viewmats(n_cameras: int = 3) -> torch.Tensor:
    """Cameras on a ring around the origin, looking inward."""
    mats = []
    for c in range(n_cameras):
        angle = 2 * torch.pi * c / n_cameras
        eye = torch.tensor([4.0 * torch.cos(torch.tensor(angle)), 1.0, 4.0 * torch.sin(torch.tensor(angle))])
        forward = torch.nn.functional.normalize(-eye, dim=0)
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0])), dim=0)
        up = torch.linalg.cross(right, forward)
        rotation = torch.stack([right, up, forward], dim=0)
        mat = torch.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = -rotation @ eye
        mats.append(mat)
    return torch.stack(mats, dim=0).to(_DEVICE)


@requires_cuda
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
@pytest.mark.parametrize("sh_degree_t", [0, 1, 2])
def test_forward_matches_the_published_basis(sh_degree: int, sh_degree_t: int) -> None:
    """Colour is the product basis of eval_shfs_4d, not a sum of independent terms.

    Both degrees are swept: a run spends its first steps at a temporal degree below the maximum,
    where the kernel stops the block loop early.
    """
    model = _model(sh_degree)
    viewmats = _viewmats()

    colors, visible = _fused_colors(model, sh_degree, viewmats, sh_degree_t=sh_degree_t)
    expected = _reference_colors(model, sh_degree, sh_degree_t, viewmats)

    assert visible.any(), "nothing projected: the fixture would prove nothing"
    torch.testing.assert_close(colors[visible], expected[visible], rtol=0, atol=1e-5)


@requires_cuda
def test_a_temporal_block_changes_view_dependence() -> None:
    """What separates this basis from the additive one: the same Gaussian's view dependence moves with tau."""
    model = _model(sh_degree=3)
    viewmats = _viewmats()

    # An additive temporal term shifts every camera by the same amount; a product basis does not.
    late, visible = _fused_colors(model, 3, viewmats, tau=0.3)
    early, _ = _fused_colors(model, 3, viewmats, tau=-0.3)
    delta = torch.where(visible.unsqueeze(-1), late - early, torch.zeros_like(late))
    spread = (delta - delta.mean(dim=0, keepdim=True)).abs().max()

    assert spread > 1e-3


@requires_cuda
@pytest.mark.parametrize("sh_degree_t", [0, 1, 2])
def test_backward_matches_autograd_through_the_oracle(sh_degree_t: int) -> None:
    """Every colour parameter's gradient and the direction path, against autograd through the oracle.

    ``means`` is the one that catches a mis-weighted temporal block in the direction Jacobian: the
    coefficient gradients would still look right.
    """
    from anycam4d.cuda.autograd import EWAFusedFunction

    model = _model(sh_degree=3, seed=11)
    viewmats = _viewmats()
    splats = model.splats
    for key in ("sh0", "shN", "sh_spacetime", "means"):
        splats[key].requires_grad_(True)

    appearance = model.appearance_inputs(sh_degree_t=sh_degree_t)
    Ks = torch.tensor([[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]], device=_DEVICE)
    Ks = Ks.unsqueeze(0).expand(viewmats.shape[0], -1, -1).contiguous()

    coeffs = splats["sh_spacetime"]
    _, _, colors, _, _, radii, _, _ = EWAFusedFunction.apply(
        splats["quats_l"].detach(),
        splats["quats_r"].detach(),
        splats["scales_4d_log"].detach(),
        splats["means"],
        splats["mu_t"].detach(),
        splats["opacities"].detach(),
        splats["sh0"],
        splats["shN"],
        coeffs,
        3,
        appearance.kind_id,
        appearance.n_terms,
        appearance.stride,
        viewmats,
        Ks,
        torch.empty(0, device=Ks.device),
        0,
        _TAU,
        model.cov_eps,
        0.3,
        0.01,
        1e10,
        0.0,
        256,
        256,
        False,
    )
    visible = (radii > 0).all(dim=-1)
    colors[visible].sum().backward()
    compared = ("sh0", "shN", "sh_spacetime", "means")
    fused_grads = tuple(splats[key].grad.clone() for key in compared)

    for key in compared:
        splats[key].grad = None
    expected = _reference_colors(model, 3, sh_degree_t, viewmats)
    expected[visible].sum().backward()

    for fused, name in zip(fused_grads, compared, strict=True):
        torch.testing.assert_close(fused, splats[name].grad, rtol=1e-4, atol=1e-5)


def test_storage_is_a_block_per_temporal_order() -> None:
    """The parameter is the published layout: one spatial-SH block per temporal order."""
    model = FourDGaussians.from_random(4, sh_degree=3, appearance_model=AppearanceType.SH4D_MULTIPLICATIVE)

    assert model.splats["sh_spacetime"].shape == (4, MAX_TEMPORAL_DEGREE * 16, 3)
    assert "sh_temporal" not in model.splats


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, (0, 0)),
        (999, (0, 0)),
        (1000, (1, 0)),
        (3000, (3, 0)),
        (3999, (3, 0)),
        (4000, (3, 1)),
        (5000, (3, 2)),
        (30_000, (3, 2)),
    ],
)
def test_the_degree_curriculum_saturates_space_before_time(step: int, expected: tuple[int, int]) -> None:
    """Published 4DGS raises the spatial degree to its maximum first, then the temporal one."""
    assert active_degrees(step, interval=1000, max_sh_degree=3) == expected


def test_checkpoint_round_trip_keeps_the_basis() -> None:
    """The stored appearance value is what decides how sh_spacetime is read back."""
    model = FourDGaussians.from_random(6, sh_degree=3, appearance_model=AppearanceType.SH4D_MULTIPLICATIVE)
    model.splats["sh_spacetime"].data.normal_(0.0, 0.5)
    state = model.state_dict_with_meta(norm_scale=1.0, norm_center=torch.zeros(3), t_min=0.0, t_max=1.0)

    loaded = FourDGaussians.from_checkpoint(state, source="round-trip.pt")

    assert loaded.appearance_model is AppearanceType.SH4D_MULTIPLICATIVE
    torch.testing.assert_close(loaded.splats["sh_spacetime"], model.splats["sh_spacetime"], rtol=0, atol=0)


def test_a_wrong_width_parameter_is_refused() -> None:
    """The kernel splits the stride into MAX_TEMPORAL_DEGREE blocks; another width reads wrong offsets."""
    model = FourDGaussians.from_random(4, sh_degree=3, appearance_model=AppearanceType.SH4D_MULTIPLICATIVE)
    splats = model.splats
    splats["sh_spacetime"] = torch.nn.Parameter(torch.zeros(4, 3 * 16, 3))

    with pytest.raises(ValueError, match="sh_spacetime has 48 rows"):
        FourDGaussians(splats, sh_degree=3, appearance_model=AppearanceType.SH4D_MULTIPLICATIVE)
