"""Thin-lens depth of field: aperture samples against the moved pinhole, the single pass against the aperture average."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from tests.oracles.factor import cov_of


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="depth of field runs on the CUDA UT kernels")

if HAS_CUDA_EXT:
    from anycam4d.cuda.ext import ut_4d_depth_of_field_projection_fwd
    from anycam4d.renderer.depth_of_field import render_depth_of_field
    from anycam4d.renderer.depth_of_field import pass_offsets
    from anycam4d.renderer.depth_of_field import render_thin_lens
    from anycam4d.renderer.depth_of_field import vogel_disk
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend
    from tests.test_rolling_shutter import _camera
    from tests.test_rolling_shutter import _look
    from tests.test_rolling_shutter import _moving_model
    from tests.test_ut_projection import _ray_d2
    from tests.test_ut_projection import _unit_rays

CFG = RenderingConfig(backend=BackendType.ANYCAM4D)


@pytest.mark.parametrize("camera_model", [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV])
def test_zero_aperture_is_global_render(camera_model: CameraModelType) -> None:
    """No aperture and no offset reproduce the global UT render, 4D colour included."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=3)
    camera = _camera(device, camera_model, _look(device, 0.0, 0.0))
    with torch.no_grad():
        ref = UTFusedBackend(CFG).render(model, camera, 0.07, 3)
        colors, alphas = render_thin_lens(model, camera, 0.07, 3, CFG, focus=3.0)
    assert ref.alphas.max() > 0.5
    assert (colors - ref.colors).abs().max() <= 1e-6
    assert (alphas - ref.alphas).abs().max() <= 1e-6


def test_aperture_sample_is_moved_pinhole_camera() -> None:
    """One aperture sample equals the existing UT render of the camera moved by o with principal point + f o / d."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=3)
    viewmat = _look(device, 0.0, 0.0)
    focus, offset = 2.5, (0.12, -0.07)
    moved = viewmat.clone()
    moved[0, 3] -= offset[0]
    moved[1, 3] -= offset[1]
    moved_camera = _camera(device, CameraModelType.PINHOLE, moved)
    Ks = moved_camera.Ks.clone()
    Ks[0, :2, 2] += Ks[0, [0, 1], [0, 1]] * torch.tensor(offset, device=device) / focus
    moved_camera = dataclasses.replace(moved_camera, Ks=Ks)
    with torch.no_grad():
        ref = UTFusedBackend(CFG).render(model, moved_camera, 0.07, 3)
        colors, alphas = render_thin_lens(
            model, _camera(device, CameraModelType.PINHOLE, viewmat), 0.07, 3, CFG, focus=focus, offset=offset
        )
    assert ref.alphas.max() > 0.5
    assert (colors - ref.colors).abs().max() <= 1e-4
    assert (alphas - ref.alphas).abs().max() <= 1e-4


def _one_gaussian(device: torch.device, position: tuple[float, float, float]) -> FourDGaussians:
    """One static isotropic Gaussian of opacity 0.9 and white colour, σ = 0.01·z (about 2 px sharp at f = 205)."""
    model = FourDGaussians.from_random(n_points=1, sh_degree=0, appearance_model="sh4d_additive")
    with torch.no_grad():
        model.splats["means"].copy_(torch.tensor([position]))
        model.splats["mu_t"].zero_()
        model.splats["quats_l"].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.splats["quats_r"].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model.splats["scales_4d_log"].copy_(torch.tensor([[math.log(0.01 * position[2])] * 3 + [3.0]]))
        model.splats["opacities"].fill_(math.log(0.9 / 0.1))
        model.splats["sh0"].fill_(1.0)
    return model.to(device)


def _moments(alpha: torch.Tensor) -> tuple[float, float, float]:
    """Integrated alpha and its second central moments in x and y (px²)."""
    a = alpha[0, ..., 0].double()
    ys, xs = torch.meshgrid(
        torch.arange(a.shape[0], device=a.device) + 0.5, torch.arange(a.shape[1], device=a.device) + 0.5, indexing="ij"
    )
    energy = a.sum()
    mx, my = (a * xs).sum() / energy, (a * ys).sum() / energy
    return energy.item(), ((a * (xs - mx) ** 2).sum() / energy).item(), ((a * (ys - my) ** 2).sum() / energy).item()


@pytest.mark.parametrize("samples", [1, 8])
@pytest.mark.parametrize("camera_model", [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV])
@pytest.mark.parametrize(
    "position",
    [(0.0, 0.0, 1.2), (0.3, -0.2, 3.5), (0.0, 0.0, 6.0), (0.0, 0.0, 2.0), (0.45, 0.4, 1.0)],
    ids=["near", "far-off-axis", "very-far", "in-focus", "near-oblique"],
)
def test_blurred_render_matches_aperture_average(camera_model: CameraModelType, position: tuple, samples: int) -> None:
    """An isolated Gaussian blurred in one pass, or in 8 partly blurred ones, carries the energy and second moment of
    1024 averaged unblurred aperture samples.

    Incidence stays ≤ 45°: past ~60° on a wide fisheye the radial moment overestimates the edge-on disk. The sharp core
    is ~2 px so the blurred peak stays well above the 1/255 cutoff, whose tail loss on a sub-pixel isolated Gaussian is a
    known limit.
    """
    device = torch.device("cuda")
    size, focus, aperture = 256, 2.0, 0.08
    K = torch.tensor([[[0.8 * size, 0.0, size / 2], [0.0, 0.8 * size, size / 2], [0.0, 0.0, 1.0]]], device=device)
    dist = torch.zeros(1, 12 if camera_model == CameraModelType.PINHOLE else 4, device=device)
    if camera_model == CameraModelType.FISHEYE_OPENCV:
        dist[0, :2] = torch.tensor([0.02, -0.003])
    camera = BatchedCamera(torch.eye(4, device=device)[None], K, camera_model, size, size, dist)
    model = _one_gaussian(device, position)
    with torch.no_grad():
        ref = (
            sum(
                render_thin_lens(model, camera, 0.0, 0, CFG, focus=focus, offset=o)[1]
                for o in vogel_disk(aperture, 1024)
            )
            / 1024
        )
        blurred = render_depth_of_field(model, camera, 0.0, 0, CFG, focus=focus, aperture=aperture, samples=samples)[1]
    energy_ref, var_x_ref, var_y_ref = _moments(ref)
    energy, var_x, var_y = _moments(blurred)
    assert abs(energy / energy_ref - 1.0) <= 0.02
    assert abs(var_x / var_x_ref - 1.0) <= 0.05
    assert abs(var_y / var_y_ref - 1.0) <= 0.05


def test_blurred_box_holds_every_drawn_pixel() -> None:
    """On a pinhole every pixel the rasterizer draws for a dilated Gaussian lies in its tile box (invariant 4).

    Distorted lenses use the shared UT box, which blur does not change; that box is tested in test_ut_projection.py.

    Drawn is the rasterizer's own test in fp64: opacity·exp(-D²/2) ≥ 1/255 and exp(-D²/2) > 0.0113. The population spans the focus plane and reaches the near plane.
    """
    torch.manual_seed(2)
    N, W, H, f = 600, 128, 96, 60.0
    device = torch.device("cuda")
    family = ProjectionFamily.PINHOLE_FAMILY
    quats_l = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    quats_r = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales_4d_log = torch.randn(N, 4, device=device) * 0.5 - 2.5
    scales_4d_log[:, 3] = 3.0
    z = torch.rand(N, device=device) * 4.0 + 0.05
    xy = (torch.rand(N, 2, device=device) * 2.0 - 1.0) * z[:, None] * 0.6
    means = torch.cat([xy, z[:, None]], dim=-1)
    zeros = torch.zeros(N, device=device)
    opacities = torch.rand(N, device=device) * 8.0 - 4.0
    sh0 = torch.rand(N, 1, 3, device=device)
    shN = torch.zeros(N, 0, 3, device=device)
    K = torch.tensor([[[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]]], device=device)
    dist_t = torch.zeros(1, 12, device=device)
    means2d, _, radii, _, means3d, factors, opac = ut_4d_depth_of_field_projection_fwd(
        quats_l, quats_r, scales_4d_log, means, zeros, opacities, sh0, shN, torch.empty(0, device=device),
        0, 0, 0, 0, torch.eye(4, device=device)[None], K, dist_t,
        1.5, 0.15, 0.0, 0.0, 0.0, 1e-4, 0.3, 0.01, 1e10, 0.0, W, H, int(family),
    )  # fmt: skip
    means2d = means2d[0].double()
    radii = radii[0].double()

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xs, ys = xs.reshape(-1).double(), ys.reshape(-1).double()
    d, _ = _unit_rays(xs, ys, f, W / 2, H / 2, [], fisheye=False)
    sigma = cov_of(factors.double())
    lam = torch.linalg.inv(sigma.view(N, 3, 3))
    p = -means3d.double()
    lam_p = (lam @ p[:, :, None]).squeeze(-1)
    d2 = _ray_d2((d @ lam_p.T).T, (p * lam_p).sum(-1)[:, None], torch.einsum("pi,nij,pj->np", d, lam, d))
    response = torch.exp(-0.5 * d2)
    drawn = (opac.double()[:, None] * response >= 1.0 / 255.0) & (response > 0.0113)
    drawn &= means3d[:, 2:3].double() > 0.01  # the projection culls behind the near plane

    px, py = xs + 0.5, ys + 0.5
    blurred = 0
    for i in torch.nonzero(drawn.any(-1)).squeeze(-1).tolist():
        x, y = px[drawn[i]], py[drawn[i]]
        assert radii[i].min() > 0, f"Gaussian {i} draws pixels but was culled"
        assert ((x - means2d[i, 0]).abs() <= radii[i, 0]).all() and ((y - means2d[i, 1]).abs() <= radii[i, 1]).all(), (
            f"Gaussian {i}: a drawn pixel lies outside the tile box"
        )
        blurred += int(abs(1.0 - means3d[i, 2].item() / 1.5) > 0.5)
    assert blurred >= 50, "population must exercise strongly blurred Gaussians"


@pytest.mark.parametrize("samples", [3, 4, 8, 16])
def test_pass_offsets_are_centred_and_isotropic(samples: int) -> None:
    """The K pass offsets carry covariance A²/4·(1 - 1/K)·I, so with each pass's blur they sum to the aperture's."""
    aperture = 0.3
    pts = torch.tensor(pass_offsets(aperture, samples), dtype=torch.float64)
    cov = pts.T @ pts / samples
    assert pts.mean(0).abs().max() < 1e-12
    assert torch.allclose(cov, torch.eye(2, dtype=torch.float64) * aperture**2 / 4 * (1 - 1 / samples), atol=1e-12)


def test_pass_offsets_reject_two_samples_and_collapse_without_aperture() -> None:
    """Two offsets are collinear (no isotropic blur); a zero aperture is one centred pass, never NaN."""
    with pytest.raises(ValueError, match="samples"):
        pass_offsets(0.3, 2)
    with pytest.raises(ValueError, match="samples"):
        pass_offsets(0.3, 0)
    assert pass_offsets(0.0, 8) == [(0.0, 0.0)]
