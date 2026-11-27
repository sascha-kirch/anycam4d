"""Rolling-shutter render: per-row equality with the global-shutter UT render, and the per-row pose."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.valid_mask import pack_validity
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from tests.fused_stages import fused_ut_project
from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as TILE
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from tests.oracles.factor import cov_of


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="rolling shutter runs on the CUDA UT kernels")

if HAS_CUDA_EXT:
    from anycam4d.renderer.rolling_shutter import project_rolling_shutter
    from anycam4d.renderer.rolling_shutter import render_rolling_shutter
    from anycam4d.renderer.rolling_shutter import row_viewmats
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend
    from tests.test_ut_projection import _ray_d2

W, H = 64, 48


def _moving_model(device: torch.device, appearance: str, sh_degree: int, n_temporal_freqs: int = 4) -> FourDGaussians:
    """Random 4D rotations, so every Gaussian has a velocity and a finite temporal extent."""
    torch.manual_seed(0)
    model = FourDGaussians.from_random(
        n_points=3000, sh_degree=sh_degree, appearance_model=appearance, n_temporal_freqs=n_temporal_freqs
    )
    n = model.splats["means"].shape[0]
    with torch.no_grad():
        model.splats["means"].mul_(0.6)
        model.splats["mu_t"].uniform_(-0.15, 0.15)
        model.splats["quats_l"].copy_(torch.randn(n, 4))
        model.splats["quats_r"].copy_(torch.randn(n, 4))
        model.splats["scales_4d_log"].copy_(torch.empty(n, 4).uniform_(-3.5, -2.0))
        model.splats["opacities"].fill_(1.0)
        model.splats["sh0"].copy_(torch.rand(n, 1, 3))
        model.splats["shN"].copy_(0.2 * torch.randn_like(model.splats["shN"]))
        for key in ("sh_temporal", "sh_spacetime"):
            if key in model.splats:
                model.splats[key].copy_(0.2 * torch.randn_like(model.splats[key]))
    return model.to(device)


def _camera(device: torch.device, model: CameraModelType, viewmat: torch.Tensor) -> BatchedCamera:
    k = torch.tensor([[[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]]], device=device)
    if model == CameraModelType.FISHEYE_OPENCV:
        dist = torch.tensor([[0.05, -0.01, 0.002, -0.0003]], device=device)
    else:
        dist = torch.zeros(1, 12, device=device)
    return BatchedCamera(viewmat.unsqueeze(0), k, model, W, H, dist)


def _look(device: torch.device, x: float, yaw: float, z: float = -3.0) -> torch.Tensor:
    """World-to-camera with centre (x, 0, z) and a yaw about the camera's y axis."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], device=device)
    vm = torch.eye(4, device=device)
    vm[:3, :3] = rot
    vm[:3, 3] = -rot @ torch.tensor([x, 0.0, z], device=device)
    return vm


@pytest.mark.parametrize("camera_model", [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV])
def test_zero_readout_is_global_shutter(camera_model: CameraModelType) -> None:
    """No readout and one pose per row reproduces the global-shutter render, with 4D colour and the validity gate."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=3)
    vm = _look(device, 0.0, 0.0)
    camera = _camera(device, camera_model, vm)
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    disk = (xx + 0.5 - W / 2) ** 2 + (yy + 0.5 - H / 2) ** 2 < (0.45 * W) ** 2
    camera = dataclasses.replace(camera, valid_mask=pack_validity(disk)[None].contiguous())
    cfg = RenderingConfig(backend=BackendType.ANYCAM4D)
    with torch.no_grad():
        ref = UTFusedBackend(cfg).render(model, camera, 0.07, 3)
        rows = vm.expand(1, H, 4, 4).contiguous()
        colors, alphas = render_rolling_shutter(model, camera, rows, 0.07, 0.0, 3, cfg)
    assert ref.alphas.max() > 0.5
    assert alphas[0, ~disk].max() == 0.0
    assert (colors - ref.colors).abs().max() <= 1e-6
    assert (alphas - ref.alphas).abs().max() <= 1e-6


@pytest.mark.parametrize("camera_model", [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV])
@pytest.mark.parametrize("ego_motion", [False, True])
def test_each_row_is_the_global_render_at_its_time_and_pose(camera_model: CameraModelType, ego_motion: bool) -> None:
    """Row r equals row r of the global-shutter render at tau_r = tau + ((r + 0.5)/H - 0.5) readout and pose r.

    One colour for every Gaussian: colour then reads alpha alone, which does not depend on blend order. The blend
    order (one sort key per Gaussian, not per row) and the colour held at tau are the two approximations by design.
    """
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=0, n_temporal_freqs=0)
    with torch.no_grad():
        model.splats["sh0"].fill_(0.7)
    tau, readout = 0.0, 0.3
    start = _look(device, -0.3, -0.1)
    end = _look(device, 0.3, 0.1) if ego_motion else start
    camera = _camera(device, camera_model, start)
    rows = row_viewmats(start, end, H)
    cfg = RenderingConfig(backend=BackendType.ANYCAM4D)
    backend = UTFusedBackend(cfg)

    with torch.no_grad():
        colors, alphas = render_rolling_shutter(model, camera, rows.unsqueeze(0), tau, readout, 0, cfg)
        ref_colors = torch.empty_like(colors)
        ref_alphas = torch.empty_like(alphas)
        for r in range(H):
            tau_r = tau + ((r + 0.5) / H - 0.5) * readout
            out = backend.render(model, _camera(device, camera_model, rows[r]), tau_r, 0)
            ref_colors[:, r] = out.colors[:, r]
            ref_alphas[:, r] = out.alphas[:, r]

    # fp32: the row mean is mean(tau) + v dtau, not a fresh Schur at tau_r; one ulp of tau alone moves alpha ~7e-5.
    assert ref_alphas.max() > 0.5
    assert (colors - ref_colors).abs().max() <= 5e-4
    assert (alphas - ref_alphas).abs().max() <= 5e-4


def test_rolling_differs_from_global_under_motion() -> None:
    """Guard on the equality test above: with this readout the rolling and global images really differ."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=0, n_temporal_freqs=0)
    vm = _look(device, 0.0, 0.0)
    camera = _camera(device, CameraModelType.PINHOLE, vm)
    cfg = RenderingConfig(backend=BackendType.ANYCAM4D)
    with torch.no_grad():
        ref = UTFusedBackend(cfg).render(model, camera, 0.0, 0)
        _, alphas = render_rolling_shutter(model, camera, vm.expand(1, H, 4, 4).contiguous(), 0.0, 0.3, 0, cfg)
    assert (alphas - ref.alphas).abs().max() > 0.2


def _isect_tiles(
    means2d: torch.Tensor, radii: torch.Tensor, n_tiles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiles [lo, hi) per axis that gsplat's isect_tiles assigns each box."""
    t, tr = means2d / TILE, radii.float() / TILE
    return (t - tr).floor().long().clamp(min=0).minimum(n_tiles), (t + tr).ceil().long().clamp(min=0).minimum(n_tiles)


def test_tile_box_holds_every_pixel_each_row_draws() -> None:
    """Pinhole under ego- and scene motion: every pixel a row draws lies in the tile box, which is exactly the union
    of the tiles the global UT projection assigns on each row.

    Drawn is the rasterizer's own test in fp64 at the row's time and pose: opacity·exp(-D²/2) ≥ 1/255 and
    exp(-D²/2) > 0.0113, for Gaussians whose mean lies in [near, far] there (the global projection's cull). The camera
    dollies through the near plane of part of the population during the readout.
    """
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=0, n_temporal_freqs=0)
    tau, readout = 0.0, 0.3
    start, end = _look(device, -0.3, -0.15), _look(device, 0.3, 0.15, z=-2.4)
    rows = row_viewmats(start, end, H)
    cfg = dataclasses.replace(RenderingConfig(backend=BackendType.ANYCAM4D), near_plane=2.6)
    with torch.no_grad():
        camera = _camera(device, CameraModelType.PINHOLE, start)
        proj = project_rolling_shutter(model, camera, rows[None], tau, readout, 0, cfg)

    r_idx = torch.arange(H, device=device, dtype=torch.float64)
    dtau = proj.tau_first + r_idx * proj.tau_step - tau  # [H]
    mean = proj.means.double()[:, None] + proj.motion.double()[:, None, :3] * dtau[None, :, None]  # [N, H, 3]
    dt = (tau + dtau)[None] - proj.motion.double()[:, 3:4]
    gate = torch.exp(-0.5 * dt**2 * proj.inv_var_t.double()[:, None])
    opac = torch.where(gate >= model.prefilter_threshold, proj.base.double()[:, None] * gate, 0.0)  # [N, H]
    rot, trans = rows[:, :3, :3].double(), rows[:, :3, 3].double()
    depth = torch.einsum("hj,nhj->nh", rot[:, 2], mean) + trans[None, :, 2]
    kept = (depth >= cfg.near_plane) & (depth <= cfg.far_plane)
    assert (kept.any(-1) & ~kept.all(-1) & (opac > 0.1).all(-1)).sum() >= 20, "population must cross the near plane"

    sigma = cov_of(proj.cov_factors.double())
    lam = torch.linalg.inv(sigma.view(-1, 3, 3))
    xs = torch.arange(W, device=device, dtype=torch.float64) + 0.5
    u = ((xs - W / 2) / 60.0)[None].expand(H, W)
    v = ((r_idx + 0.5 - H / 2) / 60.0)[:, None].expand(H, W)
    d_cam = torch.stack([u, v, torch.ones_like(u)], -1)
    d = torch.einsum("hji,hwj->hwi", rot, d_cam)  # world ray directions [H, W, 3]
    p = -torch.einsum("hji,hj->hi", rot, trans)[None] - mean  # camera centre - mean [N, H, 3]
    lam_p = torch.einsum("nij,nhj->nhi", lam, p)
    dld = torch.einsum("hwi,nij,hwj->nhw", d, lam, d)
    d2 = _ray_d2(torch.einsum("hwi,nhi->nhw", d, lam_p), (p * lam_p).sum(-1)[..., None], dld)
    response = torch.exp(-0.5 * d2)
    drawn = (opac[..., None] * response >= 1.0 / 255.0) & (response > 0.0113) & kept[..., None]  # [N, H, W]

    radii, means2d = proj.radii[0].double(), proj.means2d[0].double()
    ys = (r_idx + 0.5)[:, None].expand(H, W)
    for i in torch.nonzero(drawn.flatten(1).any(-1)).squeeze(-1).tolist():
        assert radii[i].min() > 0, f"Gaussian {i} draws pixels but was culled"
        x, y = xs.expand(H, W)[drawn[i]], ys[drawn[i]]
        assert ((x - means2d[i, 0]).abs() <= radii[i, 0]).all() and ((y - means2d[i, 1]).abs() <= radii[i, 1]).all(), (
            f"Gaussian {i}: a drawn pixel lies outside the tile box"
        )

    # Tight: the box is the union, over rows, of the tiles the global UT projection assigns on that row's tile row.
    n_tx, n_ty = math.ceil(W / TILE), math.ceil(H / TILE)
    lo = torch.full((proj.means.shape[0], 2), 1 << 30, device=device)
    hi = torch.full((proj.means.shape[0], 2), -1, device=device)
    s = model.splats
    for r in range(H):
        g_radii, g_means2d, _, _ = fused_ut_project(
            s["quats_l"],
            s["quats_r"],
            s["scales_4d_log"],
            s["means"],
            s["mu_t"],
            s["opacities"],
            rows[r][None],
            camera.Ks,
            camera.distortion,
            tau=proj.tau_first + r * proj.tau_step,
            cov_eps=model.cov_eps,
            eps2d=cfg.eps2d,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            prefilter_threshold=model.prefilter_threshold,
            image_width=W,
            image_height=H,
            family=int(ProjectionFamily.PINHOLE_FAMILY),
        )
        t_lo, t_hi = _isect_tiles(g_means2d[0], g_radii[0], torch.tensor([n_tx, n_ty], device=device))
        on_row = (
            (g_radii[0] > 0).all(-1) & (t_lo[:, 1] <= r // TILE) & (r // TILE < t_hi[:, 1]) & (t_lo[:, 0] < t_hi[:, 0])
        )
        row_lo = torch.stack([t_lo[:, 0], torch.full_like(t_lo[:, 0], r // TILE)], -1)
        lo = torch.where(on_row[:, None], torch.minimum(lo, row_lo), lo)
        hi = torch.where(on_row[:, None], torch.maximum(hi, torch.stack([t_hi[:, 0], row_lo[:, 1] + 1], -1)), hi)
    expect_drawn = (hi >= 0).all(-1)
    rs_lo, rs_hi = _isect_tiles(proj.means2d[0], proj.radii[0], torch.tensor([n_tx, n_ty], device=device))
    rs_drawn = (proj.radii[0] > 0).all(-1)
    mismatch = (rs_drawn != expect_drawn) | (expect_drawn & ((rs_lo != lo) | (rs_hi != hi)).any(-1))
    assert expect_drawn.sum() > 1000
    assert mismatch.sum() <= 3, f"{mismatch.sum().item()} Gaussians' tile boxes differ from the per-row global union"


def test_rejects_a_non_ut_render_config() -> None:
    """The rolling-shutter kernels are UT ray kernels: an EWA or antialiased config is rejected, not ignored."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=0, n_temporal_freqs=0)
    vm = _look(device, 0.0, 0.0)
    camera = _camera(device, CameraModelType.PINHOLE, vm)
    with pytest.raises(ValueError, match="UT"):
        render_rolling_shutter(model, camera, vm.expand(1, H, 4, 4).contiguous(), 0.0, 0.0, 0, RenderingConfig())


def test_row_viewmats_translate_and_rotate_linearly() -> None:
    """Row pose at f = (r + 0.5)/H: camera centre lerped, yaw angle linear in f."""
    device = torch.device("cuda")
    rows = row_viewmats(_look(device, -1.0, -0.2), _look(device, 1.0, 0.4), H)
    for r in (0, H // 2, H - 1):
        f = (r + 0.5) / H
        expect = _look(device, -1.0 + 2.0 * f, -0.2 + 0.6 * f)
        torch.testing.assert_close(rows[r], expect, atol=1e-5, rtol=0.0)
