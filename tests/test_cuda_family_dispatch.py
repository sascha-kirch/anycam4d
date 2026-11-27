"""Projection-family dispatch: device project/unproject per family against torch, and the out-of-range abort."""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="family dispatch runs on the CUDA UT kernels")

_PINHOLE = int(ProjectionFamily.PINHOLE_FAMILY)
_FISHEYE = int(ProjectionFamily.FISHEYE_FAMILY)


def _project_ref(
    family: int, cx: float, cy: float, cz: float, fx: float, fy: float, ppx: float, ppy: float, dist: list[float]
) -> tuple[float, float]:
    """Pure-torch reference projection (camera-space xyz → pixel) per family."""
    if family == _FISHEYE:
        k1, k2, k3, k4 = dist[0], dist[1], dist[2], dist[3]
        r = math.sqrt(cx * cx + cy * cy)
        theta = math.atan2(r, cz)
        t2 = theta * theta
        theta_d = theta * (1.0 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)
        scale = (theta_d / r) if r > 1e-6 else 1.0
        return fx * scale * cx + ppx, fy * scale * cy + ppy
    # PINHOLE_FAMILY: OpenCV Brown–Conrady forward (12-slot canonical layout). Reproduces
    # CameraOps<PINHOLE_FAMILY>::project. Zero dist degenerates to plain perspective.
    return _brown_conrady_project(cx, cy, cz, fx, fy, ppx, ppy, dist)


def _brown_conrady_project(
    cx: float, cy: float, cz: float, fx: float, fy: float, ppx: float, ppy: float, d: list[float]
) -> tuple[float, float]:
    """OpenCV Brown–Conrady forward in the 12-slot [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4] layout."""
    u, v = cx / cz, cy / cz
    r2 = u * u + v * v
    num = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))  # k1,k2,k3
    den = 1.0 + r2 * (d[5] + r2 * (d[6] + r2 * d[7]))  # k4,k5,k6
    icd = num / den
    a1, a2, a3 = 2.0 * u * v, r2 + 2.0 * u * u, r2 + 2.0 * v * v
    dx = d[2] * a1 + d[3] * a2 + r2 * (d[8] + r2 * d[9])  # p1,p2,s1,s2
    dy = d[2] * a3 + d[3] * a1 + r2 * (d[10] + r2 * d[11])  # p1,p2,s3,s4
    return fx * (icd * u + dx) + ppx, fy * (icd * v + dy) + ppy


def _single_gaussian_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """One near-point Gaussian (tiny covariance) so the UT mean ≈ exact projection."""
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    return {
        "quats_l": quat.clone(),
        "quats_r": quat.clone(),
        "scales_4d_log": torch.full((1, 4), -5.0, device=device),  # exp(-5) ≈ 0.0067 → tiny
        "means": torch.tensor([[0.3, 0.2, 0.0]], device=device),
        "mu_t": torch.zeros(1, device=device),
        "opacities": torch.full((1,), 10.0, device=device),
    }


@pytest.mark.parametrize("family", [_PINHOLE, _FISHEYE])
def test_cuda_project_matches_torch_ref(family: int) -> None:
    """Device CameraOps<F>::project (via the UT projection kernel) vs a torch reference."""
    from tests.fused_stages import fused_ut_project

    device = torch.device("cuda")
    g = _single_gaussian_inputs(device)
    fx = fy = 400.0
    w, h = 256, 256
    ppx, ppy = w / 2.0, h / 2.0
    zc = 5.0
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = zc
    viewmat = viewmat.unsqueeze(0)
    k = torch.tensor([[[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]]], device=device)

    n_slots = 4 if family == _FISHEYE else 12
    dist_vals = [0.05, -0.01, 0.002, -0.0003] + [0.0] * (n_slots - 4)
    rc = torch.tensor([dist_vals[:n_slots]], device=device)

    radii, means2d, _depths, _conics = fused_ut_project(
        g["quats_l"],
        g["quats_r"],
        g["scales_4d_log"],
        g["means"],
        g["mu_t"],
        g["opacities"],
        viewmat,
        k,
        rc,
        tau=0.0,
        cov_eps=1e-6,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=w,
        image_height=h,
        family=family,
    )

    assert radii[0, 0, 0] > 0, "Gaussian should be visible"
    # camera-space center = means + viewmat translation (identity rotation)
    cx_cam, cy_cam, cz_cam = 0.3, 0.2, zc
    u_ref, v_ref = _project_ref(family, cx_cam, cy_cam, cz_cam, fx, fy, ppx, ppy, dist_vals)
    assert means2d[0, 0, 0].item() == pytest.approx(u_ref, abs=0.05)
    assert means2d[0, 0, 1].item() == pytest.approx(v_ref, abs=0.05)


def _axis_model(device: torch.device) -> FourDGaussians:
    """A single opaque, near-isotropic Gaussian fixed at the world origin."""
    model = FourDGaussians.from_random(n_points=1, sh_degree=0).to(device)
    with torch.no_grad():
        model.splats["means"].copy_(torch.zeros(1, 3, device=device))
        model.splats["scales_4d_log"].copy_(torch.full((1, 4), -3.0, device=device))
        model.splats["quats_l"].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
        model.splats["quats_r"].copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
        model.splats["opacities"].copy_(torch.full((1,), 12.0, device=device))
        model.splats["mu_t"].copy_(torch.zeros(1, device=device))
    return model


@pytest.mark.parametrize("camera_model", [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV])
def test_cuda_world_ray_matches_torch_ref(camera_model: CameraModelType) -> None:
    """A Gaussian on the optical axis peaks at the principal point for both families."""
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend

    device = torch.device("cuda")
    w = h = 64
    model = _axis_model(device)
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 4.0  # camera 4 units in front of the origin Gaussian
    viewmat = viewmat.unsqueeze(0)
    ppx, ppy = w / 2.0, h / 2.0
    k = torch.tensor([[[200.0, 0.0, ppx], [0.0, 200.0, ppy], [0.0, 0.0, 1.0]]], device=device)
    distortion = (
        torch.tensor([[0.05, -0.01, 0.002, -0.0003]], device=device)
        if camera_model == CameraModelType.FISHEYE_OPENCV
        else torch.zeros(1, 12, device=device)
    )

    backend = UTFusedBackend(RenderingConfig())
    camera = BatchedCamera(viewmat, k, camera_model, w, h, distortion)
    with torch.no_grad():
        out = backend.render(model, camera, 0.0, model.sh_degree)

    alpha = out.alphas[0, :, :, 0]
    assert torch.isfinite(alpha).all()
    peak = torch.argmax(alpha)
    py, px = int(peak // w), int(peak % w)
    # On-axis Gaussian → peak within one pixel of the principal point for both families.
    assert abs(px - ppx) <= 1.5 and abs(py - ppy) <= 1.5


def test_host_switch_default_raises() -> None:
    """An out-of-range family int aborts via the exhaustive `default: TORCH_CHECK(false)`."""
    from tests.fused_stages import fused_ut_project

    device = torch.device("cuda")
    g = _single_gaussian_inputs(device)
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 5.0
    viewmat = viewmat.unsqueeze(0)
    k = torch.tensor([[[400.0, 0.0, 128.0], [0.0, 400.0, 128.0], [0.0, 0.0, 1.0]]], device=device)
    rc = torch.zeros(1, 12, device=device)  # 12-slot → passes the family-keyed stride check

    with pytest.raises(RuntimeError):
        fused_ut_project(
            g["quats_l"],
            g["quats_r"],
            g["scales_4d_log"],
            g["means"],
            g["mu_t"],
            g["opacities"],
            viewmat,
            k,
            rc,
            tau=0.0,
            cov_eps=1e-6,
            eps2d=0.3,
            near_plane=0.01,
            far_plane=1e10,
            prefilter_threshold=0.0,
            image_width=256,
            image_height=256,
            family=99,
        )
        torch.cuda.synchronize()
