"""Brown-Conrady and Kannala-Brandt projection on the UT path against OpenCV, and model reductions."""

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


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="OpenCV distortion runs on the CUDA UT kernels")

_PINHOLE = int(ProjectionFamily.PINHOLE_FAMILY)
_FISHEYE = int(ProjectionFamily.FISHEYE_FAMILY)


def _brown_conrady_project(u: float, v: float, d: list[float]) -> tuple[float, float]:
    """Normalised distorted coords for normalised undistorted (u, v); 12-slot layout."""
    r2 = u * u + v * v
    num = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))
    den = 1.0 + r2 * (d[5] + r2 * (d[6] + r2 * d[7]))
    icd = num / den
    a1, a2, a3 = 2.0 * u * v, r2 + 2.0 * u * u, r2 + 2.0 * v * v
    dx = d[2] * a1 + d[3] * a2 + r2 * (d[8] + r2 * d[9])
    dy = d[2] * a3 + d[3] * a1 + r2 * (d[10] + r2 * d[11])
    return icd * u + dx, icd * v + dy


def _newton_undistort(xd: float, yd: float, d: list[float], n_iter: int = 5) -> tuple[float, float]:
    """Analytic Newton undistort mirroring CameraOps<PINHOLE_FAMILY>::unproject."""
    x, y = xd, yd
    eps = 1e-6
    for _ in range(n_iter):
        r = x * x + y * y
        r2 = r * r
        alpha = 1.0 + r * (d[0] + r * (d[1] + r * d[4]))
        beta = 1.0 + r * (d[5] + r * (d[6] + r * d[7]))
        dd = alpha / beta
        if dd <= 0.0:
            break
        fx_res = dd * x + 2.0 * d[2] * x * y + d[3] * (r + 2.0 * x * x) + d[8] * r + d[9] * r2 - xd
        fy_res = dd * y + 2.0 * d[3] * x * y + d[2] * (r + 2.0 * y * y) + d[10] * r + d[11] * r2 - yd
        alpha_r = d[0] + r * (2.0 * d[1] + r * 3.0 * d[4])
        beta_r = d[5] + r * (2.0 * d[6] + r * 3.0 * d[7])
        d_r = (alpha_r * beta - alpha * beta_r) / (beta * beta)
        d_x, d_y = 2.0 * x * d_r, 2.0 * y * d_r
        fx_x = dd + d_x * x + 2.0 * d[2] * y + 6.0 * d[3] * x + 2.0 * x * (d[8] + 2.0 * d[9] * r)
        fx_y = d_y * x + 2.0 * d[2] * x + 2.0 * d[3] * y + 2.0 * y * (d[8] + 2.0 * d[9] * r)
        fy_x = d_x * y + 2.0 * d[3] * y + 2.0 * d[2] * x + 2.0 * x * (d[10] + 2.0 * d[11] * r)
        fy_y = dd + d_y * y + 2.0 * d[3] * x + 6.0 * d[2] * y + 2.0 * y * (d[10] + 2.0 * d[11] * r)
        det = fx_y * fy_x - fx_x * fy_y
        if abs(det) < eps:
            break
        ddx = (fx_res * fy_y - fy_res * fx_y) / det
        ddy = (fy_res * fx_x - fx_res * fy_x) / det
        x += ddx
        y += ddy
        if abs(ddx) < eps and abs(ddy) < eps:
            break
    return x, y


def _tiny_gaussians(means: torch.Tensor) -> dict[str, torch.Tensor]:
    """N tiny opaque Gaussians at given camera-space means (so UT mean ≈ exact projection)."""
    n = means.shape[0]
    device = means.device
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4).contiguous()
    return {
        "quats_l": quat.clone(),
        "quats_r": quat.clone(),
        "scales_4d_log": torch.full((n, 4), -6.0, device=device),  # exp(-6) ≈ 0.0025 → tiny
        "means": means.contiguous(),
        "mu_t": torch.zeros(n, device=device),
        "opacities": torch.full((n,), 10.0, device=device),
    }


def _project_points(means: torch.Tensor, K: torch.Tensor, dist12: torch.Tensor, w: int, h: int):
    """Project camera-space points via the UT projection kernel (identity viewmat). Returns means2d, radii."""
    from tests.fused_stages import fused_ut_project

    device = means.device
    g = _tiny_gaussians(means)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    radii, means2d, _d, _c = fused_ut_project(
        g["quats_l"],
        g["quats_r"],
        g["scales_4d_log"],
        g["means"],
        g["mu_t"],
        g["opacities"],
        viewmat,
        K.unsqueeze(0),
        dist12.unsqueeze(0),
        tau=0.0,
        cov_eps=1e-6,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=w,
        image_height=h,
        family=_PINHOLE,
    )
    return means2d[0], radii[0]


def _project_points_fisheye(means: torch.Tensor, K: torch.Tensor, dist4: torch.Tensor, w: int, h: int):
    """Project camera-space points via the UT fisheye-family kernel. Returns means2d, radii."""
    from tests.fused_stages import fused_ut_project

    device = means.device
    g = _tiny_gaussians(means)
    viewmat = torch.eye(4, device=device).unsqueeze(0)
    radii, means2d, _d, _c = fused_ut_project(
        g["quats_l"],
        g["quats_r"],
        g["scales_4d_log"],
        g["means"],
        g["mu_t"],
        g["opacities"],
        viewmat,
        K.unsqueeze(0),
        dist4.unsqueeze(0),
        tau=0.0,
        cov_eps=1e-6,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=w,
        image_height=h,
        family=_FISHEYE,
    )
    return means2d[0], radii[0]


def _grid_points(device: torch.device, zc: float = 4.0) -> torch.Tensor:
    """A grid of camera-space points inside the valid cone (small normalised radius)."""
    xs = torch.linspace(-0.6, 0.6, 5)
    ys = torch.linspace(-0.6, 0.6, 5)
    pts = [[x.item(), y.item(), zc] for x in xs for y in ys]
    return torch.tensor(pts, device=device, dtype=torch.float32)


def test_opencv_distortion_vs_cv2() -> None:
    """Forward OPENCV projection matches cv2.projectPoints for classic 5-param distortion."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    device = torch.device("cuda")
    w, h = 512, 512
    fx = fy = 400.0
    ppx, ppy = w / 2.0, h / 2.0
    K = torch.tensor([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], device=device)

    k1, k2, p1, p2, k3 = 0.15, -0.05, 0.001, -0.002, 0.01
    dist12 = torch.zeros(12, device=device)
    dist12[:5] = torch.tensor([k1, k2, p1, p2, k3], device=device)

    pts = _grid_points(device)
    means2d, radii = _project_points(pts, K, dist12, w, h)

    obj = pts.cpu().numpy().astype(np.float64).reshape(-1, 1, 3)
    cv_dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
    cv_K = K.cpu().numpy().astype(np.float64)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), cv_K, cv_dist)
    img = torch.tensor(img.reshape(-1, 2), device=device, dtype=torch.float32)

    visible = radii[:, 0] > 0
    assert visible.any(), "no visible grid points"
    err = (means2d[visible] - img[visible]).abs().max().item()
    # UT-mean reconstruction + float32: sub-pixel tolerance.
    assert err < 0.2, f"OPENCV vs cv2.projectPoints max error {err:.4f}px"


def test_opencv_project_zero_dist_equals_pinhole() -> None:
    """OPENCV with all-zero distortion projects identically to PINHOLE."""
    device = torch.device("cuda")
    w, h = 256, 256
    K = torch.tensor([[300.0, 0.0, w / 2.0], [0.0, 300.0, h / 2.0], [0.0, 0.0, 1.0]], device=device)
    pts = _grid_points(device)
    zeros = torch.zeros(12, device=device)

    m_opencv, r_opencv = _project_points(pts, K, zeros, w, h)
    m_pinhole, r_pinhole = _project_points(pts, K, zeros, w, h)
    vis = (r_opencv[:, 0] > 0) & (r_pinhole[:, 0] > 0)
    assert vis.any()
    assert torch.allclose(m_opencv[vis], m_pinhole[vis], atol=1e-3)


def _ut_render(camera_model: CameraModelType, dist: torch.Tensor | None, seed: int = 7):
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend

    device = torch.device("cuda")
    w = h = 64
    torch.manual_seed(seed)
    model = FourDGaussians.from_random(n_points=300, sh_degree=2).to(device)
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 3.0
    viewmat = viewmat.unsqueeze(0)
    K = torch.tensor([[70.0, 0.0, w / 2.0], [0.0, 70.0, h / 2.0], [0.0, 0.0, 1.0]], device=device).unsqueeze(0)
    backend = UTFusedBackend(RenderingConfig())
    camera = BatchedCamera(viewmat, K, camera_model, w, h, dist)
    with torch.no_grad():
        return backend.render(model, camera, 0.0, model.sh_degree)


def test_zero_distortion_collapses() -> None:
    """In a render, OPENCV with zero distortion == PINHOLE within tol."""
    device = torch.device("cuda")
    out_opencv = _ut_render(CameraModelType.OPENCV, torch.zeros(1, 12, device=device))
    out_pinhole = _ut_render(CameraModelType.PINHOLE, torch.zeros(1, 12, device=device))
    assert torch.allclose(out_opencv.colors, out_pinhole.colors, atol=1e-4)
    assert torch.allclose(out_opencv.alphas, out_pinhole.alphas, atol=1e-4)


def test_opencv_subset_equiv_full() -> None:
    """OPENCV(5) == OPENCV_FULL with high-order slots zeroed → identical render."""
    device = torch.device("cuda")
    d5 = torch.zeros(1, 12, device=device)
    d5[0, :5] = torch.tensor([0.2, -0.05, 0.001, -0.002, 0.01], device=device)
    out_opencv = _ut_render(CameraModelType.OPENCV, d5)
    out_full = _ut_render(CameraModelType.OPENCV_FULL, d5.clone())
    assert torch.allclose(out_opencv.colors, out_full.colors, atol=1e-5)
    assert torch.allclose(out_opencv.alphas, out_full.alphas, atol=1e-5)


def test_opencv_full_vs_cv2_rational() -> None:
    """Forward OPENCV_FULL (12-coeff rational + thin-prism) matches cv2.projectPoints."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    device = torch.device("cuda")
    w, h = 512, 512
    fx = fy = 400.0
    K = torch.tensor([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], device=device)

    # [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
    coeffs = [0.15, -0.05, 0.001, -0.002, 0.01, 0.03, -0.01, 0.005, 0.0008, -0.0006, 0.0004, -0.0002]
    dist12 = torch.tensor(coeffs, device=device)

    pts = _grid_points(device)
    means2d, radii = _project_points(pts, K, dist12, w, h)

    obj = pts.cpu().numpy().astype(np.float64).reshape(-1, 1, 3)
    # cv2 distCoeffs order for the rational+thin-prism model: [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
    cv_dist = np.array(coeffs, dtype=np.float64)
    cv_K = K.cpu().numpy().astype(np.float64)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), cv_K, cv_dist)
    img = torch.tensor(img.reshape(-1, 2), device=device, dtype=torch.float32)

    visible = radii[:, 0] > 0
    assert visible.any(), "no visible grid points"
    err = (means2d[visible] - img[visible]).abs().max().item()
    assert err < 0.2, f"OPENCV_FULL vs cv2.projectPoints max error {err:.4f}px"


def test_fisheye_distortion_vs_cv2() -> None:
    """Forward FISHEYE_OPENCV (KB) matches cv2.fisheye.projectPoints within tol."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    device = torch.device("cuda")
    w, h = 512, 512
    fx = fy = 250.0
    K = torch.tensor([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], device=device)

    k1, k2, k3, k4 = 0.05, -0.01, 0.002, -0.0003
    dist4 = torch.tensor([k1, k2, k3, k4], device=device)

    pts = _grid_points(device)
    means2d, radii = _project_points_fisheye(pts, K, dist4, w, h)

    obj = pts.cpu().numpy().astype(np.float64).reshape(-1, 1, 3)
    cv_dist = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
    cv_K = K.cpu().numpy().astype(np.float64)
    img, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), cv_K, cv_dist)
    img = torch.tensor(img.reshape(-1, 2), device=device, dtype=torch.float32)

    visible = radii[:, 0] > 0
    assert visible.any(), "no visible grid points"
    err = (means2d[visible] - img[visible]).abs().max().item()
    assert err < 0.2, f"FISHEYE_OPENCV vs cv2.fisheye.projectPoints max error {err:.4f}px"


def test_fisheye_equidistant_equals_fisheye_zero() -> None:
    """FISHEYE_EQUIDISTANT renders bit-identical to FISHEYE_OPENCV with zero coefficients."""
    device = torch.device("cuda")
    zeros4 = torch.zeros(1, 4, device=device)
    out_eq = _ut_render(CameraModelType.FISHEYE_EQUIDISTANT, zeros4)
    out_fish = _ut_render(CameraModelType.FISHEYE_OPENCV, zeros4.clone())
    assert torch.equal(out_eq.colors, out_fish.colors)
    assert torch.equal(out_eq.alphas, out_fish.alphas)


def test_fisheye_equidistant_vs_cv2() -> None:
    """Forward FISHEYE_EQUIDISTANT (r=f*theta) matches cv2.fisheye.projectPoints with D=0 (D-4)."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    device = torch.device("cuda")
    w, h = 512, 512
    fx = fy = 250.0
    K = torch.tensor([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], device=device)

    dist4 = torch.zeros(4, device=device)  # equidistant = KB with k1=k2=k3=k4=0

    pts = _grid_points(device)
    means2d, radii = _project_points_fisheye(pts, K, dist4, w, h)

    obj = pts.cpu().numpy().astype(np.float64).reshape(-1, 1, 3)
    cv_dist = np.zeros((4, 1), dtype=np.float64)
    cv_K = K.cpu().numpy().astype(np.float64)
    img, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), cv_K, cv_dist)
    img = torch.tensor(img.reshape(-1, 2), device=device, dtype=torch.float32)

    visible = radii[:, 0] > 0
    assert visible.any(), "no visible grid points"
    err = (means2d[visible] - img[visible]).abs().max().item()
    assert err < 0.2, f"FISHEYE_EQUIDISTANT vs cv2.fisheye.projectPoints max error {err:.4f}px"


@pytest.mark.parametrize(
    "dist5",
    [
        [0.1, -0.03, 0.0, 0.0, 0.0],  # moderate radial
        [0.2, 0.05, 0.001, -0.001, 0.0],  # radial + tangential
    ],
)
def test_undistort_newton_converges(dist5: list[float]) -> None:
    """Project ∘ unproject ≈ identity for moderate distortion."""
    d = dist5 + [0.0] * 7
    max_err = 0.0
    for u in (-0.4, -0.1, 0.0, 0.2, 0.5):
        for v in (-0.3, 0.0, 0.4):
            xd, yd = _brown_conrady_project(u, v, d)  # forward distort
            xu, yu = _newton_undistort(xd, yd, d)  # invert
            max_err = max(max_err, math.hypot(xu - u, yu - v))
    assert max_err < 1e-5, f"Newton undistort round-trip error {max_err:.2e}"
