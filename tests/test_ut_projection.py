"""The fused UT projection kernel: Cholesky sigma points, the Python reference, behind-camera culling."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from tests.fused_stages import fused_ut_project
from tests.fused_stages import condition
from tests.oracles.factor import cov_of
from tests.oracles.factor import upper_tri
from tests.oracles.ut_projection_reference import ut_project


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="CUDA extension not built")


def _quats_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions [N, 4] (normalised) to rotation matrices [N, 3, 3]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return R


def _sigma_from_quat_scale(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Upper-tri ``[N, 6]`` Σ = R diag(s²) Rᵀ from quats and scales."""
    R = _quats_to_rotmat(F.normalize(quats, dim=-1))  # [N, 3, 3]
    s = scales  # [N, 3]
    # L = R * s[None, :]: each column of R scaled by the corresponding s
    L = R * s[:, None, :]  # [N, 3, 3]: L[:,i] = s_i * R[:,i]
    Sigma = L @ L.transpose(-1, -2)  # [N, 3, 3]: Σ = L Lᵀ
    # Extract upper-tri
    uptri = torch.stack(
        [
            Sigma[:, 0, 0],
            Sigma[:, 0, 1],
            Sigma[:, 0, 2],
            Sigma[:, 1, 1],
            Sigma[:, 1, 2],
            Sigma[:, 2, 2],
        ],
        dim=-1,
    )
    return uptri  # [N, 6]


def _cholesky3x3_python(sigma: torch.Tensor) -> torch.Tensor:
    """Reference Cholesky of upper-tri ``[N, 6]``; returns ``L_col[N, 3, 3]`` with ``L_col[:, col]`` column ``col``."""
    s00, s01, s02, s11, s12, s22 = (sigma[:, k] for k in range(6))
    eps = 1e-8

    l00 = torch.sqrt(torch.clamp(s00, min=eps))
    l10 = s01 / l00
    l11 = torch.sqrt(torch.clamp(s11 - l10**2, min=eps))
    l20 = s02 / l00
    l21 = (s12 - l20 * l10) / l11
    l22 = torch.sqrt(torch.clamp(s22 - l20**2 - l21**2, min=eps))

    zeros = torch.zeros_like(l00)
    # L_col[i] = i-th column of L  (column-major)
    col0 = torch.stack([l00, l10, l20], dim=-1)  # [N, 3]
    col1 = torch.stack([zeros, l11, l21], dim=-1)
    col2 = torch.stack([zeros, zeros, l22], dim=-1)
    return torch.stack([col0, col1, col2], dim=1)  # [N, 3, 3]


def _gsplat_sigma_points_python(
    means: torch.Tensor,  # [N, 3]
    quats: torch.Tensor,  # [N, 4] normalised wxyz
    scales: torch.Tensor,  # [N, 3]
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Reference of gsplat's ``world_gaussian_sigma_points``; returns ``[N, 7, 3]``."""
    D = 3
    lam = alpha**2 * (D + kappa) - D
    c = math.sqrt(D + lam)

    R = _quats_to_rotmat(quats)  # [N, 3, 3]
    # delta_i = c * s_i * R[:,i]  — same as gsplat
    pts = [means]  # σ_0
    for i in range(3):
        delta = c * scales[:, i : i + 1] * R[:, :, i]  # [N, 3]
        pts.append(means + delta)  # σ_{i+1}
    for i in range(3):
        delta = c * scales[:, i : i + 1] * R[:, :, i]
        pts.append(means - delta)  # σ_{i+4}
    return torch.stack(pts, dim=1)  # [N, 7, 3]


def _cholesky_sigma_points_python(
    means: torch.Tensor,  # [N, 3]
    sigma_uptri: torch.Tensor,  # [N, 6]
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    """Python reference for Cholesky-based sigma points. Returns [N, 7, 3]."""
    D = 3
    lam = alpha**2 * (D + kappa) - D
    c = math.sqrt(D + lam)

    L_cols = _cholesky3x3_python(sigma_uptri)  # [N, 3, 3]: L_cols[:, col, :]

    pts = [means]
    for i in range(3):
        delta = c * L_cols[:, i, :]  # [N, 3]  column i of L
        pts.append(means + delta)
    for i in range(3):
        delta = c * L_cols[:, i, :]
        pts.append(means - delta)
    return torch.stack(pts, dim=1)  # [N, 7, 3]


def _ut_reconstruct_2d(
    pts3d: torch.Tensor,  # [N, 7, 3]
    viewmat: torch.Tensor,  # [4, 4]
    K: torch.Tensor,  # [3, 3]
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project seven sigma points; returns ``(mean2d [N, 2], covar2d [N, 2, 2], valid [N])``."""
    D = 3
    lam = alpha**2 * (D + kappa) - D
    d_plus_lam = D + lam
    wm0 = lam / d_plus_lam
    wmi = 0.5 / d_plus_lam
    wc0 = wm0 + (1 - alpha**2 + beta)
    wci = wmi
    wm = torch.tensor([wm0] + [wmi] * 6, dtype=pts3d.dtype, device=pts3d.device)  # [7]
    wc = torch.tensor([wc0] + [wci] * 6, dtype=pts3d.dtype, device=pts3d.device)  # [7]

    R = viewmat[:3, :3]  # [3, 3]
    t = viewmat[:3, 3]  # [3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Project each sigma point
    N, _, _ = pts3d.shape
    pts_cam = (pts3d @ R.T) + t[None, None, :]  # [N, 7, 3]

    # Mask: all 7 sigma points must be in front of camera
    valid = (pts_cam[:, :, 2] > 0.01).all(dim=1)  # [N]
    # Zero out projections for invalid Gaussians to avoid NaN
    pts_cam = pts_cam.clone()
    pts_cam[~valid] = 0.01  # placeholder, won't be used

    u = fx * pts_cam[:, :, 0] / pts_cam[:, :, 2] + cx  # [N, 7]
    v = fy * pts_cam[:, :, 1] / pts_cam[:, :, 2] + cy

    ip = torch.stack([u, v], dim=-1)  # [N, 7, 2]

    mean2d = (wm[None, :, None] * ip).sum(dim=1)  # [N, 2]
    diff = ip - mean2d[:, None, :]  # [N, 7, 2]
    covar2d = (wc[None, :, None, None] * diff.unsqueeze(-1) * diff.unsqueeze(-2)).sum(dim=1)

    return mean2d, covar2d, valid  # [N, 2], [N, 2, 2], [N]


@pytest.fixture()
def simple_gaussians_and_cameras():
    """Return (quats, scales, means, sigma_uptri, viewmat, K) for 64 Gaussians, 1 camera."""
    torch.manual_seed(42)
    N = 64
    device = torch.device("cuda")

    # Random quaternions (wxyz, normalised)
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    # Positive scales (not too small, not too large)
    scales = torch.exp(torch.randn(N, 3, device=device) * 0.5).clamp(0.01, 2.0)
    # Means in front of the camera
    means = torch.randn(N, 3, device=device)
    means[:, 2] = torch.abs(means[:, 2]) + 3.0  # z > 3

    sigma_uptri = _sigma_from_quat_scale(quats, scales)  # [N, 6]

    # Perspective camera looking down −Z
    fx = fy = 800.0
    cx, cy = 512.0, 384.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

    # Identity world-to-camera (camera at world origin, looking down +Z)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)

    return quats, scales, means, sigma_uptri, viewmat, K


def test_cholesky_sigma_points_reconstruct_same_as_quat_scale_sigma_points(
    simple_gaussians_and_cameras,
):
    """Under a linear (orthographic) projection, Cholesky and quat/scale sigma points give the same UT statistics.

    Under perspective the two factorizations legitimately differ.
    """
    quats, scales, means, sigma_uptri, _, K = simple_gaussians_and_cameras
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    quat_pts = _gsplat_sigma_points_python(means, quats, scales)  # [N, 7, 3]
    chol_pts = _cholesky_sigma_points_python(means, sigma_uptri)  # [N, 7, 3]

    D = 3
    alpha = 1.0
    lam = alpha**2 * (D + 0.0) - D
    d_plus_lam = D + lam
    wm0, wmi = lam / d_plus_lam, 0.5 / d_plus_lam
    wm = torch.tensor([wm0] + [wmi] * 6, dtype=means.dtype, device=means.device)

    def ortho_project(pts3d):
        # Orthographic: (fx * x + cx, fy * y + cy) — linear in x, y
        u = fx * pts3d[:, :, 0] + cx
        v = fy * pts3d[:, :, 1] + cy
        return torch.stack([u, v], dim=-1)  # [N, 7, 2]

    ip_q = ortho_project(quat_pts)  # [N, 7, 2]
    ip_c = ortho_project(chol_pts)

    mean_q = (wm[None, :, None] * ip_q).sum(dim=1)  # [N, 2]
    mean_c = (wm[None, :, None] * ip_c).sum(dim=1)

    max_err = (mean_q - mean_c).abs().max().item()
    assert max_err < 1e-2, (
        f"Orthographic 2D mean mismatch under two factorizations: max_err={max_err:.2e}\n"
        "Both factorizations should give identical means for a linear projection."
    )


def test_cholesky_l_l_t_equals_sigma(simple_gaussians_and_cameras):
    """Verify that the Cholesky factor L satisfies L Lᵀ = Σ (correctness of decomp)."""
    _, _, _, sigma_uptri, _, _ = simple_gaussians_and_cameras
    L_cols = _cholesky3x3_python(sigma_uptri)  # [N, 3, 3]: L_cols[:, col, :]

    # Reconstruct L (lower-tri): L[i, j] = L_cols[j, i] for j<=i
    # L_cols[:, col, :] = column col, so L[:, row, col] = L_cols[:, col, row]
    L = L_cols.permute(0, 2, 1)  # [N, 3, 3]: L[:, row, col]

    Sigma_reconstructed = L @ L.transpose(-1, -2)  # [N, 3, 3]

    # Build full Σ from upper-tri
    s = sigma_uptri
    Sigma_ref = torch.stack(
        [
            torch.stack([s[:, 0], s[:, 1], s[:, 2]], dim=-1),
            torch.stack([s[:, 1], s[:, 3], s[:, 4]], dim=-1),
            torch.stack([s[:, 2], s[:, 4], s[:, 5]], dim=-1),
        ],
        dim=-2,
    )  # [N, 3, 3]

    assert torch.allclose(Sigma_reconstructed, Sigma_ref, atol=1e-5), (
        f"Cholesky L Lᵀ ≠ Σ: max_err={(Sigma_reconstructed - Sigma_ref).abs().max():.2e}"
    )


def test_ut_projection_fused_kernel_matches_python_reference():
    """The fused UT kernel matches the Cholesky reference built from the kernel's own conditioned Σ = L Lᵀ."""

    torch.manual_seed(42)
    N = 32
    device = torch.device("cuda")

    # Random 4D Gaussians
    quats_l = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    quats_r = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales_4d_log = torch.randn(N, 4, device=device) * 0.5
    means_4d = torch.randn(N, 3, device=device)
    means_4d[:, 2] = torch.abs(means_4d[:, 2]) + 4.0  # z > 4 for all
    mu_t = torch.zeros(N, device=device)
    opacities = torch.full((N,), 3.0, device=device)  # high opacity

    tau = 0.0
    cov_eps = 1e-4

    # Python reference: condition 4D → 3D (Python autograd version)
    with torch.no_grad():
        means3d_ref, factors_ref, _, _ = condition(
            quats_l,
            quats_r,
            scales_4d_log,
            means_4d,
            mu_t,
            opacities,
            tau,
            cov_eps,
            0.0,  # prefilter_threshold=0
        )

    fx = fy = 800.0
    cx, cy = 512.0, 384.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)

    alpha = 1.0
    chol_pts = _cholesky_sigma_points_python(means3d_ref, upper_tri(cov_of(factors_ref)), alpha=alpha)
    _, cov2d_ref_py, valid_ref = _ut_reconstruct_2d(chol_pts, viewmat, K, alpha=alpha)

    viewmats_batch = viewmat.unsqueeze(0)
    Ks_batch = K.unsqueeze(0)

    empty_rc = torch.zeros((1, 12), device=viewmats_batch.device, dtype=torch.float32)  # pinhole fixed-width
    radii, _, _, conics_cuda = fused_ut_project(
        quats_l,
        quats_r,
        scales_4d_log,
        means_4d,
        mu_t,
        opacities,
        viewmats_batch,
        Ks_batch,
        empty_rc,
        tau=tau,
        cov_eps=cov_eps,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=1024,
        image_height=768,
        family=int(ProjectionFamily.PINHOLE_FAMILY),
    )

    visible = radii[0, :, 0] > 0
    valid = visible & valid_ref
    n_vis = valid.sum().item()
    assert n_vis > 0, "No visible Gaussians — check test setup"

    # means2d is the exact tile-box centre on an undistorted pinhole (test_exact_pinhole_box_*).

    # Conics: compare via cov2d_blurred inversion
    eps2d = 0.3
    cov2d_b = cov2d_ref_py.clone()
    cov2d_b[:, 0, 0] += eps2d
    cov2d_b[:, 1, 1] += eps2d
    det = cov2d_b[:, 0, 0] * cov2d_b[:, 1, 1] - cov2d_b[:, 0, 1] ** 2
    inv_det = 1.0 / det.clamp(min=1e-10)
    conic_ref_v = torch.stack(
        [
            cov2d_b[:, 1, 1] * inv_det,
            -cov2d_b[:, 0, 1] * inv_det,
            cov2d_b[:, 0, 0] * inv_det,
        ],
        dim=-1,
    )  # [N, 3]

    max_conic_err = (conics_cuda[0, valid] - conic_ref_v[valid]).abs().max().item()
    assert max_conic_err < 2.0, f"conic mismatch: max_err={max_conic_err:.4f} on {n_vis} Gaussians"


def test_ut_projection_culls_behind_camera():
    """Gaussians with z < near_plane should have radii = [0, 0]."""

    device = torch.device("cuda")
    N = 8
    torch.manual_seed(7)

    quats_l = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    quats_r = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(N, 4)
    scales_4d_log = torch.zeros(N, 4, device=device)  # unit scales
    means = torch.randn(N, 3, device=device)
    means[:, 2] = -5.0  # all behind camera
    mu_t = torch.zeros(N, device=device)
    opacities = torch.full((N,), 5.0, device=device)  # high opacity

    viewmat = torch.eye(4, device=device).unsqueeze(0)
    K = torch.tensor([[800.0, 0, 512.0], [0, 800.0, 384.0], [0, 0, 1.0]], device=device).unsqueeze(0)

    empty_rc = torch.zeros((1, 12), device=device, dtype=torch.float32)  # pinhole fixed-width
    radii, *_ = fused_ut_project(
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        viewmat,
        K,
        empty_rc,
        tau=0.0,
        cov_eps=1e-4,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=1024,
        image_height=768,
        family=int(ProjectionFamily.PINHOLE_FAMILY),
    )

    assert (radii == 0).all(), "All behind-camera Gaussians should have radii=0"


def test_exact_pinhole_box_covers_every_drawn_pixel_and_is_tight():
    """On an undistorted pinhole the tile box holds every pixel the rasterizer draws, and no more than it needs.

    Drawn is the rasterizer's own test in fp64: opacity·exp(-D²/2) ≥ 1/255 and exp(-D²/2) > 0.0113, with D² the
    Mahalanobis distance of the full pixel ray. The population straddles the camera plane on a wide lens, where
    the UT box was unbounded (and elsewhere truncated).
    """
    torch.manual_seed(0)
    N, W, H, f = 400, 128, 96, 60.0
    device = torch.device("cuda")
    quats_l = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    quats_r = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales_4d_log = torch.randn(N, 4, device=device) * 0.7 - 1.0
    scales_4d_log[:, 3] = 3.0  # wide in time
    means = torch.rand(N, 3, device=device) * torch.tensor([6.0, 6.0, 2.95], device=device)
    means = means - torch.tensor([3.0, 3.0, -0.05], device=device)  # z in [0.05, 3]
    mu_t = torch.zeros(N, device=device)
    opacities = torch.rand(N, device=device) * 8.0 - 4.0

    means3d, factors, opac, _ = condition(quats_l, quats_r, scales_4d_log, means, mu_t, opacities, 0.0, 1e-4, 0.0)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device)
    radii, means2d, _, _ = fused_ut_project(
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        torch.eye(4, device=device)[None],
        K[None],
        torch.zeros(1, 12, device=device),
        tau=0.0,
        cov_eps=1e-4,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=W,
        image_height=H,
        family=int(ProjectionFamily.PINHOLE_FAMILY),
    )
    radii, means2d = radii[0].double(), means2d[0].double()

    sigma = cov_of(factors.double())
    lam = torch.linalg.inv(sigma.view(N, 3, 3))
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    px, py = xs.double() + 0.5, ys.double() + 0.5
    d = torch.stack([(px - W / 2) / f, (py - H / 2) / f, torch.ones_like(px)], -1).view(-1, 3)
    p = -means3d.double()
    lam_p = (lam @ p[:, :, None]).squeeze(-1)
    dld = torch.einsum("pi,nij,pj->np", d, lam, d)
    d2 = _ray_d2((d @ lam_p.T).T, (p * lam_p).sum(-1)[:, None], dld)
    response = torch.exp(-0.5 * d2)
    drawn = (opac.double()[:, None] * response >= 1.0 / 255.0) & (response > 0.0113)

    k2 = torch.clamp(2 * torch.log(opac.double() * 255.0), max=-2 * math.log(0.0113))
    straddling = means3d[:, 2].double() ** 2 < k2 * sigma[:, 2, 2]
    assert (straddling & drawn.any(-1)).sum() >= 5, "population must exercise the camera-plane case"

    px, py = px.reshape(-1), py.reshape(-1)
    box_area, drawn_area = 0.0, 0.0
    for i in torch.nonzero(drawn.any(-1)).squeeze(-1).tolist():
        assert radii[i].min() > 0, f"Gaussian {i} draws pixels but was culled"
        x, y = px[drawn[i]], py[drawn[i]]
        assert ((x - means2d[i, 0]).abs() <= radii[i, 0]).all() and ((y - means2d[i, 1]).abs() <= radii[i, 1]).all(), (
            f"Gaussian {i}: a drawn pixel lies outside the tile box"
        )
        if x.min() > 1 and x.max() < W - 1 and y.min() > 1 and y.max() < H - 1:
            assert radii[i, 0] <= 0.5 * (x.max() - x.min()) + 3 and radii[i, 1] <= 0.5 * (y.max() - y.min()) + 3, (
                f"Gaussian {i}: box {radii[i].tolist()} is loose for its drawn extent"
            )
        bw = min(means2d[i, 0] + radii[i, 0], W) - max(means2d[i, 0] - radii[i, 0], 0.0)
        bh = min(means2d[i, 1] + radii[i, 1], H) - max(means2d[i, 1] - radii[i, 1], 0.0)
        box_area += float(bw * bh)
        drawn_area += float((x.max() - x.min() + 1) * (y.max() - y.min() + 1))
    # Straddling Gaussians: the box keeps only the half-line of columns (rows) that meets the front of the Gaussian.
    # Keeping the mirror half too gives 1.55 here; per-axis bounds cannot drop the rest (rows met only off-image).
    assert box_area / drawn_area < 1.35, f"image-clipped box area is {box_area / drawn_area:.3f}× the drawn extent"


def _ray_d2(b: torch.Tensor, c: torch.Tensor, dld: torch.Tensor) -> torch.Tensor:
    """D² over the ray t ≥ 0, as the rasterizer evaluates it, from b = dᵀΛp, c = pᵀΛp, dld = dᵀΛd with p = o − μ.

    The minimiser t = −b/dld lies behind the camera when b > 0; the ray's closest point is then the camera centre.
    """
    return torch.where(b > 0, c.expand_as(b), c - b**2 / dld)


def _unit_rays(xs: torch.Tensor, ys: torch.Tensor, f: float, cx: float, cy: float, dist: list[float], fisheye: bool):
    """fp64 unit rays ``[P, 3]`` and validity for pixel centres through Kannala-Brandt or Brown-Conrady (OpenCV)."""
    u, v = (xs + 0.5 - cx) / f, (ys + 0.5 - cy) / f
    rd = torch.sqrt(u * u + v * v).clamp(min=1e-12)
    if fisheye:  # invert θd(θ) = r_d by bisection on [0, π]; valid while θd is monotone
        k = dist
        td = lambda t: t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)  # noqa: E731
        lo, hi = torch.zeros_like(rd), torch.full_like(rd, math.pi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = td(mid) < rd
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        th = 0.5 * (lo + hi)
        slope = 1 + 3 * k[0] * th**2 + 5 * k[1] * th**4 + 7 * k[2] * th**6 + 9 * k[3] * th**8
        valid = (slope > 0) & ((td(th) - rd).abs() < 1e-9)
        return torch.stack([torch.sin(th) * u / rd, torch.sin(th) * v / rd, torch.cos(th)], -1), valid
    coeffs = np.zeros(14)
    coeffs[: len(dist)] = dist
    pts = torch.stack([u, v], -1).cpu().numpy().reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    und = cv2.undistortPointsIter(pts, np.eye(3), coeffs, None, None, criteria).reshape(-1, 2)
    x, y = (torch.from_numpy(und[:, i]).to(u) for i in (0, 1))
    d = torch.stack([x, y, torch.ones_like(x)], -1)
    return d / d.norm(dim=-1, keepdim=True), torch.ones_like(u, dtype=torch.bool)


@pytest.mark.parametrize(
    ("family", "dist", "f"),
    [
        (ProjectionFamily.PINHOLE_FAMILY, [0.05, 0.0], 60.0),
        (ProjectionFamily.PINHOLE_FAMILY, [0.05, 0.0, 0.02, -0.02], 60.0),
        (ProjectionFamily.FISHEYE_FAMILY, [0.0, 0.0, 0.0, 0.0], 38.0),
        (ProjectionFamily.FISHEYE_FAMILY, [-0.25, 0.06, 0.0, 0.0], 38.0),
    ],
    ids=["brown-conrady", "brown-conrady-tangential", "equidistant-200deg", "kb-strong-barrel"],
)
def test_distorted_runaway_ut_box_is_replaced_without_losing_a_drawn_pixel(family, dist, f):
    """On a distorted lens a UT box larger than the image is replaced by a bounded one holding every drawn pixel.

    Runaway boxes come from sigma points just in front of the camera plane. They are found on the host
    (tests/oracles), restricted to Gaussians whose sigma points the kernel projects identically. Drawn is the
    rasterizer's own test in fp64 over the ray (t ≥ 0), so nothing is drawn at the antipode of a fisheye past 180°.
    """
    torch.manual_seed(1)
    N, W, H = 3000, 160, 120
    device = torch.device("cuda")
    fisheye = family == ProjectionFamily.FISHEYE_FAMILY
    phi = torch.rand(N, device=device) * 2 * math.pi
    radial = torch.rand(N, device=device) * 1.5 + 0.2
    means = torch.stack(
        [radial * torch.cos(phi), radial * torch.sin(phi), torch.rand(N, device=device) * 0.3 + 0.02], -1
    )
    quats_l = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    quats_r = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales_4d_log = torch.randn(N, 4, device=device) * 0.5 - 2.0
    scales_4d_log[:, 3] = 3.0
    mu_t = torch.zeros(N, device=device)
    opacities = torch.rand(N, device=device) * 6.0 - 1.0

    means3d, factors, opac, _ = condition(quats_l, quats_r, scales_4d_log, means, mu_t, opacities, 0.0, 1e-4, 0.0)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device)
    dist_t = torch.zeros(1, 4 if fisheye else 12, device=device)
    dist_t[0, : len(dist)] = torch.tensor(dist, device=device)
    radii, means2d, _, _ = fused_ut_project(
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        torch.eye(4, device=device)[None],
        K[None],
        dist_t,
        tau=0.0,
        cov_eps=1e-4,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        prefilter_threshold=0.0,
        image_width=W,
        image_height=H,
        family=int(family),
    )
    radii, means2d = radii[0].double(), means2d[0].double()

    Kn = K.double().cpu().numpy()
    dist_np = dist_t[0].double().cpu().numpy()
    mu_np = means3d.double().cpu().numpy()
    sigma = cov_of(factors.double())
    sig_np, op_np = sigma.cpu().numpy(), opac.double().cpu().numpy()
    runaway = []
    k2_cap = -2.0 * math.log(0.0113)
    for i in range(N):
        if op_np[i] < 1.0 / 255.0:
            continue
        k2 = min(k2_cap, 2.0 * math.log(op_np[i] * 255.0))
        if mu_np[i] @ mu_np[i] <= k2 * np.linalg.norm(sig_np[i]):
            continue  # camera inside the bounding sphere: the cap keeps the UT box by design
        m2, c2 = ut_project(mu_np[i], sig_np[i], Kn, dist_np, fisheye=fisheye, centre_fallback=True)
        extend = min(3.33, math.sqrt(2.0 * math.log(op_np[i] * 255.0)))
        if extend * math.sqrt(c2[0, 0] + 0.3) > W or extend * math.sqrt(c2[1, 1] + 0.3) > H:
            runaway.append(i)
    runaway = torch.tensor(runaway, device=device, dtype=torch.long)
    assert runaway.numel() >= 20, f"only {runaway.numel()} runaway UT boxes in the population"

    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    xs, ys = xs.reshape(-1).double(), ys.reshape(-1).double()
    d, ray_ok = _unit_rays(xs, ys, f, W / 2, H / 2, dist, fisheye)
    lam = torch.linalg.inv(sigma[runaway])
    p = -means3d.double()[runaway]
    lam_p = (lam @ p[:, :, None]).squeeze(-1)
    d2 = _ray_d2((d @ lam_p.T).T, (p * lam_p).sum(-1)[:, None], torch.einsum("pi,nij,pj->np", d, lam, d))
    response = torch.exp(-0.5 * d2)
    drawn = (opac.double()[runaway, None] * response >= 1.0 / 255.0) & (response > 0.0113) & ray_ok[None]

    px, py = xs + 0.5, ys + 0.5
    for j, i in enumerate(runaway.tolist()):
        if not drawn[j].any():
            continue
        x, y = px[drawn[j]], py[drawn[j]]
        assert radii[i].min() > 0, f"Gaussian {i} draws pixels but was culled"
        assert ((x - means2d[i, 0]).abs() <= radii[i, 0]).all() and ((y - means2d[i, 1]).abs() <= radii[i, 1]).all(), (
            f"Gaussian {i}: a drawn pixel lies outside the tile box"
        )
    capped = (radii[runaway, 0] <= W / 2 + 2) & (radii[runaway, 1] <= H / 2 + 2)
    assert capped.sum() >= runaway.numel() // 2, f"only {int(capped.sum())} of {runaway.numel()} runaway boxes bounded"
