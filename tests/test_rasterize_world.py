"""World-space ray-Gaussian rasterization kernel against PyTorch references."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from anycam4d.camera.batched import BatchedCamera
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from gsplat.cuda._wrapper import isect_offset_encode
from gsplat.cuda._wrapper import isect_tiles

from tests.oracles.factor import cov_of
from tests.oracles.factor import factor_of
from tests.oracles.factor import from_upper_tri
from tests.oracles.factor import upper_tri
from anycam4d.enums import CameraModelType


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="CUDA extension not built")

if HAS_CUDA_EXT:
    from anycam4d.cuda.autograd import RasterizeWorldFunction

# Mirrors MAX_RASTER_KERNEL_DENSITY in cuda/csrc/common.h — the shader's dimensionless
# whitened-Mahalanobis 3sigma cutoff on the ray-Gaussian response.
MAX_RASTER_KERNEL_DENSITY = 0.0113


def _invert3x3_sym_python(sigma: torch.Tensor) -> torch.Tensor:
    """Invert symmetric 3x3 matrices given as upper-tri ``[N, 6]``; returns upper-tri precision."""
    s00, s01, s02, s11, s12, s22 = (sigma[:, k] for k in range(6))

    det = s00 * (s11 * s22 - s12 * s12) - s01 * (s01 * s22 - s12 * s02) + s02 * (s01 * s12 - s11 * s02)
    inv = 1.0 / det.clamp(min=torch.finfo(det.dtype).tiny)

    l00 = (s11 * s22 - s12 * s12) * inv
    l01 = -(s01 * s22 - s12 * s02) * inv
    l02 = (s01 * s12 - s11 * s02) * inv
    l11 = (s00 * s22 - s02 * s02) * inv
    l12 = -(s00 * s12 - s01 * s02) * inv
    l22 = (s00 * s11 - s01 * s01) * inv

    return torch.stack([l00, l01, l02, l11, l12, l22], dim=-1)  # [N, 6]


def _mahal_alpha_python(
    ray_origin: torch.Tensor,  # [3]
    ray_dir: torch.Tensor,  # [3] unit
    means_3d_cond: torch.Tensor,  # [N, 3]
    lambda_uptri: torch.Tensor,  # [N, 6]
    opacities: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Python reference: α = opacity · exp(−0.5 D²) using precision matrix."""
    o = ray_origin[None, :]  # [1, 3]
    d = ray_dir[None, :]  # [1, 3]
    p = o - means_3d_cond  # [N, 3]

    # Full Λ matrix per Gaussian
    l = lambda_uptri  # [N, 6]

    def lv(v):  # Λ @ v for [N, 3] vectors
        return torch.stack(
            [
                l[:, 0] * v[:, 0] + l[:, 1] * v[:, 1] + l[:, 2] * v[:, 2],
                l[:, 1] * v[:, 0] + l[:, 3] * v[:, 1] + l[:, 4] * v[:, 2],
                l[:, 2] * v[:, 0] + l[:, 4] * v[:, 1] + l[:, 5] * v[:, 2],
            ],
            dim=-1,
        )

    Ld = lv(d.expand(means_3d_cond.shape[0], 3))  # [N, 3]
    Lp = lv(p)  # [N, 3]

    a = (d.expand(means_3d_cond.shape[0], 3) * Ld).sum(-1)  # [N]
    b = (d.expand(means_3d_cond.shape[0], 3) * Lp).sum(-1)
    c = (p * Lp).sum(-1)

    D2 = c - b**2 / a.clamp(min=1e-10)
    return opacities * torch.exp(-0.5 * D2)


def _whitening_alpha_python(
    ray_origin: torch.Tensor,  # [3]
    ray_dir: torch.Tensor,  # [3] unit
    means_3d_cond: torch.Tensor,  # [N, 3]
    quats: torch.Tensor,  # [N, 4] wxyz normalised
    scales: torch.Tensor,  # [N, 3] positive
    opacities: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Python reference: gsplat whitening-based α."""
    # Build whitening transform A = S^{-1} R^T per Gaussian
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
    ).reshape(-1, 3, 3)  # [N, 3, 3]

    s_inv = 1.0 / scales  # [N, 3]
    A = s_inv[:, :, None] * R  # S^{-1} R column scaling → [N, 3, 3]
    # Actually A = S^{-1} R^T: each row of R^T scaled by 1/s_i
    A = R.transpose(-1, -2) * s_inv[:, :, None]  # [N, 3, 3]: A[i,j] = R[j,i]/s_i

    p = ray_origin[None, :] - means_3d_cond  # [N, 3]

    gro = (A @ p.unsqueeze(-1)).squeeze(-1)  # A(o-μ) [N, 3]
    grd_raw = (A @ ray_dir.unsqueeze(-1)).squeeze(-1)  # [N, 3]
    grd = F.normalize(grd_raw, dim=-1)  # normalise

    # power = -0.5 * ||grd × gro||²
    cross = torch.linalg.cross(grd, gro, dim=-1)  # [N, 3]
    D2 = (cross**2).sum(-1)  # [N]
    return opacities * torch.exp(-0.5 * D2)


def test_mahalanobis_alpha_matches_whitening_alpha():
    """Precision-matrix α must equal whitening-based α for identical Gaussians."""
    torch.manual_seed(42)
    N = 128
    device = torch.device("cuda")

    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.exp(torch.randn(N, 3, device=device) * 0.3).clamp(0.01, 2.0)
    means_3d_cond = torch.randn(N, 3, device=device)
    means_3d_cond[:, 2] += 4.0  # put in front of origin

    opacities = torch.sigmoid(torch.randn(N, device=device)) * 0.9 + 0.1

    # Build Σ from quats and scales, then invert to get Λ
    R = torch.stack(
        [
            (1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2)),
            2 * (quats[:, 1] * quats[:, 2] - quats[:, 3] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 3] + quats[:, 2] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 2] + quats[:, 3] * quats[:, 0]),
            (1 - 2 * (quats[:, 1] ** 2 + quats[:, 3] ** 2)),
            2 * (quats[:, 2] * quats[:, 3] - quats[:, 1] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 3] - quats[:, 2] * quats[:, 0]),
            2 * (quats[:, 2] * quats[:, 3] + quats[:, 1] * quats[:, 0]),
            (1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)),
        ],
        dim=-1,
    ).reshape(N, 3, 3)

    s = scales
    L = R * s[:, None, :]  # [N, 3, 3]
    Sigma_full = L @ L.transpose(-1, -2)  # [N, 3, 3]
    sigma_uptri = torch.stack(
        [
            Sigma_full[:, 0, 0],
            Sigma_full[:, 0, 1],
            Sigma_full[:, 0, 2],
            Sigma_full[:, 1, 1],
            Sigma_full[:, 1, 2],
            Sigma_full[:, 2, 2],
        ],
        dim=-1,
    )  # [N, 6]
    lambda_uptri = _invert3x3_sym_python(sigma_uptri)

    # Random ray
    ray_o = torch.tensor([0.1, -0.2, 0.0], device=device)
    ray_d = F.normalize(torch.tensor([0.0, 0.0, 1.0], device=device), dim=0)

    alpha_mahal = _mahal_alpha_python(ray_o, ray_d, means_3d_cond, lambda_uptri, opacities)
    alpha_whitening = _whitening_alpha_python(ray_o, ray_d, means_3d_cond, quats, scales, opacities)

    max_err = (alpha_mahal - alpha_whitening).abs().max().item()
    assert max_err < 1e-4, f"Mahalanobis and whitening alpha differ: max_err={max_err:.2e}"


@pytest.mark.filterwarnings(
    "ignore:Input #\\d+ requires gradient and is not a double:UserWarning"
)  # float32-only kernel
def test_rasterize_world_gradcheck():
    """gradcheck on RasterizeWorldFunction; float32, so tolerances are loose and primitives sit well inside the FOV."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from anycam4d.cuda.autograd import RasterizeWorldFunction

    torch.manual_seed(42)
    device = torch.device("cuda")

    N = 4  # very few Gaussians to keep gradcheck fast
    C = 1
    H, W = 8, 8
    TILE = 16
    tile_h = math.ceil(H / TILE)
    tile_w = math.ceil(W / TILE)
    CDIM = 3

    dtype = torch.float32

    # Well-conditioned PD covariances: small, isotropic-ish
    # Σ = 0.5 * I → all eigenvalues = 0.5
    covars_uptri_data = torch.zeros(N, 6, dtype=dtype, device=device)
    covars_uptri_data[:, 0] = 0.5  # σ00
    covars_uptri_data[:, 3] = 0.5  # σ11
    covars_uptri_data[:, 5] = 0.5  # σ22
    # Add small random off-diagonal to make the test non-trivial
    covars_uptri_data[:, 1] = 0.05 * torch.randn(N, device=device, dtype=dtype)
    covars_uptri_data[:, 2] = 0.05 * torch.randn(N, device=device, dtype=dtype)
    covars_uptri_data[:, 4] = 0.05 * torch.randn(N, device=device, dtype=dtype)
    factors = factor_of(from_upper_tri(covars_uptri_data)).requires_grad_(True)

    # Gaussians centered around image center, well in front of camera
    fx = fy = 100.0  # large focal length → small FOV → Gaussians centered in image
    cx, cy = W / 2.0, H / 2.0
    z0 = 5.0  # fixed depth
    # Place each Gaussian near the pixel centers of a 2x2 sub-image
    mu_xy = torch.tensor(
        [
            [cx - 1.5, cy - 1.5],
            [cx + 1.5, cy - 1.5],
            [cx - 1.5, cy + 1.5],
            [cx + 1.5, cy + 1.5],
        ],
        dtype=dtype,
        device=device,
    )  # pixel coordinates

    # Back-project to world: x = (px - cx) / fx * z
    mu_world = torch.zeros(N, 3, dtype=dtype, device=device)
    mu_world[:, 0] = (mu_xy[:, 0] - cx) / fx * z0
    mu_world[:, 1] = (mu_xy[:, 1] - cy) / fy * z0
    mu_world[:, 2] = z0
    means_3d_cond = mu_world.requires_grad_(True)

    # High opacities so Gaussians are clearly visible (well above 1/255 threshold)
    opacities = torch.full((N,), 0.8, dtype=dtype, device=device).requires_grad_(True)
    colors = torch.rand(C, N, CDIM, dtype=dtype, device=device).requires_grad_(True)

    viewmats = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device).unsqueeze(0)

    # Tile structure: all Gaussians cover the single tile
    means2d_px = mu_xy  # [N, 2] pixel coordinates
    radii_f = torch.ones(N, 2, dtype=torch.int32, device=device) * 3
    depths_f = torch.full((N,), z0, dtype=torch.float32, device=device)

    _, isect_ids, flatten_ids = isect_tiles(
        means2d_px.unsqueeze(0),
        radii_f.unsqueeze(0),
        depths_f.unsqueeze(0),
        TILE,
        tile_w,
        tile_h,
        packed=False,
        n_images=C,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h).reshape(C, tile_h, tile_w)

    empty_rc = torch.zeros((C, 12), device=device, dtype=dtype)  # pinhole fixed-width

    def run(means3d_, covars_, opacities_, colors_):
        rc, ra = RasterizeWorldFunction.apply(
            means3d_,
            covars_,
            opacities_,
            colors_,
            viewmats,
            Ks,
            empty_rc,
            isect_offsets,
            flatten_ids,
            W,
            H,
            int(ProjectionFamily.PINHOLE_FAMILY),
        )
        # Sum to scalar so gradcheck has a 1D output
        return (rc.sum() + ra.sum()).unsqueeze(0)

    # Loose float32 tolerances; nondet_tol covers atomicAdd in the backward.
    torch.autograd.gradcheck(
        run,
        (means_3d_cond, factors, opacities, colors),
        eps=5e-3,
        atol=0.1,
        rtol=0.1,
        nondet_tol=1e-2,
        raise_exception=True,
    )


def _rasterize_world_python_reference(
    means_3d_cond: torch.Tensor,  # [N, 3]
    covars_uptri: torch.Tensor,  # [N, 6]
    opacities: torch.Tensor,  # [N]
    colors: torch.Tensor,  # [C, N, 3]
    viewmat: torch.Tensor,  # [4, 4]
    K: torch.Tensor,  # [3, 3]
    width: int,
    height: int,
    *,
    cam_id: int = 0,
    front_to_back_indices: torch.Tensor | None = None,  # [N] depth-sorted indices
    ray_fn=None,  # (u_n, v_n) -> camera-frame ray [H, W, 3]; default = pinhole (u_n, v_n, 1)
    valid_fn=None,  # (u_n, v_n) -> [H, W] bool ray validity; default = every ray inverts
    max_alpha: float = 0.99,
    alpha_threshold: float = 1.0 / 255.0,
    t_threshold: float = 1e-4,
    max_kernel_density: float = MAX_RASTER_KERNEL_DENSITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel, per-Gaussian reference with the kernel's cutoffs, early exits and ray-validity gate.

    Returns:
        ``(colors [H, W, 3], alphas [H, W])`` for the requested camera.
    """
    device = means_3d_cond.device
    N = means_3d_cond.shape[0]

    # Camera basis ───────────────────────────────────────────────────────
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam_origin = -(R.T @ t)  # [3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Camera-space depth of each centre — used for the front-to-back ordering only.
    dz_world = (
        R[2, 0] * means_3d_cond[:, 0] + R[2, 1] * means_3d_cond[:, 1] + R[2, 2] * means_3d_cond[:, 2] + t[2]
    )  # [N]

    # The kernel rasterizes the conditioned Σ unmodified: the ray response is evaluated
    # against the primitive's own 3D extent, so nothing here depends on the camera.
    lam = _invert3x3_sym_python(covars_uptri)  # [N, 6]

    # Iterate front-to-back
    if front_to_back_indices is None:
        order = torch.argsort(dz_world)
    else:
        order = front_to_back_indices

    render_colors = torch.zeros(height, width, 3, device=device, dtype=means_3d_cond.dtype)

    # Build per-pixel rays once
    dtype = means_3d_cond.dtype
    py = torch.arange(height, device=device, dtype=dtype).view(-1, 1).expand(height, width)
    px = torch.arange(width, device=device, dtype=dtype).view(1, -1).expand(height, width)
    u_n = (px + 0.5 - cx) / fx
    v_n = (py + 0.5 - cy) / fy
    if ray_fn is None:
        dc = torch.stack([u_n, v_n, torch.ones_like(u_n)], dim=-1)  # [H, W, 3] pinhole ray
    else:
        dc = ray_fn(u_n, v_n)  # [H, W, 3]
    dw = dc @ R  # [H, W, 3]   (dc · R since R[:, j] is camera-frame basis j in world)
    dw = dw / dw.norm(dim=-1, keepdim=True)

    T = torch.ones(height, width, device=device, dtype=means_3d_cond.dtype)
    # A pixel whose ray does not invert starts "done": it never enters the compositing
    # loop, so it keeps T = 1 (alpha 0) and zero colour.
    done = torch.zeros(height, width, device=device, dtype=torch.bool)
    if valid_fn is not None:
        done = ~valid_fn(u_n, v_n)

    for gi in order.tolist():
        # Mahalanobis a, b, c per pixel for Gaussian gi
        mu = means_3d_cond[gi]  # [3]
        l = lam[gi]  # [6]
        opac = opacities[gi]
        p = cam_origin - mu  # [3]

        # Λ d   [H, W, 3]
        Ld_x = l[0] * dw[..., 0] + l[1] * dw[..., 1] + l[2] * dw[..., 2]
        Ld_y = l[1] * dw[..., 0] + l[3] * dw[..., 1] + l[4] * dw[..., 2]
        Ld_z = l[2] * dw[..., 0] + l[4] * dw[..., 1] + l[5] * dw[..., 2]
        # Λ p   [3]
        Lp = torch.stack(
            [
                l[0] * p[0] + l[1] * p[1] + l[2] * p[2],
                l[1] * p[0] + l[3] * p[1] + l[4] * p[2],
                l[2] * p[0] + l[4] * p[1] + l[5] * p[2],
            ]
        )

        a = dw[..., 0] * Ld_x + dw[..., 1] * Ld_y + dw[..., 2] * Ld_z  # [H, W]
        b = dw[..., 0] * Lp[0] + dw[..., 1] * Lp[1] + dw[..., 2] * Lp[2]  # [H, W]
        c = (p * Lp).sum()  # scalar

        valid = (a > 0.0) & (~done)
        if not valid.any():
            continue

        # Ray, not line: the minimiser t = -b/a lies behind the origin when b > 0, and the ray's closest point is o.
        D2 = torch.where(b > 0, c, (c - b * b / a.clamp(min=1e-20)).clamp(min=0.0))
        max_response = torch.exp(-0.5 * D2)
        alpha = (opac * max_response).clamp(max=max_alpha)

        keep = valid & (alpha >= alpha_threshold) & (max_response > max_kernel_density)
        if not keep.any():
            continue

        next_T = T * (1.0 - alpha)
        # Pixels whose transmittance would drop below threshold are flagged done
        # BEFORE accumulating this Gaussian (matches the kernel's early-exit).
        stop_now = keep & (next_T <= t_threshold)
        accumulate = keep & ~stop_now

        if accumulate.any():
            vis = (alpha * T) * accumulate.float()
            render_colors += vis[..., None] * colors[cam_id, gi][None, None, :]
            T = torch.where(accumulate, next_T, T)

        done = done | stop_now

    render_alphas = 1.0 - T
    return render_colors, render_alphas


def test_rasterize_world_matches_python_reference():
    """The forward kernel matches the per-pixel reference."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from anycam4d.cuda.ext import rasterize_from_world_fwd

    torch.manual_seed(0)
    device = torch.device("cuda")
    N = 24
    C = 1
    H, W = 32, 32
    TILE = 16
    tile_h = math.ceil(H / TILE)
    tile_w = math.ceil(W / TILE)

    # Random-ish Gaussians in front of identity camera
    fx = fy = 80.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device)

    # Means uniformly inside the FOV at varying depths
    depths = torch.linspace(3.0, 7.0, N, device=device)
    angles = torch.linspace(-0.25, 0.25, N, device=device)
    means_3d_cond = torch.stack([angles * depths, angles.flip(0) * depths, depths], dim=-1)

    # Random small PSD covariances (Σ = L Lᵀ with diagonal-dominant L)
    L = torch.zeros(N, 3, 3, device=device)
    L[:, 0, 0] = 0.05 + 0.05 * torch.rand(N, device=device)
    L[:, 1, 1] = 0.05 + 0.05 * torch.rand(N, device=device)
    L[:, 2, 2] = 0.05 + 0.05 * torch.rand(N, device=device)
    L[:, 1, 0] = 0.01 * torch.randn(N, device=device)
    L[:, 2, 0] = 0.01 * torch.randn(N, device=device)
    L[:, 2, 1] = 0.01 * torch.randn(N, device=device)
    Sigma = L @ L.transpose(-1, -2)
    covars = torch.stack(
        [
            Sigma[:, 0, 0],
            Sigma[:, 0, 1],
            Sigma[:, 0, 2],
            Sigma[:, 1, 1],
            Sigma[:, 1, 2],
            Sigma[:, 2, 2],
        ],
        dim=-1,
    ).contiguous()

    opacities = torch.full((N,), 0.6, device=device)
    colors = torch.rand(C, N, 3, device=device)

    # Tile assignment: project each Gaussian centre, assume small radii so every
    # Gaussian covers every tile (forces the kernel to actually evaluate them).
    means2d = torch.zeros(C, N, 2, device=device)
    means2d[0, :, 0] = means_3d_cond[:, 0] * fx / depths + cx
    means2d[0, :, 1] = means_3d_cond[:, 1] * fy / depths + cy
    radii = torch.full((C, N, 2), 4 * TILE, dtype=torch.int32, device=device)

    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths.unsqueeze(0),
        TILE,
        tile_w,
        tile_h,
        packed=False,
        n_images=C,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h).reshape(C, tile_h, tile_w)
    empty_rc = torch.zeros((1, 12), device=device, dtype=torch.float32)  # pinhole fixed-width

    rc_cuda, ra_cuda, _last = rasterize_from_world_fwd(
        means_3d_cond.contiguous(),
        factor_of(Sigma),
        opacities,
        colors.contiguous(),
        viewmat.unsqueeze(0).contiguous(),
        K.unsqueeze(0).contiguous(),
        empty_rc,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        W,
        H,
        int(ProjectionFamily.PINHOLE_FAMILY),
    )

    # Depths increase strictly and every Gaussian is in tile (0, 0), so depth order is kernel order.
    order = torch.argsort(depths)

    rc_ref, ra_ref = _rasterize_world_python_reference(
        means_3d_cond,
        covars,
        opacities,
        colors,
        viewmat,
        K,
        W,
        H,
        cam_id=0,
        front_to_back_indices=order,
    )

    # Float32 accumulation sets the tolerance. A primitive within float epsilon of the discrete 3σ cutoff
    # can be kept by one side and dropped by the other; such pixels are bounded by opacity · cutoff instead.
    straddle_bound = float(opacities.max()) * MAX_RASTER_KERNEL_DENSITY
    dc = (rc_cuda[0] - rc_ref).abs()
    da = (ra_cuda[0] - ra_ref).abs()
    n_straddle = int((dc.amax(dim=-1) > 5e-3).sum())
    err_c = dc.max().item()
    err_a = da.max().item()
    assert n_straddle <= 2, f"{n_straddle} pixels over tolerance — too many to be cutoff straddles"
    assert err_c < straddle_bound, f"color max-err vs python ref = {err_c:.4e} (bound {straddle_bound:.4e})"
    assert err_a < straddle_bound, f"alpha max-err vs python ref = {err_a:.4e} (bound {straddle_bound:.4e})"
    assert dc.mean().item() < 5e-5, f"color mean-err vs python ref = {dc.mean().item():.4e}"


# Spatial axes span `anisotropy` around 0.02 world units; a pixel is ~0.06 wide at depth 5, so 1e3 is sub-pixel thin.
# 600 primitives exceed the 256-thread block, so each tile runs several shared-memory batches.
@pytest.mark.parametrize("n_prims", [48, 600])
@pytest.mark.parametrize("anisotropy", [1.0, 1e2, 1e3])
def test_rasterize_world_backward_matches_reference(anisotropy: float, n_prims: int) -> None:
    """Mean, covariance-factor and opacity gradients match fp64 autograd through the reference, per primitive."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from anycam4d.cuda.autograd import RasterizeWorldFunction

    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    device = torch.device("cuda")
    N, H, W, TILE = n_prims, 32, 32, 16
    tile_h, tile_w = math.ceil(H / TILE), math.ceil(W / TILE)
    fx = fy = 80.0
    K = torch.tensor([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device)

    depths = torch.linspace(3.0, 7.0, N, device=device)
    xy = (torch.rand(N, 2, device=device, generator=g) - 0.5) * 0.3 * depths[:, None]
    means = torch.cat([xy, depths[:, None]], dim=-1)
    R, _ = torch.linalg.qr(torch.randn(N, 3, 3, device=device, generator=g))
    scales = 0.02 * anisotropy ** torch.tensor([0.5, 0.0, -0.5], device=device)
    Sigma = R @ torch.diag_embed(scales.expand(N, 3) ** 2) @ R.transpose(-1, -2)
    factors = factor_of(Sigma).contiguous()
    opacities = (0.2 + 0.4 * torch.rand(N, device=device, generator=g)) * min(1.0, 48 / N)
    colors = torch.rand(1, N, 3, device=device, generator=g)
    w_color = torch.randn(H, W, 3, device=device, generator=g)
    w_alpha = torch.randn(H, W, device=device, generator=g)

    means2d = torch.stack([means[:, 0] * fx / depths + W / 2.0, means[:, 1] * fy / depths + H / 2.0], dim=-1)
    radii = torch.full((1, N, 2), 4 * TILE, dtype=torch.int32, device=device)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d[None],
        radii,
        depths[None],
        TILE,
        tile_w,
        tile_h,
        packed=False,
        n_images=1,
    )
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h).reshape(1, tile_h, tile_w)
    empty_rc = torch.zeros((1, 12), device=device)

    params = [means, factors, opacities]
    fused = [p.clone().requires_grad_(True) for p in params]
    rc, ra = RasterizeWorldFunction.apply(
        *fused,
        colors,
        viewmat[None],
        K[None],
        empty_rc,
        isect_offsets,
        flatten_ids,
        W,
        H,
        int(ProjectionFamily.PINHOLE_FAMILY),
    )
    ((rc[0] * w_color).sum() + (ra[0] * w_alpha).sum()).backward()

    ref = [p.double().requires_grad_(True) for p in params]
    rc_ref, ra_ref = _rasterize_world_python_reference(
        ref[0],
        upper_tri(cov_of(ref[1])),
        ref[2],
        colors.double(),
        viewmat.double(),
        K.double(),
        W,
        H,
        front_to_back_indices=torch.argsort(depths),
    )
    ((rc_ref * w_color).sum() + (ra_ref * w_alpha).sum()).backward()

    print()
    for name, pf, pr in zip(["means", "factors", "opacities"], fused, ref):
        gf, gr = pf.grad.reshape(N, -1), pr.grad.float().reshape(N, -1)
        norm = gr.norm(dim=1)
        rel = (gf - gr).norm(dim=1) / (norm + 1e-3 * norm.median())
        print(f"  {name:10s}: rel err median {rel.median():.2e}  p90 {rel.quantile(0.9):.2e}  max {rel.max():.2e}")
        assert rel.quantile(0.9) < 1e-2, f"{name}: p90 per-primitive relative error {rel.quantile(0.9):.2e}"


def test_rasterize_world_evaluates_the_ray_not_the_line() -> None:
    """A Gaussian behind the camera is not drawn; one enclosing it is evaluated at the camera centre, fwd and bwd."""
    device = torch.device("cuda")
    H, W, TILE = 32, 32, 16
    tile_h, tile_w = math.ceil(H / TILE), math.ceil(W / TILE)
    K = torch.tensor([[40.0, 0, W / 2.0], [0, 40.0, H / 2.0], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device)
    # In front; behind the camera on the axis (the line passes through its centre); enclosing the camera.
    means = torch.tensor([[0.3, -0.2, 4.0], [0.0, 0.0, -2.0], [0.1, 0.05, -0.3]], device=device)
    sig = torch.tensor([[0.2, 0.3, 0.25], [0.3, 0.3, 0.3], [0.5, 0.35, 0.6]], device=device)
    g = torch.Generator(device="cuda").manual_seed(3)
    R, _ = torch.linalg.qr(torch.randn(3, 3, 3, device=device, generator=g))
    Sigma = R @ torch.diag_embed(sig**2) @ R.transpose(-1, -2)
    factors = factor_of(Sigma).contiguous()
    opacities = torch.tensor([0.6, 0.9, 0.5], device=device)
    colors = torch.rand(1, 3, 3, device=device, generator=g)
    order = torch.tensor([2, 0, 1], device=device)  # depth keys must be non-negative for isect_tiles
    keys = torch.empty(3, device=device)
    keys[order] = torch.arange(1.0, 4.0, device=device)

    radii = torch.full((1, 3, 2), 4 * TILE, dtype=torch.int32, device=device)
    centre = torch.full((1, 3, 2), W / 2.0, device=device)
    _, isect_ids, flatten_ids = isect_tiles(centre, radii, keys[None], TILE, tile_w, tile_h, packed=False, n_images=1)
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h).reshape(1, tile_h, tile_w)

    params = [means, factors, opacities]
    fused = [p.clone().requires_grad_(True) for p in params]
    rc, ra = RasterizeWorldFunction.apply(
        *fused,
        colors,
        viewmat[None],
        K[None],
        torch.zeros((1, 12), device=device),
        isect_offsets,
        flatten_ids,
        W,
        H,
        int(ProjectionFamily.PINHOLE_FAMILY),
    )
    w_color = torch.randn(H, W, 3, device=device, generator=g)
    ((rc[0] * w_color).sum() + ra[0].sum()).backward()

    ref = [p.double().requires_grad_(True) for p in params]
    rc_ref, ra_ref = _rasterize_world_python_reference(
        ref[0],
        upper_tri(cov_of(ref[1])),
        ref[2],
        colors.double(),
        viewmat.double(),
        K.double(),
        W,
        H,
        front_to_back_indices=order,
    )
    ((rc_ref * w_color).sum() + ra_ref.sum()).backward()

    torch.testing.assert_close(rc[0].double(), rc_ref, atol=1e-4, rtol=0)
    torch.testing.assert_close(ra[0].double(), ra_ref, atol=1e-4, rtol=0)
    # The line rule would give the behind-camera Gaussian alpha 0.9 at the centre; the ray rule leaves only the others.
    only_front = _rasterize_world_python_reference(
        means[[0, 2]].double(),
        upper_tri(Sigma[[0, 2]]).double(),
        opacities[[0, 2]].double(),
        colors[:, [0, 2]].double(),
        viewmat.double(),
        K.double(),
        W,
        H,
        front_to_back_indices=torch.tensor([1, 0], device=device),
    )[1]
    torch.testing.assert_close(ra[0].double(), only_front, atol=1e-4, rtol=0)
    assert float(ra_ref.detach().min()) > 0.2, "the camera-enclosing Gaussian must draw at every pixel"
    for name, pf, pr in zip(["means", "factors", "opacities"], fused, ref):
        torch.testing.assert_close(pf.grad.double(), pr.grad, rtol=1e-3, atol=1e-4, msg=name)


def test_rasterize_world_handles_scaled_viewmat():
    """A viewmat right-multiplied by ``T_denorm`` has an ``s·R`` block; the kernel must divide out ``s``, not assume a rotation."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from anycam4d.cuda.ext import rasterize_from_world_fwd

    torch.manual_seed(0)
    device = torch.device("cuda")
    N = 16
    C = 1
    H, W = 32, 32
    TILE = 16
    tile_h = math.ceil(H / TILE)
    tile_w = math.ceil(W / TILE)

    fx = fy = 80.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device).unsqueeze(0)

    # Reference scene (normalized-coord Gaussians + standard viewmat)
    means_norm = torch.randn(N, 3, device=device) * 0.1
    means_norm[:, 2] += 4.0
    L = 0.05 * torch.eye(3, device=device).unsqueeze(0).expand(N, 3, 3).clone()
    factors_norm = factor_of(L @ L.transpose(-1, -2)).contiguous()
    opacities = torch.full((N,), 0.7, device=device)
    colors = torch.rand(C, N, 3, device=device)
    viewmat_world = torch.eye(4, device=device).unsqueeze(0)

    # T_denorm = diag(s, s, s): (means / s, viewmat @ T_denorm) must render the same image.
    s = 2.5
    T_denorm = torch.eye(4, device=device)
    T_denorm[:3, :3] *= s
    means_scaled = means_norm / s
    factors_scaled = factors_norm / s
    viewmat_scaled = viewmat_world @ T_denorm.unsqueeze(0)

    # Tiling — every Gaussian assigned to every tile (large radii)
    depths = means_norm[:, 2].unsqueeze(0)
    means2d_px = torch.stack(
        [
            means_norm[:, 0] * fx / means_norm[:, 2] + cx,
            means_norm[:, 1] * fy / means_norm[:, 2] + cy,
        ],
        dim=-1,
    ).unsqueeze(0)
    radii = torch.full((C, N, 2), 4 * TILE, dtype=torch.int32, device=device)
    _, isect_ids, flatten_ids = isect_tiles(means2d_px, radii, depths, TILE, tile_w, tile_h, packed=False, n_images=C)
    isect_offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h).reshape(C, tile_h, tile_w)
    empty = torch.zeros((C, 12), device=device, dtype=torch.float32)  # pinhole fixed-width

    rc_ref, ra_ref, _ = rasterize_from_world_fwd(
        means_norm.contiguous(),
        factors_norm,
        opacities,
        colors.contiguous(),
        viewmat_world.contiguous(),
        K.contiguous(),
        empty,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        W,
        H,
        int(ProjectionFamily.PINHOLE_FAMILY),
    )
    rc_scaled, ra_scaled, _ = rasterize_from_world_fwd(
        means_scaled.contiguous(),
        factors_scaled,
        opacities,
        colors.contiguous(),
        viewmat_scaled.contiguous(),
        K.contiguous(),
        empty,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        W,
        H,
        int(ProjectionFamily.PINHOLE_FAMILY),
    )

    err_c = (rc_ref - rc_scaled).abs().max().item()
    err_a = (ra_ref - ra_scaled).abs().max().item()
    assert err_c < 1e-3, f"scaled-viewmat color drift = {err_c:.4e} — kernel does not handle T_denorm correctly"
    assert err_a < 1e-3, f"scaled-viewmat alpha drift = {err_a:.4e} — kernel does not handle T_denorm correctly"


def test_ut_backend_fused_matches_unfused(random_model):
    """The fused UT kernel matches the three-stage reference (condition, SH, world rasterize)."""
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.enums import BackendType
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend
    from tests.oracles.ut_pytorch_backend import UTPyTorchBackend

    device = torch.device("cuda")
    model = random_model.to(device)

    cfg = RenderingConfig(backend=BackendType.ANYCAM4D)
    backend_fused = UTFusedBackend(cfg)
    backend_unfused = UTPyTorchBackend(cfg)

    C = 2
    H, W = 64, 64
    torch.manual_seed(0)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 5.0

    Ks = torch.zeros(C, 3, 3, device=device)
    Ks[:, 0, 0] = 400.0
    Ks[:, 1, 1] = 400.0
    Ks[:, 0, 2] = W / 2.0
    Ks[:, 1, 2] = H / 2.0
    Ks[:, 2, 2] = 1.0

    camera = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, W, H)
    out_fused = backend_fused.render(model, camera, tau=0.0, sh_degree=1)
    out_unfused = backend_unfused.render(model, camera, tau=0.0, sh_degree=1)

    err_c = (out_fused.colors - out_unfused.colors).abs().max().item()
    err_a = (out_fused.alphas - out_unfused.alphas).abs().max().item()
    assert err_c < 1e-4, f"fused/unfused color drift = {err_c:.4e}"
    assert err_a < 1e-4, f"fused/unfused alpha drift = {err_a:.4e}"


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="UT packed render runs on the CUDA kernels")
def test_ut_packed_matches_dense_multi_camera(random_model: FourDGaussians) -> None:
    """Packed and dense UT agree at C>1; a packed index without ``cam*N`` would read camera 0's colours."""
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.enums import BackendType
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend

    device = torch.device("cuda")
    model = random_model.to(device)

    C = 3
    H, W = 64, 64
    torch.manual_seed(0)
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 5.0
    viewmats[:, 0, 3] = torch.linspace(-0.6, 0.6, C, device=device)  # distinct centres → per-view color
    Ks = torch.zeros(C, 3, 3, device=device)
    Ks[:, 0, 0] = Ks[:, 1, 1] = 400.0
    Ks[:, 0, 2] = W / 2.0
    Ks[:, 1, 2] = H / 2.0
    Ks[:, 2, 2] = 1.0
    camera = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, W, H)

    with torch.no_grad():
        dense = UTFusedBackend(RenderingConfig(backend=BackendType.ANYCAM4D, packed=False)).render(
            model, camera, tau=0.0, sh_degree=1
        )
        packed = UTFusedBackend(RenderingConfig(backend=BackendType.ANYCAM4D, packed=True)).render(
            model, camera, tau=0.0, sh_degree=1
        )

    torch.testing.assert_close(packed.colors, dense.colors, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(packed.alphas, dense.alphas, rtol=1e-4, atol=1e-5)


def _chol_alpha_python(
    ray_origin: torch.Tensor,  # [3]
    ray_dir: torch.Tensor,  # [3] unit
    means_3d_cond: torch.Tensor,  # [N, 3]
    sigma_uptri: torch.Tensor,  # [N, 6]
    opacities: torch.Tensor,  # [N]
) -> torch.Tensor:
    """Reference alpha via Cholesky whitening, as the kernel computes it (no explicit inverse)."""
    n = means_3d_cond.shape[0]
    s = sigma_uptri
    sigma_full = torch.stack(
        [
            s[:, 0],
            s[:, 1],
            s[:, 2],
            s[:, 1],
            s[:, 3],
            s[:, 4],
            s[:, 2],
            s[:, 4],
            s[:, 5],
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    chol = torch.linalg.cholesky(sigma_full)  # [N, 3, 3] lower
    p = (ray_origin[None, :] - means_3d_cond).unsqueeze(-1)
    d = ray_dir[None, :].expand(n, 3).unsqueeze(-1)
    yp = torch.linalg.solve_triangular(chol, p, upper=False).squeeze(-1)
    yd = torch.linalg.solve_triangular(chol, d, upper=False).squeeze(-1)
    a = (yd * yd).sum(-1)
    b = (yd * yp).sum(-1)
    c = (yp * yp).sum(-1)
    d2 = c - b**2 / a.clamp(min=1e-30)
    return opacities * torch.exp(-0.5 * d2)


def test_cholesky_whitening_matches_precision_and_whitening_references():
    """Cholesky, explicit-precision and gsplat whitening give the same alpha to float tolerance."""
    torch.manual_seed(42)
    n = 128
    device = torch.device("cuda")

    quats = F.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.exp(torch.randn(n, 3, device=device) * 0.3).clamp(0.01, 2.0)
    means_3d_cond = torch.randn(n, 3, device=device)
    means_3d_cond[:, 2] += 4.0
    opacities = torch.sigmoid(torch.randn(n, device=device)) * 0.9 + 0.1

    rot = torch.stack(
        [
            (1 - 2 * (quats[:, 2] ** 2 + quats[:, 3] ** 2)),
            2 * (quats[:, 1] * quats[:, 2] - quats[:, 3] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 3] + quats[:, 2] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 2] + quats[:, 3] * quats[:, 0]),
            (1 - 2 * (quats[:, 1] ** 2 + quats[:, 3] ** 2)),
            2 * (quats[:, 2] * quats[:, 3] - quats[:, 1] * quats[:, 0]),
            2 * (quats[:, 1] * quats[:, 3] - quats[:, 2] * quats[:, 0]),
            2 * (quats[:, 2] * quats[:, 3] + quats[:, 1] * quats[:, 0]),
            (1 - 2 * (quats[:, 1] ** 2 + quats[:, 2] ** 2)),
        ],
        dim=-1,
    ).reshape(n, 3, 3)
    chol = rot * scales[:, None, :]
    sigma_full = chol @ chol.transpose(-1, -2)
    sigma_uptri = torch.stack(
        [
            sigma_full[:, 0, 0],
            sigma_full[:, 0, 1],
            sigma_full[:, 0, 2],
            sigma_full[:, 1, 1],
            sigma_full[:, 1, 2],
            sigma_full[:, 2, 2],
        ],
        dim=-1,
    )
    lambda_uptri = _invert3x3_sym_python(sigma_uptri)

    ray_o = torch.tensor([0.1, -0.2, 0.0], device=device)
    ray_d = F.normalize(torch.tensor([0.0, 0.0, 1.0], device=device), dim=0)

    alpha_chol = _chol_alpha_python(ray_o, ray_d, means_3d_cond, sigma_uptri, opacities)
    alpha_mahal = _mahal_alpha_python(ray_o, ray_d, means_3d_cond, lambda_uptri, opacities)
    alpha_whitening = _whitening_alpha_python(ray_o, ray_d, means_3d_cond, quats, scales, opacities)

    err_mahal = (alpha_chol - alpha_mahal).abs().max().item()
    err_whitening = (alpha_chol - alpha_whitening).abs().max().item()
    assert err_mahal < 1e-4, f"Cholesky vs precision α differ: {err_mahal:.2e}"
    assert err_whitening < 1e-4, f"Cholesky vs whitening α differ: {err_whitening:.2e}"


@pytest.mark.filterwarnings(
    "ignore:Input #\\d+ requires gradient and is not a double:UserWarning"
)  # float32-only kernel
def test_rasterize_world_gradcheck_anisotropic():
    """gradcheck on an anisotropic Σ (cond ~ 8) — exercises the rewritten
    (Λv)(Λv)ᵀ Σ-gradient and the substitution-based Λp/Λd mean gradient on a
    non-isotropic covariance, complementing the near-isotropic baseline gradcheck."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from anycam4d.cuda.autograd import RasterizeWorldFunction

    torch.manual_seed(7)
    device = torch.device("cuda")
    n, c_cams, h, w, tile, cdim = 4, 1, 8, 8, 16, 3
    tile_h = math.ceil(h / tile)
    tile_w = math.ceil(w / tile)
    dtype = torch.float32

    # Anisotropic: 0.4 / 0.4 / 0.05 axes (cond ~ 8) + small off-diagonals.
    covars = torch.zeros(n, 6, dtype=dtype, device=device)
    covars[:, 0] = 0.4
    covars[:, 3] = 0.4
    covars[:, 5] = 0.05
    covars[:, 1] = 0.02 * torch.randn(n, device=device, dtype=dtype)
    covars[:, 2] = 0.02 * torch.randn(n, device=device, dtype=dtype)
    covars[:, 4] = 0.02 * torch.randn(n, device=device, dtype=dtype)
    factors = factor_of(from_upper_tri(covars)).requires_grad_(True)

    fx = fy = 100.0
    cx, cy = w / 2.0, h / 2.0
    z0 = 5.0
    mu_xy = torch.tensor(
        [
            [cx - 1.5, cy - 1.5],
            [cx + 1.5, cy - 1.5],
            [cx - 1.5, cy + 1.5],
            [cx + 1.5, cy + 1.5],
        ],
        dtype=dtype,
        device=device,
    )
    mu_world = torch.zeros(n, 3, dtype=dtype, device=device)
    mu_world[:, 0] = (mu_xy[:, 0] - cx) / fx * z0
    mu_world[:, 1] = (mu_xy[:, 1] - cy) / fy * z0
    mu_world[:, 2] = z0
    means_3d_cond = mu_world.requires_grad_(True)
    opacities = torch.full((n,), 0.8, dtype=dtype, device=device).requires_grad_(True)
    colors = torch.rand(c_cams, n, cdim, dtype=dtype, device=device).requires_grad_(True)
    viewmats = torch.eye(4, dtype=dtype, device=device).unsqueeze(0)
    Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device).unsqueeze(0)
    radii_f = torch.ones(n, 2, dtype=torch.int32, device=device) * 3
    depths_f = torch.full((n,), z0, dtype=torch.float32, device=device)
    _, isect_ids, flatten_ids = isect_tiles(
        mu_xy.unsqueeze(0),
        radii_f.unsqueeze(0),
        depths_f.unsqueeze(0),
        tile,
        tile_w,
        tile_h,
        packed=False,
        n_images=c_cams,
    )
    isect_offsets = isect_offset_encode(isect_ids, c_cams, tile_w, tile_h).reshape(c_cams, tile_h, tile_w)
    empty_rc = torch.zeros((c_cams, 12), device=device, dtype=dtype)

    def run(means3d_, covars_, opacities_, colors_):
        rc, ra = RasterizeWorldFunction.apply(
            means3d_,
            covars_,
            opacities_,
            colors_,
            viewmats,
            Ks,
            empty_rc,
            isect_offsets,
            flatten_ids,
            w,
            h,
            int(ProjectionFamily.PINHOLE_FAMILY),
        )
        return (rc.sum() + ra.sum()).unsqueeze(0)

    torch.autograd.gradcheck(
        run,
        (means_3d_cond, factors, opacities, colors),
        eps=5e-3,
        atol=0.1,
        rtol=0.1,
        nondet_tol=1e-2,
        raise_exception=True,
    )


def test_residual_d2_beats_scalar_d2_for_grazing_rays():
    """The residual form of D² is at least 10x more accurate than ``‖y_p‖² − b²/a`` on grazing rays in float32."""
    torch.manual_seed(3)
    device = torch.device("cuda")
    n = 40000
    ratio = 100.0  # axis ratio → κ(Σ) ~ 1e4

    quats = F.normalize(torch.randn(n, 4, device=device, dtype=torch.float64), dim=-1)
    scales = torch.tensor([1.0, 1.0 / ratio**0.5, 1.0 / ratio], device=device, dtype=torch.float64).expand(n, 3)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rot = torch.stack(
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
    ).reshape(n, 3, 3)
    chol_l = rot * scales[:, None, :]
    sigma = chol_l @ chol_l.transpose(-1, -2)

    p = torch.randn(n, 3, device=device, dtype=torch.float64) * 0.5
    d = F.normalize(torch.randn(n, 3, device=device, dtype=torch.float64), dim=-1)

    def whiten(dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        chol = torch.linalg.cholesky(sigma.to(dt))
        yp = torch.linalg.solve_triangular(chol, p.to(dt).unsqueeze(-1), upper=False).squeeze(-1)
        yd = torch.linalg.solve_triangular(chol, d.to(dt).unsqueeze(-1), upper=False).squeeze(-1)
        return yd, yp

    def d2_scalar(yd: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        a = (yd * yd).sum(-1)
        b = (yd * yp).sum(-1)
        c = (yp * yp).sum(-1)
        return torch.clamp(c - b * b / a.clamp(min=1e-30), min=0.0)

    def d2_residual(yd: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        a = (yd * yd).sum(-1)
        b = (yd * yp).sum(-1)
        r = yp - (b / a.clamp(min=1e-30)).unsqueeze(-1) * yd
        return (r * r).sum(-1)

    yd64, yp64 = whiten(torch.float64)
    truth = d2_residual(yd64, yp64)
    mask = (truth > 1e-6) & (truth < 20.0)  # regime where α is actually sensitive

    yd32, yp32 = whiten(torch.float32)
    err_scalar = ((d2_scalar(yd32, yp32).double() - truth).abs() / truth.clamp(min=1e-12))[mask].max().item()
    err_resid = ((d2_residual(yd32, yp32).double() - truth).abs() / truth.clamp(min=1e-12))[mask].max().item()
    print(f"\n D² rel-err vs f64 truth (κ~1e4): scalar={err_scalar:.3e}, residual={err_resid:.3e}")
    assert err_resid < err_scalar / 10.0, (
        f"residual D² should be ≥10× better: residual={err_resid:.3e} scalar={err_scalar:.3e}"
    )
