"""fp64 reference for EWA through a lens: the lens map in its textbook form, J by autodiff at the clamped point.

Independent of the kernel's ψ(u) formulation and hand-derived Hessian: the map is OpenCV's Brown–Conrady on x/z or
Kannala–Brandt on θ = atan2(r, z), and J = ∂(lens map)/∂x_c comes from ``torch.func.jacrev``, so the mean gradient
through J is autograd's second derivative.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.func import jacrev
from torch.func import vmap

ALPHA_THRESHOLD = 1.0 / 255.0


def brown_conrady(x_c: Tensor, d: Tensor) -> Tensor:
    """Normalised distorted point of camera point(s) ``[..., 3]``; 12-slot OpenCV layout."""
    u, v = x_c[..., 0] / x_c[..., 2], x_c[..., 1] / x_c[..., 2]
    r2 = u * u + v * v
    icd = (1 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))) / (1 + r2 * (d[5] + r2 * (d[6] + r2 * d[7])))
    dx = 2 * d[2] * u * v + d[3] * (r2 + 2 * u * u) + r2 * (d[8] + r2 * d[9])
    dy = d[2] * (r2 + 2 * v * v) + 2 * d[3] * u * v + r2 * (d[10] + r2 * d[11])
    return torch.stack([icd * u + dx, icd * v + dy], dim=-1)


def kannala_brandt(x_c: Tensor, d: Tensor) -> Tensor:
    """Normalised distorted point; θd = θ(1 + k1θ² + …), scale θd/r (r → 0 by the ratio's own limit in fp64)."""
    r = torch.sqrt(x_c[..., 0] ** 2 + x_c[..., 1] ** 2)
    th = torch.atan2(r, x_c[..., 2])
    t2 = th * th
    thd = th * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
    scale = thd / r
    return torch.stack([scale * x_c[..., 0], scale * x_c[..., 1]], dim=-1)


def bc_monotone(x_c: Tensor, d: Tensor) -> Tensor:
    """Brown–Conrady radial domain: den > 0 and d r_d / d r > 0, written on r_d(r) = r·N/D directly."""
    r2 = (x_c[..., 0] ** 2 + x_c[..., 1] ** 2) / x_c[..., 2] ** 2
    r = torch.sqrt(r2).detach().requires_grad_(True)
    s = r * r
    rd = r * (1 + s * (d[0] + s * (d[1] + s * d[4]))) / (1 + s * (d[5] + s * (d[6] + s * d[7])))
    (slope,) = torch.autograd.grad(rd.sum(), r)
    den = 1 + r2 * (d[5] + r2 * (d[6] + r2 * d[7]))
    return (den > 1e-8) & (slope > 0)


def kb_monotone(x_c: Tensor, d: Tensor) -> Tensor:
    """Kannala–Brandt domain: dθd/dθ > 0 at the point's incidence angle."""
    th = torch.atan2(torch.sqrt(x_c[..., 0] ** 2 + x_c[..., 1] ** 2), x_c[..., 2]).detach().requires_grad_(True)
    t2 = th * th
    thd = th * (1 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
    (slope,) = torch.autograd.grad(thd.sum(), th)
    return slope > 0


def bc_clamp_limits(K: Tensor, d: Tensor, width: int, height: int, n_edge: int = 257) -> Tensor:
    """[x_pos, x_neg, y_pos, y_neg]: the undistorted image border's extent per side plus 0.3 of the half extent."""
    t = np.linspace(0.0, 1.0, n_edge)
    u = np.concatenate([t * width, t * width, np.zeros(n_edge), np.full(n_edge, float(width))])
    v = np.concatenate([np.zeros(n_edge), np.full(n_edge, float(height)), t * height, t * height])
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    p = cv2.undistortPointsIter(np.stack([u, v], -1)[:, None], K.numpy(), d.numpy(), None, None, crit)[:, 0]
    p = torch.from_numpy(p)
    # Past the largest distorted radius there is no inner-branch preimage: keep only points that reproject.
    target = torch.from_numpy(
        np.stack([(u - K[0, 2].item()) / K[0, 0].item(), (v - K[1, 2].item()) / K[1, 1].item()], -1)
    )
    reproj = brown_conrady(torch.cat([p, torch.ones(len(p), 1, dtype=p.dtype)], dim=1), d)
    p = p[(reproj - target).abs().sum(dim=1) < 1e-4]
    # Inner branch only: monotone on all of [0, r], not just at r (the polynomial can turn monotone again).
    frac = torch.linspace(1 / 32, 1.0, 32, dtype=p.dtype)
    rays = torch.cat([p[:, None, :] * frac[None, :, None], torch.ones(len(p), 32, 1, dtype=p.dtype)], dim=-1)
    p = p[bc_monotone(rays.reshape(-1, 3), d).reshape(len(p), 32).all(dim=1)]
    if len(p) == 0:  # no border point on the inner branch: nothing on screen needs clamping
        return torch.full((4,), torch.inf, dtype=torch.float64)
    ext = torch.stack([p[:, 0].max(), -p[:, 0].min(), p[:, 1].max(), -p[:, 1].min()])
    margin_x, margin_y = 0.3 * 0.5 * (ext[0] + ext[1]), 0.3 * 0.5 * (ext[2] + ext[3])
    return ext + torch.stack([margin_x, margin_x, margin_y, margin_y])


@dataclass
class LensEwaOut:
    """Per-pair outputs; ``visible`` is the cull mask, the rest are meaningful where it is set."""

    means2d: Tensor
    conics: Tensor
    opac: Tensor
    radii: Tensor
    visible: Tensor


def lens_ewa_project(
    x_c: Tensor,
    cov_c: Tensor,
    opacity: Tensor,
    K: Tensor,
    dist: Tensor,
    fisheye: bool,
    width: int,
    height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    calc_compensations: bool,
) -> LensEwaOut:
    """EWA of camera-space Gaussians ``x_c [M, 3]``, ``cov_c [M, 3, 3]`` through one lens; differentiable."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    lens = (lambda p: kannala_brandt(p, dist)) if fisheye else (lambda p: brown_conrady(p, dist))
    f = torch.stack([fx, fy])
    c = torch.stack([cx, cy])

    z = x_c[:, 2]
    in_depth = (z >= near_plane) & (z <= far_plane)
    # Culled rows only need finite values, also in the backward: off-axis, since KB's θd/r is 0/0 at r = 0.
    x_safe = torch.where(in_depth[:, None], x_c, torch.tensor([0.1, 0.0, 1.0], dtype=x_c.dtype, device=x_c.device))
    domain = kb_monotone(x_safe, dist) if fisheye else bc_monotone(x_safe, dist)

    means2d = lens(x_safe) * f + c

    x_j = x_safe
    if not fisheye:  # persp_proj's frustum clamp at the undistorted image extent, J evaluated at the clamped point
        lim = bc_clamp_limits(K, dist, width, height)
        zj = x_safe[:, 2]
        px = torch.clamp(x_safe[:, 0] / zj, -lim[1].item(), lim[0].item())
        py = torch.clamp(x_safe[:, 1] / zj, -lim[3].item(), lim[2].item())
        x_j = torch.stack([px * zj, py * zj, zj], dim=-1)
    J = vmap(jacrev(lens))(x_j) * f[None, :, None]  # [M, 2, 3]

    cov2d = J @ cov_c @ J.transpose(1, 2)
    det_raw = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    cov2d = cov2d + eps2d * torch.eye(2, dtype=x_c.dtype, device=x_c.device)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    comp = torch.sqrt(torch.clamp(det_raw / det, min=0.0)) if calc_compensations else torch.ones_like(det)
    det_ok = det > 0

    extend = torch.clamp(
        torch.sqrt(2 * torch.log(torch.clamp(opacity, min=ALPHA_THRESHOLD) / ALPHA_THRESHOLD)), max=3.33
    )
    radii = torch.ceil(extend[:, None] * torch.sqrt(torch.stack([cov2d[:, 0, 0], cov2d[:, 1, 1]], -1).clamp(min=0)))
    m = means2d.detach()
    on_screen = (m[:, 0] + radii[:, 0] > 0) & (m[:, 0] - radii[:, 0] < width)
    on_screen &= (m[:, 1] + radii[:, 1] > 0) & (m[:, 1] - radii[:, 1] < height)

    visible = in_depth & domain & det_ok & (opacity >= ALPHA_THRESHOLD) & on_screen
    inv = torch.linalg.inv(torch.where(det_ok[:, None, None], cov2d, torch.eye(2, dtype=x_c.dtype, device=x_c.device)))
    conics = torch.stack([inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 1]], dim=-1)
    return LensEwaOut(means2d, conics, opacity * comp, radii.to(torch.int32), visible)
