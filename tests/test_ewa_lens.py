"""Fused EWA through Brown–Conrady and Kannala–Brandt lenses against the fp64 autodiff oracle, by field stratum."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.covariance import schur_condition
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.appearance import APPEARANCE_CUDA_ID
from anycam4d.enums import AppearanceType
from tests.oracles.ewa_lens import lens_ewa_project


pytestmark = pytest.mark.skipif(
    not (HAS_CUDA_EXT and torch.cuda.is_available()), reason="fused EWA needs the CUDA extension"
)

_PINHOLE, _FISHEYE = 0, 1
_W, _H = 256, 192
_TAU, _COV_EPS, _EPS2D, _NEAR, _FAR = 0.4, 1e-6, 0.3, 0.01, 1e10
_PARAMS = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities"]

# 12-slot barrel with every coefficient live; radial fold at r ≈ 1.33. Its undistorted border puts the +y clamp
# limit at 1.08, inside the fold; the x limits lie past it.
_BC = [-0.19, 0.01, 0.02, -0.015, 0.0, 0.01, 0.0, 0.0, 0.01, -0.005, 0.008, 0.003]
_BC_FOLD = 1.329
# Kannala–Brandt with its fold dθd/dθ = 0 at θ ≈ 1.33 rad (76°), inside the image.
_KB = [-0.1, -0.03, 0.0, 0.0]
_KB_FOLD = 1.330

# Incidence-angle bands per family: the KB θ/r limit, mid field, image corner, just inside the fold, past the
# BC frustum clamp (a centre off-screen whose footprint is not).
_STRATA = {
    _PINHOLE: {
        "axis": (1e-5, 3e-3),
        "mid": (0.2, 0.5),
        "corner": (0.6, 0.8),
        "fold": (math.atan(0.93 * _BC_FOLD), math.atan(0.99 * _BC_FOLD)),
        "clamp": (math.atan(1.12), math.atan(1.27)),
    },
    _FISHEYE: {
        "axis": (1e-5, 3e-3),
        "mid": (0.2, 0.5),
        "corner": (0.8, 1.1),
        "fold": (0.93 * _KB_FOLD, 0.99 * _KB_FOLD),
    },
}
_CASES = [(fam, s) for fam, strata in _STRATA.items() for s in strata]


def _camera(family: int) -> tuple[Tensor, Tensor, Tensor]:
    """Two cameras at one pose; the second has half the distortion, so a per-camera index bug shows."""
    g = torch.Generator().manual_seed(3)
    R, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    R = R * torch.sign(torch.det(R))
    vm = torch.eye(4, dtype=torch.float64)
    vm[:3, :3] = R
    vm[:3, 3] = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    f = 160.0 if family == _PINHOLE else 100.0
    K = torch.tensor([[f, 0.0, 128.3], [0.0, f * 1.01, 95.7], [0.0, 0.0, 1.0]], dtype=torch.float64)
    d = torch.tensor(_BC if family == _PINHOLE else _KB, dtype=torch.float64)
    return vm.expand(2, 4, 4).clone(), K.expand(2, 3, 3).clone(), torch.stack([d, 0.5 * d])


def _population(family: int, stratum: str, viewmat: Tensor, n: int = 2048) -> dict[str, Tensor]:
    """Gaussians whose centres sit in one incidence band of the first camera."""
    g = torch.Generator().manual_seed(17 + 7 * family + list(_STRATA[family]).index(stratum))
    lo, hi = _STRATA[family][stratum]
    th = lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)
    phi = 2 * math.pi * torch.rand(n, generator=g, dtype=torch.float64)
    if stratum == "clamp":  # along +y, the side whose limit is inside the fold
        phi = 0.5 * math.pi + 0.2 * (phi / math.pi - 1)
    if stratum == "corner":
        phi = math.atan2(_H, _W) + 0.5 * math.pi * torch.randint(0, 4, (n,), generator=g).double()
    depth = 1.5 + 2.5 * torch.rand(n, generator=g, dtype=torch.float64)
    x_c = depth[:, None] * torch.stack([th.sin() * phi.cos(), th.sin() * phi.sin(), th.cos()], dim=-1)
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    means = (x_c - t) @ R  # world = Rᵀ (x_c − t)

    big = 0.6 if stratum == "clamp" else 0.0  # off-screen centres need a footprint that reaches the image
    spatial = math.log(0.03) + big + 0.6 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    return {
        "quats_l": torch.randn(n, 4, generator=g, dtype=torch.float64),
        "quats_r": torch.randn(n, 4, generator=g, dtype=torch.float64),
        "scales_4d_log": torch.cat([spatial, torch.full((n, 1), math.log(0.5), dtype=torch.float64)], dim=1),
        "means": means,
        "mu_t": _TAU + 0.05 * torch.randn(n, generator=g, dtype=torch.float64),
        "opacities": torch.randn(n, generator=g, dtype=torch.float64) + 1.0,
    }


def _fused(pop: dict[str, Tensor], vm: Tensor, Ks: Tensor, dist: Tensor, family: int, packed: bool, aa: bool):
    """CUDA fp32 outputs as dense [C, N] tensors, visibility, and a closure for the backward."""
    from anycam4d.cuda.autograd import EWAFusedFunction
    from anycam4d.cuda.autograd import EWAFusedPackedFunction

    params = [pop[k].float().cuda().requires_grad_(True) for k in _PARAMS]
    n = params[0].shape[0]
    sh = (torch.zeros(n, 1, 3, device="cuda"), torch.zeros(n, 0, 3, device="cuda"), torch.zeros(n, 0, 3, device="cuda"))
    scalars = (
        0,
        APPEARANCE_CUDA_ID[AppearanceType.SH4D_ADDITIVE],
        0,
        0,
        vm.float().cuda(),
        Ks.float().cuda(),
        dist.float().cuda(),
        family,
        _TAU,
        _COV_EPS,
        _EPS2D,
        _NEAR,
        _FAR,
        0.0,
        _W,
        _H,
        aa,
    )
    C = vm.shape[0]
    if packed:
        cam, gid, _, m2d, con, _, opac, _, radii, _, _ = EWAFusedPackedFunction.apply(*params, *sh, *scalars)
        cam, gid = cam.long(), gid.long()
        out = {}
        for name, val, w in [("means2d", m2d, 2), ("conics", con, 3), ("opac", opac, 1), ("radii", radii, 2)]:
            dense = torch.zeros(C, n, w, dtype=val.dtype, device="cuda")
            dense = dense.index_put((cam, gid), val.reshape(-1, w))
            out[name] = dense.squeeze(-1) if w == 1 else dense
        vis = torch.zeros(C, n, dtype=torch.bool, device="cuda")
        vis[cam, gid] = True
    else:
        m2d, con, _, opac, _, radii, _, _ = EWAFusedFunction.apply(*params, *sh, *scalars)
        out = {"means2d": m2d, "conics": con, "opac": opac, "radii": radii}
        vis = (radii > 0).all(dim=-1)
    return params, out, vis


def _oracle(pop: dict[str, Tensor], vm: Tensor, Ks: Tensor, dist: Tensor, family: int, aa: bool):
    """fp64 reference: conditioning, world → camera, lens EWA per camera."""
    params = [pop[k].clone().requires_grad_(True) for k in _PARAMS]
    ql, qr, s4d, mn, mt, op = params
    cov4 = build_cov4d_rot(ql, qr, s4d)
    mean_corr, cov3 = schur_condition(cov4, _TAU, mt, eps=_COV_EPS)
    wt = torch.exp(-0.5 * (_TAU - mt) ** 2 / (cov4[:, 3, 3] + TEMPORAL_VARIANCE_EPS))
    opacity = torch.sigmoid(op) * wt
    outs = []
    for c in range(vm.shape[0]):
        R, t = vm[c, :3, :3], vm[c, :3, 3]
        x_c = (mn + mean_corr) @ R.T + t
        cov_c = R @ cov3 @ R.T
        outs.append(
            lens_ewa_project(x_c, cov_c, opacity, Ks[c], dist[c], family == _FISHEYE, _W, _H, _EPS2D, _NEAR, _FAR, aa)
        )
    stack = {k: torch.stack([getattr(o, k) for o in outs]) for k in ("means2d", "conics", "opac", "radii", "visible")}
    return params, stack


def _rel_p99(got: Tensor, ref: Tensor) -> float:
    """p99 over primitives of the per-primitive relative error, floored at 1e-3 of the median norm."""
    got, ref = got.reshape(got.shape[0], -1), ref.reshape(ref.shape[0], -1)
    norm = ref.norm(dim=1)
    rel = (got - ref).norm(dim=1) / (norm + 1e-3 * norm.median() + 1e-30)
    return torch.quantile(rel, 0.99).item()


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("aa", [False, True], ids=["classic", "aa"])
@pytest.mark.parametrize(
    ("family", "stratum"), _CASES, ids=[f"{'bc' if f == _PINHOLE else 'kb'}-{s}" for f, s in _CASES]
)
def test_lens_ewa_matches_fp64_oracle(family: int, stratum: str, packed: bool, aa: bool) -> None:
    """mean2d, conic, radii, opacity and every input gradient match the fp64 oracle on the pairs both draw."""
    vm, Ks, dist = _camera(family)
    pop = _population(family, stratum, vm[0])
    p_f, out_f, vis_f = _fused(pop, vm, Ks, dist, family, packed, aa)
    p_r, out_r = _oracle(pop, vm, Ks, dist, family, aa)
    vis_r = out_r["visible"].cuda()

    both = vis_f & vis_r
    assert both[0].sum() > 200, f"too few visible pairs in stratum {stratum}: {int(both[0].sum())}"
    disagree = (vis_f ^ vis_r).float().mean().item()
    assert disagree < 5e-3, f"visibility disagrees on {disagree:.2%} of pairs"

    ref = {k: v.cuda() for k, v in out_r.items()}
    assert (out_f["means2d"][both] - ref["means2d"][both]).abs().max() < 2e-3
    assert _rel_p99(out_f["conics"][both].double(), ref["conics"][both]) < 1e-3
    assert (out_f["radii"][both] - ref["radii"][both]).abs().max() <= 1
    # At the fold the footprint degenerates to a line and det(Σ2d_raw) cancels in fp32: the oracle evaluated in
    # fp32 is itself 9e-4 off there, so the compensated opacity gets that floor.
    opac_atol = 1e-3 if stratum == "fold" else 1e-4
    torch.testing.assert_close(out_f["opac"][both].double(), ref["opac"][both], atol=opac_atol, rtol=1e-4)

    g = torch.Generator(device="cuda").manual_seed(5)
    weights = {
        k: torch.randn(out_f[k].shape, device="cuda", generator=g, dtype=torch.float64)
        for k in ("means2d", "conics", "opac")
    }
    # Conics are ~1/σ² and dwarf the rest; scale them so every output reaches the gradient.
    weights["conics"] *= 1e-2
    loss_f = sum((out_f[k].double() * w)[both].sum() for k, w in weights.items())
    loss_r = sum((ref[k] * w)[both].sum() for k, w in weights.items())
    loss_f.backward()
    loss_r.backward()
    for name, a, b in zip(_PARAMS, p_f, p_r, strict=True):
        err = _rel_p99(a.grad.double(), b.grad.cuda())
        assert err < 1e-2, f"{name}: p99 relative gradient error {err:.2e}"


@pytest.mark.parametrize("stratum", ["mid", "corner"])
def test_equidistant_matches_gsplat_fisheye_ewa(stratum: str) -> None:
    """KB with k = 0 is gsplat 1.5.3's fisheye EWA: mean2d, conic, radii and gradients agree (gsplat's ε fudges
    make it inexact on the axis, so the comparison stays off it)."""
    from gsplat.cuda._wrapper import fully_fused_projection

    vm, Ks, _ = _camera(_FISHEYE)
    dist = torch.zeros(2, 4, dtype=torch.float64)
    pop = _population(_FISHEYE, stratum, vm[0])
    p_f, out_f, vis_f = _fused(pop, vm, Ks, dist, _FISHEYE, False, False)

    params = [pop[k].float().cuda().requires_grad_(True) for k in _PARAMS]
    ql, qr, s4d, mn, mt, op = params
    cov4 = build_cov4d_rot(ql, qr, s4d)
    mean_corr, cov3 = schur_condition(cov4, _TAU, mt, eps=_COV_EPS)
    wt = torch.exp(-0.5 * (_TAU - mt) ** 2 / (cov4[:, 3, 3] + TEMPORAL_VARIANCE_EPS))
    opac = torch.sigmoid(op) * wt
    covars = cov3[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    radii, means2d, _, conics, _ = fully_fused_projection(
        mn + mean_corr,
        covars,
        None,
        None,
        vm.float().cuda(),
        Ks.float().cuda(),
        _W,
        _H,
        eps2d=_EPS2D,
        near_plane=_NEAR,
        far_plane=_FAR,
        camera_model="fisheye",
        opacities=opac,
    )
    vis_g = (radii > 0).all(dim=-1)

    both = vis_f & vis_g
    assert both.sum() > 1000
    assert (vis_f ^ vis_g).float().mean() < 5e-3
    assert (out_f["means2d"][both] - means2d[both]).abs().max() < 2e-3
    assert _rel_p99(out_f["conics"][both].double(), conics[both].double()) < 1e-3
    assert (out_f["radii"][both] - radii[both]).abs().max() <= 1

    g = torch.Generator(device="cuda").manual_seed(9)
    w_m = torch.randn(means2d.shape, device="cuda", generator=g)
    w_c = 1e-2 * torch.randn(conics.shape, device="cuda", generator=g)
    ((out_f["means2d"] * w_m)[both].sum() + (out_f["conics"] * w_c)[both].sum()).backward()
    ((means2d * w_m)[both].sum() + (conics * w_c)[both].sum()).backward()
    for name, a, b in zip(_PARAMS[:5], p_f, params, strict=False):
        err = _rel_p99(a.grad.double(), b.grad.double())
        assert err < 1e-2, f"{name}: p99 relative gradient error vs gsplat {err:.2e}"


@pytest.mark.parametrize("family", [_PINHOLE, _FISHEYE], ids=["bc", "kb"])
def test_lens_ewa_mean_is_the_opencv_projection(family: int) -> None:
    """mean2d is cv2.projectPoints / cv2.fisheye.projectPoints of the centre, the same K on both sides."""
    cv2 = pytest.importorskip("cv2")

    vm, Ks, dist = _camera(family)
    pop = _population(family, "corner" if family == _FISHEYE else "mid", vm[0])
    pop["mu_t"] = torch.full_like(pop["mu_t"], _TAU)  # no Schur mean correction: the centre is `means`
    _, out, vis = _fused(pop, vm, Ks, dist, family, False, False)
    x_c = pop["means"] @ vm[0, :3, :3].T + vm[0, :3, 3]

    obj = x_c.numpy().reshape(-1, 1, 3)
    K, d = Ks[0].numpy(), dist[0].numpy()
    zero = torch.zeros(3, dtype=torch.float64).numpy()
    project = cv2.fisheye.projectPoints if family == _FISHEYE else cv2.projectPoints
    pix, _ = project(obj, zero, zero, K, d)
    pix = torch.from_numpy(pix.reshape(-1, 2)).cuda()
    seen = vis[0]
    assert seen.sum() > 500
    assert (out["means2d"][0][seen].double() - pix[seen]).abs().max() < 2e-3


def _ut_vs_ewa_medians(family: int, sigma: float) -> tuple[float, float]:
    """Median mean2d distance (px) and median relative Σ2d difference, UT vs lens EWA, Gaussians of size σ."""
    from anycam4d.cuda.autograd import UTFusedFunction

    vm, Ks, dist = _camera(family)
    pop = _population(family, "corner", vm[0])
    g = torch.Generator().manual_seed(4)
    pop["scales_4d_log"][:, :3] = math.log(sigma) + 0.3 * torch.randn(pop["means"].shape[0], 3, generator=g).double()
    _, ewa, vis = _fused(pop, vm, Ks, dist, family, False, False)

    n = pop["means"].shape[0]
    params = [pop[k].float().cuda() for k in _PARAMS]
    sh = (torch.zeros(n, 1, 3, device="cuda"), torch.zeros(n, 0, 3, device="cuda"), torch.zeros(n, 0, 3, device="cuda"))
    means2d, conics, _, radii, *_ = UTFusedFunction.apply(
        *params,
        *sh,
        0,
        APPEARANCE_CUDA_ID[AppearanceType.SH4D_ADDITIVE],
        0,
        0,
        vm.float().cuda(),
        Ks.float().cuda(),
        dist.float().cuda(),
        _TAU,
        _COV_EPS,
        _EPS2D,
        _NEAR,
        _FAR,
        0.0,
        _W,
        _H,
        family,
    )
    both = vis & (radii > 0).all(dim=-1)

    def cov2d(con: Tensor) -> Tensor:
        a, b, c = con.unbind(-1)
        det = a * c - b * b
        return torch.stack([c / det - _EPS2D, -b / det, a / det - _EPS2D], dim=-1)

    ce, cu = cov2d(ewa["conics"][both]), cov2d(conics[both])
    rel = (ce - cu).abs().sum(-1) / (ce[:, 0] + ce[:, 2])
    return (ewa["means2d"][both] - means2d[both]).norm(dim=-1).median().item(), rel.median().item()


@pytest.mark.parametrize("family", [_PINHOLE, _FISHEYE], ids=["bc", "kb"])
def test_lens_ewa_and_ut_converge_for_small_gaussians(family: int) -> None:
    """Both backends share one lens: their projected mean and covariance meet as σ → 0, at the σ² rate.

    Medians, because UT's means2d/conic are a tile bound where its box is replaced (invariant 4), and an image
    comparison would measure the two rasterizers (screen-space eps2d vs ray-space), not the lens.
    """
    mean_big, cov_big = _ut_vs_ewa_medians(family, 0.01)
    mean_small, cov_small = _ut_vs_ewa_medians(family, 0.003)
    assert mean_small < 3e-3 and cov_small < 2e-4
    assert mean_big / mean_small > 4 and cov_big / cov_small > 4
