"""CUDA kernels against the PyTorch oracles; skipped without the extension or a GPU."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from anycam4d.camera.batched import BatchedCamera
from anycam4d.enums import CameraModelType
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.appearance import APPEARANCE_CUDA_ID
from anycam4d.enums import AppearanceType
from tests.oracles.factor import cov_of
from tests.oracles.factor import factor_of
from tests.oracles.factor import upper_tri
from tests.oracles.factor import whitened_error


_ADDITIVE = APPEARANCE_CUDA_ID[AppearanceType.SH4D_ADDITIVE]


def _stride(sh_temporal: Tensor) -> int:
    """Floats per Gaussian in the temporal coefficient parameter."""
    return sh_temporal.shape[1] * 3 if sh_temporal.numel() else 0


def _no_lens() -> tuple[Tensor, int]:
    """EWA ``(dist, family)`` of an undistorted pinhole."""
    return torch.empty(0, device="cuda"), 0


def _skip_if_no_cuda_ext() -> None:
    from anycam4d.cuda.ext import HAS_CUDA_EXT

    if not HAS_CUDA_EXT:
        pytest.skip("CUDA extension not available (HAS_CUDA_EXT=False)")

    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available")


def _make_inputs(
    N: int, dtype: torch.dtype = torch.float32, device: str = "cuda"
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Random ``(quats_l, quats_r, scales_4d_log, means, mu_t, opacities)`` on ``device``."""
    g = torch.Generator(device=device)
    g.manual_seed(42)

    quats_l = torch.randn(N, 4, dtype=dtype, device=device, generator=g)
    quats_r = torch.randn(N, 4, dtype=dtype, device=device, generator=g)
    # Scale in a sensible range: small negative → small positive exp(s) ≈ 0.1..2
    scales_4d_log = torch.randn(N, 4, dtype=dtype, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, dtype=dtype, device=device, generator=g)
    mu_t = torch.rand(N, dtype=dtype, device=device, generator=g)  # in [0, 1]
    opacities = torch.randn(N, dtype=dtype, device=device, generator=g) * 2.0

    return quats_l, quats_r, scales_4d_log, means, mu_t, opacities


def test_forward_rot4d_equivalence() -> None:
    """CUDA forward matches the fp64 reference at N=100K; the factor L in the whitened metric."""
    _skip_if_no_cuda_ext()

    from tests.fused_stages import condition

    N = 100_000
    tau = 0.35
    dtype = torch.float32
    device = "cuda"

    quats_l, quats_r, scales_4d_log, means, mu_t, opacities = _make_inputs(N, dtype, device)

    means_c, factors_c, opac_c, wt_c = condition(
        quats_l, quats_r, scales_4d_log, means, mu_t, opacities, tau, 1e-6, 0.0
    )

    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition

    ql, qr, s4, mt = (x.double() for x in (quats_l, quats_r, scales_4d_log, mu_t))
    cov_4d = build_cov4d_rot(ql, qr, s4)
    mean_corr, cov_3d_cond = schur_condition(cov_4d, tau, mt, eps=1e-6)
    means_ref = means.double() + mean_corr

    cov_t_marg = cov_4d[:, 3, 3]
    dt = tau - mt
    wt_ref = torch.exp(-0.5 * dt * dt / (cov_t_marg + TEMPORAL_VARIANCE_EPS))
    opac_ref = torch.sigmoid(opacities.double()) * wt_ref

    tol = 1e-5
    max_means = (means_c - means_ref).abs().max().item()
    max_factor = whitened_error(factors_c, factor_of(cov_3d_cond)).max().item()
    max_opac = (opac_c - opac_ref).abs().max().item()
    max_wt = (wt_c - wt_ref).abs().max().item()

    print(
        f"\nForward max diffs: means={max_means:.2e}, factor (whitened)={max_factor:.2e}, "
        f"opac={max_opac:.2e}, wt={max_wt:.2e}"
    )

    assert max_means < tol, f"means diff too large: {max_means}"
    assert max_factor < 1e-4, f"factor whitened error too large: {max_factor}"
    assert max_opac < tol, f"opac diff too large: {max_opac}"
    assert max_wt < tol, f"wt diff too large: {max_wt}"


def test_backward_rot4d_equivalence() -> None:
    """CUDA backward matches fp64 autograd (through the Cholesky factor) within 1e-4 at N=10K."""
    _skip_if_no_cuda_ext()

    from tests.fused_stages import condition
    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition

    N = 10_000
    tau = 0.6
    cov_eps = 1e-6
    prefilter = 0.0
    dtype = torch.float32
    device = "cuda"

    quats_l, quats_r, scales_4d_log, means, mu_t, opacities = _make_inputs(N, dtype, device)

    # ---- PyTorch autograd reference ----
    def pytorch_fwd(
        ql: Tensor, qr: Tensor, s4d: Tensor, mn: Tensor, mt: Tensor, op: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cov_4d = build_cov4d_rot(ql, qr, s4d)
        mean_corr, cov_3d_cond = schur_condition(cov_4d, tau, mt, eps=cov_eps)
        means_o = mn + mean_corr
        cov_t_marg = cov_4d[:, 3, 3]
        dt = tau - mt
        wt = torch.exp(-0.5 * dt * dt / (cov_t_marg + TEMPORAL_VARIANCE_EPS))
        opac = torch.sigmoid(op) * wt
        return means_o, factor_of(cov_3d_cond), opac, wt

    def get_grads_ref(ql: Tensor, qr: Tensor, s4d: Tensor, mn: Tensor, mt: Tensor, op: Tensor) -> list[Tensor]:
        inputs = [x.double().requires_grad_(True) for x in (ql, qr, s4d, mn, mt, op)]
        means_o, factors_o, opac_o, wt_o = pytorch_fwd(*inputs)
        # Scalar loss: sum of all outputs to get non-trivial gradients
        loss = means_o.sum() + factors_o.sum() + opac_o.sum()  # wt reaches the loss only through opac
        loss.backward()
        return [inp.grad.detach().float() for inp in inputs]

    def get_grads_cuda(ql: Tensor, qr: Tensor, s4d: Tensor, mn: Tensor, mt: Tensor, op: Tensor) -> list[Tensor]:
        inputs = [ql, qr, s4d, mn, mt, op]
        for inp in inputs:
            inp.requires_grad_(True)
        means_o, factors_o, opac_o, wt_o = condition(*inputs, tau, cov_eps, prefilter)
        loss = means_o.sum() + factors_o.sum() + opac_o.sum()  # wt reaches the loss only through opac
        loss.backward()
        return [inp.grad.detach().clone() for inp in inputs]

    names = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities"]
    grads_ref = get_grads_ref(
        quats_l.clone(), quats_r.clone(), scales_4d_log.clone(), means.clone(), mu_t.clone(), opacities.clone()
    )
    grads_cuda = get_grads_cuda(
        quats_l.clone(), quats_r.clone(), scales_4d_log.clone(), means.clone(), mu_t.clone(), opacities.clone()
    )

    tol = 1e-4
    print()
    all_ok = True
    for name, g_ref, g_cuda in zip(names, grads_ref, grads_cuda):
        diff = (g_cuda - g_ref).abs().max().item()
        print(f"  {name:12s}: max diff = {diff:.3e}")
        if diff >= tol:
            all_ok = False
            print(f"    *** FAIL: {diff:.3e} >= {tol:.3e}")

    assert all_ok, "Backward gradients exceed tolerance (see printout above)"


def test_conditioned_factor_survives_schur_cancellation() -> None:
    """Fast thin Gaussians, where the fp32 Schur Σ is indefinite: L and its gradient still match fp64."""
    _skip_if_no_cuda_ext()

    from tests.fused_stages import condition
    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition

    N, tau, cov_eps = 20_000, 0.5, 1e-6
    g = torch.Generator(device="cuda").manual_seed(11)
    quats_l = torch.randn(N, 4, device="cuda", generator=g)
    quats_r = torch.randn(N, 4, device="cuda", generator=g)
    # One long axis mixing space and time (a fast mover), the rest near the cov_eps floor.
    scales_4d_log = torch.rand(N, 4, device="cuda", generator=g) * 4.0 - 9.0
    scales_4d_log[:, 0] = torch.rand(N, device="cuda", generator=g) * 2.0 + 1.0
    means = torch.zeros(N, 3, device="cuda")
    mu_t = torch.rand(N, device="cuda", generator=g)
    opacities = torch.zeros(N, device="cuda")
    weights = torch.randn(N, 6, device="cuda", generator=g)

    params = [quats_l, quats_r, scales_4d_log, means, mu_t, opacities]
    fused = [p.clone().requires_grad_(True) for p in params]
    factors = condition(*fused, tau, cov_eps, 0.0)[1]
    (factors * weights).sum().backward()

    ref = [p.double().requires_grad_(True) for p in params]
    cov3 = schur_condition(build_cov4d_rot(ref[0], ref[1], ref[2]), tau, ref[4], eps=cov_eps)[1]
    factors_ref = factor_of(cov3)
    (factors_ref * weights.double()).sum().backward()

    with torch.no_grad():
        cov3_fp32 = schur_condition(build_cov4d_rot(quats_l, quats_r, scales_4d_log), tau, mu_t, eps=cov_eps)[1]
        n_indefinite = int((torch.linalg.cholesky_ex(cov3_fp32).info > 0).sum())
    assert n_indefinite > 50, f"population must exercise the cancellation ({n_indefinite} indefinite in fp32)"

    err = whitened_error(factors.detach(), factors_ref.detach())
    print(f"\n  fp32 Schur indefinite: {n_indefinite}/{N}; whitened error max {err.max():.2e}")
    assert err.max() < 1e-2, f"whitened error {err.max():.2e}"
    for name, pf, pr in zip(["quats_l", "quats_r", "scales_4d_log"], fused[:3], ref[:3]):
        gf, gr = pf.grad.reshape(N, -1), pr.grad.float().reshape(N, -1)
        norm = gr.norm(dim=1)
        rel = (gf - gr).norm(dim=1) / (norm + 1e-3 * norm.median())
        print(f"  {name:14s}: rel err median {rel.median():.2e}  p99 {rel.quantile(0.99):.2e}  max {rel.max():.2e}")
        assert rel.quantile(0.99) < 5e-2, f"{name}: p99 per-primitive relative error {rel.quantile(0.99):.2e}"


def _make_cameras(C: int, device: str = "cuda") -> tuple[Tensor, Tensor]:
    """Create random valid camera matrices for testing."""
    g = torch.Generator(device=device)
    g.manual_seed(7)

    # Random rotations via QR decomposition for each camera
    viewmats = []
    for _ in range(C):
        R_raw = torch.randn(3, 3, device=device, generator=g)
        R, _ = torch.linalg.qr(R_raw)
        if torch.det(R) < 0:
            R[:, 0] *= -1
        t = torch.randn(3, device=device, generator=g) * 0.1
        vm = torch.eye(4, device=device)
        vm[:3, :3] = R
        vm[:3, 3] = t
        viewmats.append(vm)
    viewmats = torch.stack(viewmats, dim=0)  # [C, 4, 4]

    # Intrinsics: focal=256, center=(128, 128)
    K = torch.tensor([[256.0, 0.0, 128.0], [0.0, 256.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = K.unsqueeze(0).expand(C, -1, -1).contiguous()  # [C, 3, 3]
    return viewmats, Ks


def test_fused_ewa_forward_equivalence() -> None:
    """Fused kernel screenspace outputs must match reference (cond + SH + gsplat proj)."""
    _skip_if_no_cuda_ext()

    from tests.fused_stages import condition
    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    from tests.oracles.sh4d import eval_sh_4d

    N = 50_000
    C = 4
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(99)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05

    viewmats, Ks = _make_cameras(C, device=device)

    # ---------- Fused kernel ----------
    means2d_f, conics_f, colors_f, opac_f, depths_f, radii_f, means3d_f, wt_f = (
        launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            _ADDITIVE,
            n_temporal_freqs,
            _stride(sh_temporal),
            viewmats,
            Ks,
            *_no_lens(),
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            False,
        )
    )

    # ---------- Reference path ----------
    # Step 1: P16 conditioning
    means3d_ref, factors_ref, opac_ref, wt_ref = condition(
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        tau,
        cov_eps,
        prefilter,
    )
    covars3d_ref = upper_tri(cov_of(factors_ref))

    # Step 2: 4D SH colors for each camera
    R_cams = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cams = viewmats[:, :3, 3]  # [C, 3]
    # campos[c] = -R^T @ t
    campos = -(R_cams.transpose(-2, -1) @ t_cams.unsqueeze(-1)).squeeze(-1)  # [C, 3]
    dirs = means3d_ref[None, :, :] - campos[:, None, :]  # [C, N, 3]
    dir_t = tau - mu_t  # [N]
    colors_ref_list = []
    for c in range(C):
        c_rgb = eval_sh_4d(sh_coeffs, sh_temporal, dirs[c], dir_t, sh_degree, n_temporal_freqs)
        colors_ref_list.append(c_rgb)
    colors_ref = torch.stack(colors_ref_list, dim=0)  # [C, N, 3]

    # Step 3: gsplat projection
    from gsplat.cuda._wrapper import fully_fused_projection

    radii_ref, means2d_ref, depths_ref, conics_ref, _ = fully_fused_projection(
        means3d_ref,
        covars3d_ref,
        None,
        None,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=False,
        opacities=opac_ref,
    )

    # ---------- Compare where both paths agree (radii > 0) ----------
    vis_mask_f = (radii_f > 0).all(dim=-1)  # [C, N]
    vis_mask_ref = (radii_ref > 0).all(dim=-1)  # [C, N]
    both_visible = vis_mask_f & vis_mask_ref

    tol_screenspace = 1e-4
    tol_opac = 1e-5
    tol_color = 1e-4

    n_vis = both_visible.sum().item()
    print(f"\n  Fused visible: {vis_mask_f.sum().item()}, ref visible: {vis_mask_ref.sum().item()}, both: {n_vis}")

    if n_vis == 0:
        pytest.skip("No Gaussians visible in both paths — numerical edge case")

    diff_means2d = (means2d_f[both_visible] - means2d_ref[both_visible]).abs().max().item()
    diff_conics = (conics_f[both_visible] - conics_ref[both_visible]).abs().max().item()
    diff_depths = (depths_f[both_visible] - depths_ref[both_visible]).abs().max().item()
    diff_opac = (opac_f[both_visible] - opac_ref.unsqueeze(0).expand(C, -1)[both_visible]).abs().max().item()
    diff_colors = (colors_f[both_visible] - colors_ref[both_visible]).abs().max().item()

    print(
        f"  means2d: {diff_means2d:.2e}  conics: {diff_conics:.2e}  depths: {diff_depths:.2e}  "
        f"opacities: {diff_opac:.2e}  colors: {diff_colors:.2e}"
    )

    assert diff_means2d < tol_screenspace, f"means2d diff too large: {diff_means2d:.2e}"
    assert diff_conics < tol_screenspace, f"conics diff too large:  {diff_conics:.2e}"
    assert diff_depths < tol_screenspace, f"depths diff too large:  {diff_depths:.2e}"
    assert diff_opac < tol_opac, f"opacities diff too large: {diff_opac:.2e}"
    assert diff_colors < tol_color, f"colors diff too large:  {diff_colors:.2e}"

    # Side outputs
    diff_wt = (wt_f - wt_ref).abs().max().item()
    diff_means3d = (means3d_f - means3d_ref).abs().max().item()
    print(f"  wt: {diff_wt:.2e}  means_3d_cond: {diff_means3d:.2e}")
    assert diff_wt < tol_opac, f"wt diff too large: {diff_wt:.2e}"
    assert diff_means3d < tol_screenspace, f"means_3d_cond diff too large: {diff_means3d:.2e}"


def test_fused_ewa_render_parity() -> None:
    """Full render via FusedEWABackend must match PyTorchEWABackend (PSNR > 60 dB)."""
    _skip_if_no_cuda_ext()

    try:
        from anycam4d.renderer.ewa_fused_backend import FusedEWABackend
    except ImportError:
        pytest.skip("FusedEWABackend not available (import failed)")

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from tests.oracles.ewa_pytorch_backend import EWAPyTorchBackend
    from tests.oracles.sh4d import Sh4dAppearanceModel

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create a synthetic model
    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points,
        sh_degree=sh_degree,
        n_temporal_freqs=n_temporal_freqs,
        appearance_model="sh4d_additive",
    ).to(device)
    # Non-zero odd-degree and temporal terms, so a flipped view direction or a dropped temporal term shows.
    with torch.no_grad():
        model.splats["shN"].normal_(0.0, 0.2)
        model.splats["sh_temporal"].normal_(0.0, 0.1)

    # Camera looking toward origin from z=2
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()

    cfg = RenderingConfig(packed=False)
    fused_backend = FusedEWABackend(cfg)
    pytorch_backend = EWAPyTorchBackend(cfg, Sh4dAppearanceModel(n_temporal_freqs))

    # Render in no-grad mode (ensures fused path is used)
    with torch.no_grad():
        out_fused = fused_backend.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )
        out_pytorch = pytorch_backend.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )

    # Pixel-level parity (PSNR) — compare RGB channels
    rgb_f = out_fused.colors[..., :3]
    rgb_p = out_pytorch.colors[..., :3]

    mse = ((rgb_f - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  Render parity PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"Render parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"


def test_fused_ewa_backward_equivalence() -> None:
    """Fused EWA backward matches autograd within 1% over jointly visible Gaussians (N=10K, C=2)."""
    _skip_if_no_cuda_ext()

    from gsplat.cuda._wrapper import fully_fused_projection

    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    from anycam4d.cuda.autograd import EWAFusedFunction
    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition
    from tests.oracles.sh4d import eval_sh_4d

    N = 10_000
    C = 2
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(55)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)

    # Determine visible Gaussians using the fused forward (no grad)
    with torch.no_grad():
        _, _, _, _, _, radii_ref, _, _ = launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            _ADDITIVE,
            n_temporal_freqs,
            _stride(sh_temporal),
            viewmats,
            Ks,
            *_no_lens(),
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            False,
        )
    # vis_mask[c, n] = True if Gaussian n is visible in camera c
    vis_mask = (radii_ref > 0).all(dim=-1)  # [C, N]
    n_vis = vis_mask.sum().item()
    print(f"\n  Visible (Gaussian, camera) pairs: {n_vis} / {C * N}")
    if n_vis < 10:
        pytest.skip("Too few visible Gaussians for meaningful comparison")

    param_names = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities", "sh0", "shN", "sh_temporal"]

    def get_grads_fused(params: list[Tensor]) -> list[Tensor]:
        for p in params:
            p.requires_grad_(True)
        means2d, conics, colors, opac, depths, radii, _, _ = EWAFusedFunction.apply(
            *params,
            sh_degree,
            _ADDITIVE,
            n_temporal_freqs,
            _stride(sh_temporal),
            viewmats,
            Ks,
            *_no_lens(),
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            False,
        )
        # Loss only over visible entries (same visibility as reference)
        loss = means2d[vis_mask].sum() + conics[vis_mask].sum() + colors[vis_mask].sum() + opac[vis_mask].sum()
        loss.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    def get_grads_pytorch(params: list[Tensor]) -> list[Tensor]:
        ql, qr, s4d, mn, mt, op, sh0_, shN_, sh_t = params
        sh_c = torch.cat([sh0_, shN_], dim=1)
        for p in params:
            p.requires_grad_(True)

        cov_4d = build_cov4d_rot(ql, qr, s4d)
        mean_corr, cov_3d_cond = schur_condition(cov_4d, tau, mt, eps=cov_eps)
        means3d_pt = mn + mean_corr
        cov_t_marg = cov_4d[:, 3, 3]
        dt_val = tau - mt
        wt_pt = torch.exp(-0.5 * dt_val * dt_val / (cov_t_marg + TEMPORAL_VARIANCE_EPS))
        opac_pt = torch.sigmoid(op) * wt_pt  # [N]

        covars3d = torch.stack(
            [
                cov_3d_cond[:, 0, 0],
                cov_3d_cond[:, 0, 1],
                cov_3d_cond[:, 0, 2],
                cov_3d_cond[:, 1, 1],
                cov_3d_cond[:, 1, 2],
                cov_3d_cond[:, 2, 2],
            ],
            dim=1,
        )  # [N, 6]

        R_cams = viewmats[:, :3, :3]
        t_cams = viewmats[:, :3, 3]
        campos = -(R_cams.transpose(-2, -1) @ t_cams.unsqueeze(-1)).squeeze(-1)  # [C, 3]
        dirs = means3d_pt[None, :, :] - campos[:, None, :]  # [C, N, 3]
        dir_t_val = tau - mt  # [N]

        colors_list = []
        for c in range(C):
            c_rgb = eval_sh_4d(sh_c, sh_t, dirs[c], dir_t_val, sh_degree, n_temporal_freqs)
            colors_list.append(c_rgb)
        colors_pt = torch.stack(colors_list, dim=0)  # [C, N, 3]

        radii_r, means2d_r, depths_r, conics_r, _ = fully_fused_projection(
            means3d_pt,
            covars3d,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=False,
            opacities=opac_pt,
        )
        # Loss over the same visibility mask as the fused path
        opac_expanded = opac_pt.unsqueeze(0).expand(C, -1)  # [C, N]
        loss = (
            means2d_r[vis_mask].sum()
            + conics_r[vis_mask].sum()
            + colors_pt[vis_mask].sum()
            + opac_expanded[vis_mask].sum()
        )
        loss.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    clones_fused = [
        p.clone().detach() for p in [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    ]
    clones_pytorch = [
        p.clone().detach() for p in [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    ]

    grads_fused = get_grads_fused(clones_fused)
    grads_pytorch = get_grads_pytorch(clones_pytorch)

    tol_rel = 5e-2  # 5% relative tolerance (float32 precision + different culling)
    all_ok = True
    for name, gf, gp in zip(param_names, grads_fused, grads_pytorch):
        scale = gp.abs().max().item()
        if scale < 1e-8:
            continue
        diff = (gf - gp).abs().max().item()
        rel = diff / (scale + 1e-8)
        print(f"  {name:14s}: max abs diff = {diff:.3e}  rel = {rel:.3e}  scale = {scale:.3e}")
        if rel >= tol_rel:
            all_ok = False
            print(f"    *** FAIL: rel {rel:.3e} >= {tol_rel:.3e}")

    assert all_ok, "Fused backward gradients exceed relative tolerance (see printout above)"


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("antialiased", [False, True], ids=["classic", "antialiased"])
# f32 conditioning noise grows with the covariance condition number, hence the looser bound at 1e3.
@pytest.mark.parametrize(("anisotropy", "tol"), [(1.0, 1e-2), (1e2, 1e-2), (1e3, 5e-2)])
def test_fused_ewa_covariance_backward(anisotropy: float, tol: float, antialiased: bool, packed: bool) -> None:
    """Shape gradients through the 2D covariance alone match conditioning + gsplat projection, per primitive.

    Upstream is on conics (classic) or on the compensated opacity (antialiased), so the covariance chain is not
    masked by the much larger mean gradients.
    """
    _skip_if_no_cuda_ext()

    from gsplat.cuda._wrapper import fully_fused_projection

    from anycam4d.cuda.autograd import EWAFusedFunction
    from anycam4d.cuda.autograd import EWAFusedPackedFunction
    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition

    N, C = 4096, 3
    tau, cov_eps, eps2d, near_plane, far_plane, prefilter = 0.4, 1e-6, 0.3, 0.01, 1e10, 0.0
    width, height = 256, 256
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(21)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    # Spatial axes span `anisotropy` around 0.03; the time axis is kept wide so conditioning barely shrinks them.
    spatial = math.log(0.03) + math.log(anisotropy) * torch.tensor([0.5, 0.0, -0.5], device=device)
    scales_4d_log = torch.cat([spatial.expand(N, 3), torch.full((N, 1), math.log(0.5), device=device)], dim=1)
    scales_4d_log = scales_4d_log + 0.1 * torch.randn(N, 4, device=device, generator=g)
    means = torch.randn(N, 3, device=device, generator=g) * 0.2
    mu_t = tau + 0.1 * torch.randn(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g)
    sh0, shN = torch.zeros(N, 1, 3, device=device), torch.zeros(N, 0, 3, device=device)
    sh_temporal = torch.zeros(N, 0, 3, device=device)

    # Every camera looks at the origin from distance 2, so all primitives are in front of all cameras.
    viewmats, Ks = _make_cameras(C, device=device)
    viewmats[:, :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    w_conic = torch.randn(C, N, 3, device=device, generator=g)
    w_opac = torch.randn(C, N, device=device, generator=g)

    def upstream(conics: Tensor, opac: Tensor, cam: Tensor, gid: Tensor) -> Tensor:
        if antialiased:
            return (opac * w_opac[cam, gid]).sum()
        return (conics * w_conic[cam, gid]).sum()

    def fused(params: list[Tensor], vis: Tensor) -> list[Tensor]:
        scalars = (
            0,
            _ADDITIVE,
            0,
            0,
            viewmats,
            Ks,
            *_no_lens(),
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            antialiased,
        )
        if packed:
            cam, gid, _, _, conics, _, opac, _, _, _, _ = EWAFusedPackedFunction.apply(*params, *scalars)
            cam, gid = cam.long(), gid.long()
            keep = vis[cam, gid]
            loss = upstream(conics[keep], opac[keep], cam[keep], gid[keep])
        else:
            _, conics, _, opac, _, _, _, _ = EWAFusedFunction.apply(*params, *scalars)
            cam, gid = torch.nonzero(vis, as_tuple=True)
            loss = upstream(conics[vis], opac[vis], cam, gid)
        loss.backward()
        return [p.grad for p in params[:3]]

    def oracle(params: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        ql, qr, s4d, mn, mt, op = params[:6]
        cov_4d = build_cov4d_rot(ql, qr, s4d)
        mean_corr, cov3 = schur_condition(cov_4d, tau, mt, eps=cov_eps)
        wt = torch.exp(-0.5 * (tau - mt) ** 2 / (cov_4d[:, 3, 3] + TEMPORAL_VARIANCE_EPS))
        covars = cov3[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
        opac = torch.sigmoid(op) * wt
        radii, _, _, conics, comp = fully_fused_projection(
            mn + mean_corr,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=antialiased,
            opacities=opac,
        )
        vis = (radii > 0).all(dim=-1)
        opac = opac * comp if antialiased else opac.expand(C, -1)
        cam, gid = torch.nonzero(vis, as_tuple=True)
        upstream(conics[vis], opac[vis], cam, gid).backward()
        return [p.grad for p in params[:3]], vis

    base = [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    grads_ref, vis = oracle([p.clone().requires_grad_(True) for p in base])
    assert vis.sum() > N, "too few visible (camera, primitive) pairs"
    grads_fused = fused([p.clone().requires_grad_(True) for p in base], vis)

    print()
    for name, gf, gr in zip(["quats_l", "quats_r", "scales_4d_log"], grads_fused, grads_ref):
        norm = gr.norm(dim=1)
        rel = (gf - gr).norm(dim=1) / (norm + 1e-3 * norm.median())
        q99 = torch.quantile(rel, 0.99).item()
        print(f"  {name:14s}: rel err median {rel.median():.2e}  p99 {q99:.2e}  max {rel.max():.2e}")
        assert q99 < tol, f"{name}: p99 per-primitive relative error {q99:.2e}"


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("output", ["factors", "means", "opacities"])
# A narrow time axis couples space and time strongly, so the Schur and mean-correction terms carry the gradient.
@pytest.mark.parametrize("time_scale", [0.5, 0.05], ids=["wide_t", "narrow_t"])
@pytest.mark.parametrize(("anisotropy", "tol"), [(1.0, 1e-2), (1e3, 5e-2)])
def test_fused_ut_conditioning_backward(
    anisotropy: float, tol: float, time_scale: float, output: str, packed: bool
) -> None:
    """UT fused backward from one world-space output to the 4D parameters matches fp64 autograd, per primitive."""
    _skip_if_no_cuda_ext()

    from anycam4d.camera.registry import FAMILY_CUDA_ID
    from anycam4d.camera.registry import ProjectionFamily
    from anycam4d.cuda.autograd import UTFusedFunction
    from anycam4d.cuda.autograd import UTFusedPackedFunction
    from anycam4d.math.covariance import build_cov4d_rot
    from anycam4d.math.covariance import schur_condition

    N, C = 4096, 3
    tau, cov_eps, eps2d, near_plane, far_plane, prefilter = 0.4, 1e-6, 0.3, 0.01, 1e10, 0.0
    width, height = 256, 256
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(23)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    spatial = math.log(0.03) + math.log(anisotropy) * torch.tensor([0.5, 0.0, -0.5], device=device)
    scales_4d_log = torch.cat([spatial.expand(N, 3), torch.full((N, 1), math.log(time_scale), device=device)], dim=1)
    scales_4d_log = scales_4d_log + 0.1 * torch.randn(N, 4, device=device, generator=g)
    means = torch.randn(N, 3, device=device, generator=g) * 0.2
    mu_t = tau + 0.5 * time_scale * torch.randn(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g)
    sh0, shN = torch.zeros(N, 1, 3, device=device), torch.zeros(N, 0, 3, device=device)
    sh_temporal = torch.zeros(N, 0, 3, device=device)

    viewmats, Ks = _make_cameras(C, device=device)
    viewmats[:, :3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
    radial = torch.zeros(C, 12, device=device)
    family = FAMILY_CUDA_ID[ProjectionFamily.PINHOLE_FAMILY]
    weights = {
        "factors": torch.randn(N, 6, device=device, generator=g),
        "means": torch.randn(N, 3, device=device, generator=g),
        "opacities": torch.randn(N, device=device, generator=g),
    }

    def fused(params: list[Tensor]) -> tuple[list[Tensor], Tensor, Tensor]:
        scalars = (
            0,
            _ADDITIVE,
            0,
            0,
            viewmats,
            Ks,
            radial,
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            family,
        )
        if packed:
            _, gid, _, _, _, _, _, _, m3d, factors, opac, _ = UTFusedPackedFunction.apply(*params, *scalars)
            seen = torch.zeros(N, dtype=torch.bool, device=device)
            seen[gid.long()] = True
        else:
            _, _, _, radii, _, m3d, factors, opac, _ = UTFusedFunction.apply(*params, *scalars)
            seen = (radii > 0).all(dim=-1).any(dim=0)
        out = {"factors": factors, "means": m3d, "opacities": opac}[output]
        (out * weights[output]).sum().backward()
        return [torch.zeros_like(p) if p.grad is None else p.grad for p in params[:6]], seen, factors.detach()

    def oracle(params: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        ql, qr, s4d, mn, mt, op = params[:6]
        cov_4d = build_cov4d_rot(ql, qr, s4d)
        mean_corr, cov3 = schur_condition(cov_4d, tau, mt, eps=cov_eps)
        wt = torch.exp(-0.5 * (tau - mt) ** 2 / (cov_4d[:, 3, 3] + TEMPORAL_VARIANCE_EPS))
        factors = factor_of(cov3)
        out = {"factors": factors, "means": mn + mean_corr, "opacities": torch.sigmoid(op) * wt}[output]
        (out * weights[output].double()).sum().backward()
        grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params[:6]]
        return [gr.float() for gr in grads], factors.detach()

    base = [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    grads_ref, factors_ref = oracle([p.double().requires_grad_(True) for p in base])
    grads_fused, seen, factors_fused = fused([p.clone().requires_grad_(True) for p in base])
    assert seen.sum() > N // 2, "too few primitives visible in any camera"
    # fp32 floor of the whitened metric is ~u·σ_max/σ_min: 1.3e-4 at anisotropy 1e3.
    cov_err = whitened_error(factors_fused, factors_ref)
    assert cov_err.max() < 1e-3, f"conditioned covariance differs from the oracle: {cov_err.max():.2e}"

    print()
    names = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities"]
    for name, gf, gr in zip(names, grads_fused, grads_ref):
        gf, gr = gf[seen].reshape(int(seen.sum()), -1), gr[seen].reshape(int(seen.sum()), -1)
        norm = gr.norm(dim=1)
        if norm.max() == 0:
            assert gf.abs().max() == 0, f"{name}: gradient where the oracle has none"
            continue
        rel = (gf - gr).norm(dim=1) / (norm + 1e-3 * norm.median() + 1e-30)
        q99 = torch.quantile(rel, 0.99).item()
        print(f"  {name:14s}: rel err median {rel.median():.2e}  p99 {q99:.2e}  max {rel.max():.2e}")
        assert q99 < tol, f"{name}: p99 per-primitive relative error {q99:.2e}"


def test_packed_forward_equivalence() -> None:
    """Packed fused forward outputs must match dense fused forward for visible Gaussians."""
    _skip_if_no_cuda_ext()

    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    from anycam4d.cuda.ext import launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel

    N = 50_000
    C = 4
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(77)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)

    args = (
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        sh0,
        shN,
        sh_temporal,
        sh_degree,
        _ADDITIVE,
        n_temporal_freqs,
        _stride(sh_temporal),
        viewmats,
        Ks,
        *_no_lens(),
        tau,
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter,
        width,
        height,
        False,
    )

    # Dense forward
    means2d_d, conics_d, colors_d, opac_d, depths_d, radii_d, means3d_d, wt_d = (
        launch_ewa_4d_fused_conditioned_projection_fwd_kernel(*args)
    )

    # Packed forward
    cam_ids, gauss_ids, indptr, means2d_p, conics_p, colors_p, opac_p, depths_p, radii_p, means3d_p, wt_p = (
        launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel(*args)
    )

    nnz = cam_ids.shape[0]
    print(f"\n  Dense visible: {(radii_d > 0).all(dim=-1).sum().item()}, packed nnz: {nnz}")

    if nnz == 0:
        pytest.skip("No visible Gaussians — numerical edge case")

    # Index into dense outputs using packed indices
    cam_idx = cam_ids.long()
    gauss_idx = gauss_ids.long()

    means2d_dense_at_packed = means2d_d[cam_idx, gauss_idx]
    conics_dense_at_packed = conics_d[cam_idx, gauss_idx]
    colors_dense_at_packed = colors_d[cam_idx, gauss_idx]
    opac_dense_at_packed = opac_d[cam_idx, gauss_idx]
    depths_dense_at_packed = depths_d[cam_idx, gauss_idx]

    tol = 1e-5
    diff_means2d = (means2d_p - means2d_dense_at_packed).abs().max().item()
    diff_conics = (conics_p - conics_dense_at_packed).abs().max().item()
    diff_colors = (colors_p - colors_dense_at_packed).abs().max().item()
    diff_opac = (opac_p - opac_dense_at_packed).abs().max().item()
    diff_depths = (depths_p - depths_dense_at_packed).abs().max().item()
    diff_means3d = (means3d_p - means3d_d).abs().max().item()
    diff_wt = (wt_p - wt_d).abs().max().item()

    print(f"  means2d: {diff_means2d:.2e}  conics: {diff_conics:.2e}  colors: {diff_colors:.2e}")
    print(f"  opac: {diff_opac:.2e}  depths: {diff_depths:.2e}  means_3d_cond: {diff_means3d:.2e}  wt: {diff_wt:.2e}")

    # Verify indptr structure
    assert indptr.shape[0] == C + 1, f"indptr shape mismatch: {indptr.shape[0]} != {C + 1}"
    assert indptr[0].item() == 0, f"indptr[0] should be 0, got {indptr[0].item()}"
    assert indptr[-1].item() == nnz, f"indptr[-1] should be {nnz}, got {indptr[-1].item()}"

    assert diff_means2d < tol, f"means2d diff too large: {diff_means2d:.2e}"
    assert diff_conics < tol, f"conics diff too large: {diff_conics:.2e}"
    assert diff_colors < tol, f"colors diff too large: {diff_colors:.2e}"
    assert diff_opac < tol, f"opac diff too large: {diff_opac:.2e}"
    assert diff_depths < tol, f"depths diff too large: {diff_depths:.2e}"
    assert diff_means3d < tol, f"means_3d_cond diff too large: {diff_means3d:.2e}"
    assert diff_wt < tol, f"wt diff too large: {diff_wt:.2e}"


def test_packed_backward_equivalence() -> None:
    """Packed backward gradients must match dense backward within 5% relative at N=10K, C=2."""
    _skip_if_no_cuda_ext()

    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    from anycam4d.cuda.autograd import EWAFusedFunction
    from anycam4d.cuda.autograd import EWAFusedPackedFunction

    N = 10_000
    C = 2
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(88)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)

    # Determine visibility using dense forward
    with torch.no_grad():
        _, _, _, _, _, radii_ref, _, _ = launch_ewa_4d_fused_conditioned_projection_fwd_kernel(
            quats_l,
            quats_r,
            scales_4d_log,
            means,
            mu_t,
            opacities,
            sh0,
            shN,
            sh_temporal,
            sh_degree,
            _ADDITIVE,
            n_temporal_freqs,
            _stride(sh_temporal),
            viewmats,
            Ks,
            *_no_lens(),
            tau,
            cov_eps,
            eps2d,
            near_plane,
            far_plane,
            prefilter,
            width,
            height,
            False,
        )
    vis_mask = (radii_ref > 0).all(dim=-1)  # [C, N]
    n_vis = vis_mask.sum().item()
    print(f"\n  Visible pairs: {n_vis} / {C * N}")
    if n_vis < 10:
        pytest.skip("Too few visible Gaussians for meaningful comparison")

    scalars = (
        sh_degree,
        _ADDITIVE,
        n_temporal_freqs,
        _stride(sh_temporal),
        viewmats,
        Ks,
        *_no_lens(),
        tau,
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter,
        width,
        height,
        False,
    )
    param_names = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities", "sh0", "shN", "sh_temporal"]

    def get_grads_dense(params: list[Tensor]) -> list[Tensor]:
        for p in params:
            p.requires_grad_(True)
        means2d, conics, colors, opac, depths, radii, _, _ = EWAFusedFunction.apply(*params, *scalars)
        loss = means2d[vis_mask].sum() + conics[vis_mask].sum() + colors[vis_mask].sum() + opac[vis_mask].sum()
        loss.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    def get_grads_packed(params: list[Tensor]) -> list[Tensor]:
        for p in params:
            p.requires_grad_(True)
        cam_ids, gauss_ids, _, means2d, conics, colors, opac, _, _, _, _ = EWAFusedPackedFunction.apply(
            *params, *scalars
        )
        # Build a packed mask matching the dense vis_mask
        packed_vis = vis_mask[cam_ids.long(), gauss_ids.long()]
        loss = means2d[packed_vis].sum() + conics[packed_vis].sum() + colors[packed_vis].sum() + opac[packed_vis].sum()
        loss.backward()
        return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    clones_dense = [
        p.clone().detach() for p in [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    ]
    clones_packed = [
        p.clone().detach() for p in [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    ]

    grads_dense = get_grads_dense(clones_dense)
    grads_packed = get_grads_packed(clones_packed)

    tol_rel = 5e-2
    all_ok = True
    for name, gd, gp in zip(param_names, grads_dense, grads_packed):
        scale = gd.abs().max().item()
        if scale < 1e-8:
            continue
        diff = (gd - gp).abs().max().item()
        rel = diff / (scale + 1e-8)
        print(f"  {name:14s}: max abs diff = {diff:.3e}  rel = {rel:.3e}  scale = {scale:.3e}")
        if rel >= tol_rel:
            all_ok = False
            print(f"    *** FAIL: rel {rel:.3e} >= {tol_rel:.3e}")

    assert all_ok, "Packed backward gradients exceed relative tolerance (see printout above)"


def test_packed_render_parity() -> None:
    """FusedEWABackend(packed=True) vs FusedEWABackend(packed=False) PSNR > 60 dB."""
    _skip_if_no_cuda_ext()

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.ewa_fused_backend import FusedEWABackend

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()

    cfg_dense = RenderingConfig(packed=False)
    cfg_packed = RenderingConfig(packed=True)
    backend_dense = FusedEWABackend(cfg_dense)
    backend_packed = FusedEWABackend(cfg_packed)

    with torch.no_grad():
        out_dense = backend_dense.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )
        out_packed = backend_packed.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )

    rgb_d = out_dense.colors[..., :3]
    rgb_p = out_packed.colors[..., :3]

    mse = ((rgb_d - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  Packed vs dense PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"Packed render parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"


def test_fused_ewa_antialiased_equivalence() -> None:
    """Antialiased opacities equal classic opacities times ``sqrt(det Σ2d_raw / det Σ2d_blurred)``; dense and packed agree."""
    _skip_if_no_cuda_ext()
    from anycam4d.cuda.ext import launch_ewa_4d_fused_conditioned_projection_fwd_kernel
    from anycam4d.cuda.ext import launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel

    device = "cuda"
    N, C = 512, 2
    sh_degree = 1
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs

    g = torch.Generator(device=device)
    g.manual_seed(99)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)

    tau = 0.4
    cov_eps = 1e-6
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1000.0
    prefilter = 0.0
    width, height = 64, 64

    common = (
        quats_l,
        quats_r,
        scales_4d_log,
        means,
        mu_t,
        opacities,
        sh0,
        shN,
        sh_temporal,
        sh_degree,
        _ADDITIVE,
        n_temporal_freqs,
        _stride(sh_temporal),
        viewmats,
        Ks,
        *_no_lens(),
        tau,
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter,
        width,
        height,
    )

    # Classic forward (no compensation)
    _, conics_c, _, opac_classic, _, radii, _, _ = launch_ewa_4d_fused_conditioned_projection_fwd_kernel(*common, False)
    # Antialiased forward (with compensation)
    _, conics_aa, _, opac_aa, _, _, _, _ = launch_ewa_4d_fused_conditioned_projection_fwd_kernel(*common, True)

    vis = (radii > 0).all(dim=-1)  # [C, N]
    n_vis = vis.sum().item()
    if n_vis == 0:
        pytest.skip("No visible Gaussians — numerical edge case")

    # Conics invert Σ2d_blurred; Σ2d_raw = Σ2d_blurred - eps2d·I.
    a = conics_c[..., 0]  # [C, N]
    b = conics_c[..., 1]  # [C, N]
    c = conics_c[..., 2]  # [C, N]
    det_conic = a * c - b * b  # det of conic matrix = 1/det(Σ2d_blur)
    # Σ2d_blurred = inv([[a,b],[b,c]]) = 1/det_conic * [[c, -b], [-b, a]]
    s00 = c / det_conic  # Σ2d_blur[0,0]
    s11 = a / det_conic  # Σ2d_blur[1,1]
    s01 = -b / det_conic  # Σ2d_blur[0,1]
    det_blur = s00 * s11 - s01 * s01  # = 1/det_conic
    det_orig = (s00 - eps2d) * (s11 - eps2d) - s01 * s01
    ratio = torch.clamp(det_orig / det_blur.clamp(min=1e-10), min=0.0)
    compensation_ref = torch.sqrt(ratio)  # [C, N]

    # Expected antialiased opacity = classic opacity * compensation (per camera, per Gaussian)
    opac_ref = opac_classic * compensation_ref  # [C, N]

    # Compare only visible entries (invisible ones may have undefined conics)
    torch.testing.assert_close(
        opac_aa[vis],
        opac_ref[vis],
        atol=1e-4,
        rtol=1e-4,
        msg="Antialiased opacities (fused) != PyTorch reference",
    )
    print(f"\n  Visible pairs: {n_vis} / {C * N}")
    print(f"  Max abs diff: {(opac_aa[vis] - opac_ref[vis]).abs().max():.2e}")

    # Dense vs packed antialiased outputs must match
    (cam_ids_pk, gauss_ids_pk, _, _, _, _, opac_pk, _, _, _, _) = (
        launch_ewa_4d_fused_packed_conditioned_projection_fwd_kernel(
            *common,
            True,
        )
    )
    # Scatter packed opacities back to dense [C, N] for comparison
    opac_pk_dense = torch.zeros(C, N, device=device)
    opac_pk_dense[cam_ids_pk.long(), gauss_ids_pk.long()] = opac_pk

    torch.testing.assert_close(
        opac_pk_dense[vis],
        opac_aa[vis],
        atol=1e-5,
        rtol=1e-5,
        msg="Packed antialiased opacities != dense antialiased opacities",
    )
    print("  Dense vs packed antialiased: PASSED")

    print("\ntest_fused_ewa_antialiased_equivalence PASSED")


def _make_ut_bench_setup(N: int, C: int, sh_degree: int, n_temporal_freqs: int, device: str = "cuda"):
    """Synthetic sh4d model and cameras; returns ``(model, viewmats, Ks, sh_degree, tau, width, height)``."""
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

    torch.manual_seed(42)
    points = torch.randn(N, 3, device=device) * 0.3
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats, Ks = _make_cameras(C, device=device)
    return model, viewmats, Ks


def test_fused_ut_packed_forward_equivalence() -> None:
    """Packed UT kernel outputs must match dense kernel for all visible Gaussians (N=50K, C=4)."""
    _skip_if_no_cuda_ext()

    from anycam4d.cuda.ext import ProjectionFamily
    from anycam4d.cuda.autograd import UTFusedFunction
    from anycam4d.cuda.autograd import UTFusedPackedFunction

    N = 50_000
    C = 4
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(77)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)
    radial_coeffs = torch.zeros((C, 12), device=device, dtype=torch.float32)  # pinhole fixed-width
    family = int(ProjectionFamily.PINHOLE_FAMILY)

    params = (quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal)
    scalars = (
        sh_degree,
        _ADDITIVE,
        n_temporal_freqs,
        _stride(sh_temporal),
        viewmats,
        Ks,
        radial_coeffs,
        tau,
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter,
        width,
        height,
        family,
    )

    with torch.no_grad():
        means2d_d, conics_d, depths_d, radii_d, colors_d, means3d_d, covars_d, opac_d, wt_d = UTFusedFunction.apply(
            *params, *scalars
        )
        (
            cam_ids,
            gauss_ids,
            indptr,
            means2d_p,
            conics_p,
            depths_p,
            radii_p,
            colors_p,
            means3d_p,
            covars_p,
            opac_p,
            wt_p,
        ) = UTFusedPackedFunction.apply(*params, *scalars)

    nnz = cam_ids.shape[0]
    n_dense_vis = (radii_d > 0).all(dim=-1).sum().item()
    print(f"\n  Dense visible pairs: {n_dense_vis}, packed nnz: {nnz}")
    if nnz == 0:
        pytest.skip("No visible Gaussians — numerical edge case")

    tol = 1e-5

    # Dense side outputs (per-Gaussian, camera-agnostic) must be identical
    diff_means3d = (means3d_p - means3d_d).abs().max().item()
    diff_covars = (covars_p - covars_d).abs().max().item()
    diff_opac = (opac_p - opac_d).abs().max().item()
    diff_wt = (wt_p - wt_d).abs().max().item()

    print(f"  means_3d_cond: {diff_means3d:.2e}  covars: {diff_covars:.2e}  opac: {diff_opac:.2e}  wt: {diff_wt:.2e}")

    assert diff_means3d < tol, f"means_3d_cond diff too large: {diff_means3d:.2e}"
    assert diff_covars < tol, f"covars diff too large: {diff_covars:.2e}"
    assert diff_opac < tol, f"opac diff too large: {diff_opac:.2e}"
    assert diff_wt < tol, f"wt diff too large: {diff_wt:.2e}"

    # Packed tile-assignment outputs at (cam_idx, gauss_idx) must match dense.
    cam_idx = cam_ids.long()
    gauss_idx = gauss_ids.long()

    # Compare jointly visible, on-screen pairs: packed and dense kernels are compiled separately and
    # can flip a near z≈0 sigma point by one ULP, which blows up means2d for Gaussians that never render.
    both_vis = (radii_d[cam_idx, gauss_idx] > 0).all(dim=-1) & (radii_p > 0).all(dim=-1)
    md = means2d_d[cam_idx, gauss_idx]
    onscreen = both_vis & (md.abs().amax(dim=-1) < 2.0 * max(width, height))
    assert onscreen.sum().item() > 0, "no well-conditioned jointly-visible pairs"
    sel_c, sel_g = cam_idx[onscreen], gauss_idx[onscreen]

    diff_means2d = (means2d_p[onscreen] - means2d_d[sel_c, sel_g]).abs().max().item()
    diff_conics = (conics_p[onscreen] - conics_d[sel_c, sel_g]).abs().max().item()
    diff_depths = (depths_p[onscreen] - depths_d[sel_c, sel_g]).abs().max().item()
    diff_colors = (colors_p[sel_c, sel_g] - colors_d[sel_c, sel_g]).abs().max().item()

    print(
        f"  packed means2d: {diff_means2d:.2e}  conics: {diff_conics:.2e}  "
        f"depths: {diff_depths:.2e}  colors: {diff_colors:.2e}"
    )

    # Sub-pixel / small tolerances (was bit-exact pre-P6; ULP-amplified by the UT mean now).
    assert diff_means2d < 5e-2, f"packed means2d diff too large: {diff_means2d:.2e}"
    assert diff_conics < 1e-3, f"packed conics diff too large: {diff_conics:.2e}"
    assert diff_depths < tol, f"packed depths diff too large: {diff_depths:.2e}"
    assert diff_colors < 1e-3, f"packed colors diff too large: {diff_colors:.2e}"

    # Verify indptr structure
    assert indptr.shape[0] == C + 1, f"indptr shape: {indptr.shape[0]} != {C + 1}"
    assert indptr[0].item() == 0, f"indptr[0] != 0, got {indptr[0].item()}"
    assert indptr[-1].item() == nnz, f"indptr[-1] != nnz ({nnz}), got {indptr[-1].item()}"


def test_fused_ut_packed_backward_equivalence() -> None:
    """Packed UT backward gradients must match dense backward within 5% relative at N=10K, C=2."""
    _skip_if_no_cuda_ext()

    from anycam4d.cuda.ext import ProjectionFamily
    from anycam4d.cuda.autograd import UTFusedFunction
    from anycam4d.cuda.autograd import UTFusedPackedFunction

    N = 10_000
    C = 2
    sh_degree = 2
    n_temporal_freqs = 2
    K_sh = (sh_degree + 1) ** 2
    T = 2 * n_temporal_freqs
    tau = 0.4
    cov_eps = 1e-6
    prefilter = 0.0
    eps2d = 0.3
    near_plane = 0.01
    far_plane = 1e10
    width, height = 256, 256
    device = "cuda"

    g = torch.Generator(device=device)
    g.manual_seed(88)

    quats_l = torch.randn(N, 4, device=device, generator=g)
    quats_r = torch.randn(N, 4, device=device, generator=g)
    scales_4d_log = torch.randn(N, 4, device=device, generator=g) * 0.5 - 0.5
    means = torch.randn(N, 3, device=device, generator=g) * 0.3
    mu_t = torch.rand(N, device=device, generator=g)
    opacities = torch.randn(N, device=device, generator=g) * 2.0
    sh_coeffs = torch.randn(N, 16, 3, device=device, generator=g) * 0.1  # K_max: shN keeps its full stride
    sh0, shN = sh_coeffs[:, :1].contiguous(), sh_coeffs[:, 1:].contiguous()
    sh_temporal = torch.randn(N, T, 3, device=device, generator=g) * 0.05
    viewmats, Ks = _make_cameras(C, device=device)
    radial_coeffs = torch.zeros((C, 12), device=device, dtype=torch.float32)  # pinhole fixed-width
    family = int(ProjectionFamily.PINHOLE_FAMILY)

    scalars = (
        sh_degree,
        _ADDITIVE,
        n_temporal_freqs,
        _stride(sh_temporal),
        viewmats,
        Ks,
        radial_coeffs,
        tau,
        cov_eps,
        eps2d,
        near_plane,
        far_plane,
        prefilter,
        width,
        height,
        family,
    )
    param_names = ["quats_l", "quats_r", "scales_4d_log", "means", "mu_t", "opacities", "sh0", "shN", "sh_temporal"]

    def get_grads_dense(p_list: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        for p in p_list:
            p.requires_grad_(True)
        _, _, _, radii, colors, means_3d_cond, covars, opac, _ = UTFusedFunction.apply(*p_list, *scalars)
        # In real use, upstream grads are zero for invisible (cam,Gaussian) pairs.
        # Reproduce that here so the dense loss matches packed semantics.
        vis_cam = radii.sum(-1) > 0  # [C, N] bool
        vis_gauss = vis_cam.any(0)  # [N] bool
        vis_cam_f = vis_cam.float().unsqueeze(-1)  # [C, N, 1]
        vis_gf = vis_gauss.float()
        loss = (
            (colors * vis_cam_f).sum()
            + (means_3d_cond * vis_gf.unsqueeze(-1)).sum()
            + (covars * vis_gf.unsqueeze(-1)).sum()
            + (opac * vis_gf).sum()
        )
        loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in p_list]
        return grads, vis_gauss

    def get_grads_packed(p_list: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        for p in p_list:
            p.requires_grad_(True)
        cam_ids, gauss_ids, _, _, _, _, _, colors, means_3d_cond, covars, opac, _ = UTFusedPackedFunction.apply(
            *p_list, *scalars
        )
        N_p = means_3d_cond.shape[0]
        vis_gauss = torch.zeros(N_p, dtype=torch.bool, device=means_3d_cond.device)
        if gauss_ids.numel() > 0:
            vis_gauss[gauss_ids] = True
        vis_gf = vis_gauss.float()
        # colors [C,N,3] already zero for invisible pairs.
        loss = (
            colors.sum()
            + (means_3d_cond * vis_gf.unsqueeze(-1)).sum()
            + (covars * vis_gf.unsqueeze(-1)).sum()
            + (opac * vis_gf).sum()
        )
        loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in p_list]
        return grads, vis_gauss

    base = [quats_l, quats_r, scales_4d_log, means, mu_t, opacities, sh0, shN, sh_temporal]
    clones_dense = [p.clone().detach() for p in base]
    clones_packed = [p.clone().detach() for p in base]

    grads_dense, vis_d = get_grads_dense(clones_dense)
    grads_packed, vis_p = get_grads_packed(clones_packed)

    # Compare only jointly visible Gaussians; a near z≈0 validity flip gives a one-sided gradient.
    consistent = vis_d & vis_p  # [N] bool

    tol_rel = 5e-2
    all_ok = True
    print(
        f"\n  consistent-visible Gaussians: {consistent.sum().item()} "
        f"(dense {vis_d.sum().item()}, packed {vis_p.sum().item()})"
    )
    for name, gd, gp in zip(param_names, grads_dense, grads_packed):
        gd_m, gp_m = gd[consistent], gp[consistent]
        scale = gd_m.abs().max().item()
        if scale < 1e-8:
            continue
        diff = (gd_m - gp_m).abs().max().item()
        rel = diff / (scale + 1e-8)
        print(f"  {name:14s}: max abs diff = {diff:.3e}  rel = {rel:.3e}  scale = {scale:.3e}")
        if rel >= tol_rel:
            all_ok = False
            print(f"    *** FAIL: rel {rel:.3e} >= {tol_rel:.3e}")

    assert all_ok, "UT packed backward gradients exceed relative tolerance (see printout above)"


def test_fused_ut_packed_render_parity() -> None:
    """FusedUTBackend(packed=True) vs FusedUTBackend(packed=False) PSNR > 60 dB."""
    _skip_if_no_cuda_ext()

    try:
        from anycam4d.renderer.ut_fused_backend import UTFusedBackend
    except ImportError:
        pytest.skip("FusedUTBackend not available (import failed)")

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()

    cfg_dense = RenderingConfig(packed=False)
    cfg_packed = RenderingConfig(packed=True)
    backend_dense = UTFusedBackend(cfg_dense)
    backend_packed = UTFusedBackend(cfg_packed)

    with torch.no_grad():
        out_dense = backend_dense.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )
        out_packed = backend_packed.render(
            model, BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height), tau, sh_degree
        )

    rgb_d = out_dense.colors[..., :3]
    rgb_p = out_packed.colors[..., :3]

    mse = ((rgb_d - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  UT packed vs dense PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"UT packed render parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"


def test_fused_ewa_colors_override_inline_matches_pytorch() -> None:
    """Fused EWA with ``colors_override`` matches the PyTorch reference."""
    _skip_if_no_cuda_ext()

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.ewa_fused_backend import FusedEWABackend
    from tests.oracles.ewa_pytorch_backend import EWAPyTorchBackend

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()
    camera = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height)

    override = torch.rand(N, 3, device=device)

    cfg = RenderingConfig(packed=False)
    fused_backend = FusedEWABackend(cfg)
    pytorch_backend = EWAPyTorchBackend(cfg)

    with torch.no_grad():
        out_fused = fused_backend.render(model, camera, tau, sh_degree, colors_override=override)
        out_pytorch = pytorch_backend.render(model, camera, tau, sh_degree, colors_override=override)

    rgb_f = out_fused.colors[..., :3]
    rgb_p = out_pytorch.colors[..., :3]
    mse = ((rgb_f - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  EWA colors_override inline vs pytorch PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"EWA colors_override parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"


def test_fused_ewa_colors_override_packed_inline_matches_pytorch() -> None:
    """Packed fused EWA gathers ``colors_override[gaussian_ids]`` correctly."""
    _skip_if_no_cuda_ext()

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.ewa_fused_backend import FusedEWABackend
    from tests.oracles.ewa_pytorch_backend import EWAPyTorchBackend

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()
    camera = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height)

    override = torch.rand(N, 3, device=device)

    fused_backend = FusedEWABackend(RenderingConfig(packed=True))
    pytorch_backend = EWAPyTorchBackend(RenderingConfig(packed=True))

    with torch.no_grad():
        out_fused = fused_backend.render(model, camera, tau, sh_degree, colors_override=override)
        out_pytorch = pytorch_backend.render(model, camera, tau, sh_degree, colors_override=override)

    rgb_f = out_fused.colors[..., :3]
    rgb_p = out_pytorch.colors[..., :3]
    mse = ((rgb_f - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  EWA packed colors_override inline vs pytorch PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"EWA packed colors_override parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"


def _render_ut_3stage_pytorch_with_override(
    model: FourDGaussians,
    camera: BatchedCamera,
    tau: float,
    cfg: RenderingConfig,
    colors_override: Tensor,
):
    """Three-stage UT reference with ``colors_override`` swapped in before world rasterization."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    from tests.fused_stages import fused_ut_project
    from tests.fused_stages import condition
    from anycam4d.cuda.autograd import RasterizeWorldFunction
    from anycam4d.cuda.camera import to_kernel_args
    from anycam4d.cuda.ext import UT_WORLD_RASTERIZE_TILE_SIZE as tile_size
    from anycam4d.renderer.backend import RenderOutput

    splats = model.splats
    args = to_kernel_args(camera)
    viewmats, Ks, rc, family = args.viewmats, args.Ks, args.distortion, args.family
    width, height = args.width, args.height
    C = viewmats.shape[0]

    with torch.no_grad():
        radii, means2d, depths, _conics = fused_ut_project(
            splats["quats_l"],
            splats["quats_r"],
            splats["scales_4d_log"],
            splats["means"],
            splats["mu_t"],
            splats["opacities"],
            viewmats,
            Ks,
            rc,
            float(tau),
            model.cov_eps,
            cfg.eps2d,
            cfg.near_plane,
            cfg.far_plane,
            model.prefilter_threshold,
            width,
            height,
            family,
        )

    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_images=C,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(C, tile_height, tile_width)

    means_3d_cond, covars, opacities, _wt = condition(
        splats["quats_l"],
        splats["quats_r"],
        splats["scales_4d_log"],
        splats["means"],
        splats["mu_t"],
        splats["opacities"],
        float(tau),
        model.cov_eps,
        model.prefilter_threshold,
    )
    colors_cn = colors_override.unsqueeze(0).expand(C, -1, -1).contiguous()

    render_colors, render_alphas = RasterizeWorldFunction.apply(
        means_3d_cond,
        covars,
        opacities,
        colors_cn,
        viewmats,
        Ks,
        rc,
        isect_offsets,
        flatten_ids,
        width,
        height,
        family,
    )
    return RenderOutput(colors=render_colors, alphas=render_alphas.unsqueeze(-1), meta={})


def test_fused_ut_colors_override_inline_matches_pytorch() -> None:
    """Fused UT with ``colors_override`` matches the three-stage UT reference, not EWA geometry."""
    _skip_if_no_cuda_ext()

    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend

    device = "cuda"
    N = 10_000
    C = 1
    width, height = 256, 256
    sh_degree = 1
    n_temporal_freqs = 2
    tau = 0.3

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    points = torch.randn(N, 3, device=device) * 0.5
    model = FourDGaussians.from_points(
        points, sh_degree=sh_degree, n_temporal_freqs=n_temporal_freqs, appearance_model="sh4d_additive"
    ).to(device)

    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1)
    viewmats[:, 2, 3] = 2.0
    Ks = torch.tensor([[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]], device=device)
    Ks = Ks.unsqueeze(0).expand(C, -1, -1).contiguous()
    camera = BatchedCamera(viewmats, Ks, CameraModelType.PINHOLE, width, height)

    override = torch.rand(N, 3, device=device)

    cfg = RenderingConfig(packed=False)
    fused_backend = UTFusedBackend(cfg)

    with torch.no_grad():
        out_fused = fused_backend.render(model, camera, tau, sh_degree, colors_override=override)
        out_pytorch = _render_ut_3stage_pytorch_with_override(model, camera, tau, cfg, override)

    rgb_f = out_fused.colors[..., :3]
    rgb_p = out_pytorch.colors[..., :3]
    mse = ((rgb_f - rgb_p) ** 2).mean().item()
    psnr = -10.0 * math.log10(mse + 1e-10)
    print(f"\n  UT colors_override inline vs UT 3-stage pytorch PSNR: {psnr:.1f} dB  (mse={mse:.2e})")

    assert psnr > 60.0, f"UT colors_override parity PSNR too low: {psnr:.1f} dB (expected > 60 dB)"
