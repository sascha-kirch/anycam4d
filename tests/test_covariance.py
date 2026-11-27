"""Tests for 4D covariance math: rot_4d parameterization and Schur complement conditioning."""

from __future__ import annotations

import math

import pytest
import torch

from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.covariance import build_rotation_4d
from anycam4d.math.covariance import cov4d_factor_from_rot4d
from anycam4d.math.covariance import intrinsic_spatial_factor_from_rot4d
from anycam4d.math.covariance import intrinsic_spatial_from_rot4d
from anycam4d.math.covariance import rot4d_from_cov4d
from anycam4d.math.covariance import rot4d_from_matrix
from anycam4d.math.covariance import schur_condition
from anycam4d.math.covariance import shrink_conditional_4d
from anycam4d.math.covariance import temporal_marginal_from_rot4d
from anycam4d.math.temporal import TEMPORAL_SIGMA_FLOOR
from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS
from anycam4d.primitives.fourdgaussians.conditioning import temporal_marginal_std
from anycam4d.strategy._ops import marginal_temporal_std as ops_marginal_temporal_std
from anycam4d.strategy._ops import marginal_temporal_variance as ops_marginal_temporal_variance


def test_schur_output_is_psd():
    """cov_3d_cond eigenvalues are all positive after Schur conditioning."""
    torch.manual_seed(4)
    N = 128
    ql = torch.randn(N, 4)
    qr = torch.randn(N, 4)
    scales = torch.randn(N, 4)
    Sigma = build_cov4d_rot(ql, qr, scales)
    mu_t = torch.rand(N)

    _, cov_3d_cond = schur_condition(Sigma, tau=0.5, mu_t=mu_t)

    eigs = torch.linalg.eigvalsh(cov_3d_cond)  # [N, 3]
    assert (eigs > 0).all(), f"Non-PSD cov_3d_cond: min eigenvalue = {eigs.min():.2e}"


def test_gradcheck_schur_condition():
    """torch.autograd.gradcheck passes for schur_condition at float64."""
    torch.manual_seed(42)
    N = 5
    ql = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    qr = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    scales = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    mu_t = torch.rand(N, dtype=torch.float64, requires_grad=True)
    tau = 0.5

    def fn(q_l: torch.Tensor, q_r: torch.Tensor, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        Sigma = build_cov4d_rot(q_l, q_r, s)
        _, cov_3d_cond = schur_condition(Sigma, tau, m)
        return cov_3d_cond

    assert torch.autograd.gradcheck(fn, (ql, qr, scales, mu_t), eps=1e-6, atol=1e-4)


def test_build_rotation_4d_is_orthogonal():
    """R @ R^T == I and det(R) == +1 for random unit quaternions."""
    torch.manual_seed(10)
    N = 100
    ql = torch.randn(N, 4)
    qr = torch.randn(N, 4)
    R = build_rotation_4d(ql, qr)  # [N, 4, 4]

    assert R.shape == (N, 4, 4)

    eye = torch.eye(4).unsqueeze(0).expand(N, -1, -1)
    RRt = torch.bmm(R, R.transpose(-2, -1))
    assert torch.allclose(RRt, eye, atol=1e-5), f"R @ R^T != I; max err = {(RRt - eye).abs().max():.2e}"

    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(N), atol=1e-5), f"det(R) != 1; max err = {(dets - 1).abs().max():.2e}"


def test_build_cov4d_rot_identity_rotation_is_diagonal():
    """With identity quaternions, cov_4d == diag(exp(scales_4d_log)^2)."""
    torch.manual_seed(12)
    N = 20
    ql = torch.zeros(N, 4)
    ql[:, 0] = 1.0
    qr = torch.zeros(N, 4)
    qr[:, 0] = 1.0
    scales = torch.randn(N, 4)

    cov = build_cov4d_rot(ql, qr, scales)

    expected_diag = torch.exp(scales) ** 2  # [N, 4]
    expected = torch.diag_embed(expected_diag)  # [N, 4, 4]

    assert torch.allclose(cov, expected, atol=1e-5), (
        f"Identity rotation should give diagonal cov; max off-diagonal = {(cov - expected).abs().max():.2e}"
    )


def test_gradcheck_rot4d_through_schur():
    """torch.autograd.gradcheck through build_cov4d_rot + schur_condition at float64."""
    torch.manual_seed(14)
    N = 4
    ql = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    qr = torch.randn(N, 4, dtype=torch.float64, requires_grad=True)
    scales = torch.zeros(N, 4, dtype=torch.float64, requires_grad=True)
    mu_t = torch.rand(N, dtype=torch.float64, requires_grad=True)
    tau = 0.5

    def fn(q_l: torch.Tensor, q_r: torch.Tensor, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        cov_4d = build_cov4d_rot(q_l, q_r, s)
        _, cov_3d_cond = schur_condition(cov_4d, tau, m)
        return cov_3d_cond

    assert torch.autograd.gradcheck(fn, (ql, qr, scales, mu_t), eps=1e-6, atol=1e-4)


def _random_rot4d_params(n: int, seed: int = 7):
    torch.manual_seed(seed)
    return torch.randn(n, 4), torch.randn(n, 4), torch.randn(n, 4)


def test_cov4d_factor_reconstructs_the_4d_covariance():
    """``L Lᵀ == cov_4d``: the factor is a square root of the full 4D covariance."""
    ql, qr, s = _random_rot4d_params(64)
    L = cov4d_factor_from_rot4d(ql, qr, s)
    assert torch.allclose(torch.bmm(L, L.transpose(-2, -1)), build_cov4d_rot(ql, qr, s), atol=1e-5)


def test_intrinsic_factor_reconstructs_the_intrinsic_covariance():
    """``B Bᵀ == cov_3d_intrinsic``: B is a square root, so ``B eps`` samples that covariance."""
    ql, qr, s = _random_rot4d_params(64)
    B = intrinsic_spatial_factor_from_rot4d(ql, qr, s)
    assert B.shape == (64, 3, 3)
    assert torch.allclose(torch.bmm(B, B.transpose(-2, -1)), intrinsic_spatial_from_rot4d(ql, qr, s), atol=1e-5)


def test_temporal_marginal_matches_the_full_4d_covariance():
    """``temporal_marginal_from_rot4d == build_cov4d_rot(...)[:, 3, 3]`` to float32 round-off."""
    ql, qr, s = _random_rot4d_params(512)
    ref = build_cov4d_rot(ql, qr, s)[:, 3, 3]
    got = temporal_marginal_from_rot4d(ql, qr, s)
    assert got.shape == (512,)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_temporal_marginal_matches_on_degenerate_inputs():
    """Single primitive, identical left/right quaternions, and scales spanning several decades."""
    # N = 1.
    ql, qr, s = _random_rot4d_params(1)
    assert torch.allclose(temporal_marginal_from_rot4d(ql, qr, s), build_cov4d_rot(ql, qr, s)[:, 3, 3], rtol=1e-5)

    # Identical quaternions (q_l == q_r), which collapses the isoclinic pair to one rotation.
    q = torch.randn(32, 4)
    s = torch.randn(32, 4)
    assert torch.allclose(temporal_marginal_from_rot4d(q, q, s), build_cov4d_rot(q, q, s)[:, 3, 3], rtol=1e-5)

    # Scales spanning six decades — the regime where the discarded terms would matter if the
    # einsum were an approximation rather than the same sum.
    ql, qr, _ = _random_rot4d_params(64, seed=3)
    wide = torch.linspace(-7.0, 7.0, 64).unsqueeze(-1) * torch.tensor([1.0, -1.0, 0.5, -0.25])
    ref = build_cov4d_rot(ql, qr, wide)[:, 3, 3]
    assert torch.allclose(temporal_marginal_from_rot4d(ql, qr, wide), ref, rtol=1e-5)


def test_temporal_marginal_normalizes_quaternions_like_build_cov4d_rot():
    """Unnormalized input must agree — normalization is applied at use time by convention."""
    ql, qr, s = _random_rot4d_params(64, seed=11)
    ql, qr = ql * 7.5, qr * 0.03
    ref = build_cov4d_rot(ql, qr, s)[:, 3, 3]
    assert torch.allclose(temporal_marginal_from_rot4d(ql, qr, s), ref, rtol=1e-5)


def _time_row_on_first_spatial_axis() -> tuple[torch.Tensor, torch.Tensor]:
    """Quaternions ``(0, 0, 0, 1)``, ``(1, 0, 0, 0)``: ``R_4d[3] = (∓1, 0, 0, 0)``, so ``cov_t_marg = exp(2 s_x)``."""
    return torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def test_a_spatial_axis_can_own_the_time_row_entirely():
    ql, qr = _time_row_on_first_spatial_axis()
    row_t = build_rotation_4d(ql, qr)[0, 3, :]
    assert torch.allclose(row_t.abs(), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    # s_x = +2 (a broad spatial axis), s_t = -5 (a razor-thin temporal parameter).
    s = torch.tensor([[2.0, 0.0, 0.0, -5.0]])
    cov_t = temporal_marginal_from_rot4d(ql, qr, s)
    assert torch.allclose(cov_t, torch.exp(torch.tensor([4.0])), rtol=1e-5)
    # Four orders of magnitude apart from what the intrinsic scale claims.
    assert cov_t.item() / math.exp(-10.0) > 1e4


def test_temporal_marginal_agrees_with_gradients_of_the_full_path():
    """Same value *and* same gradient as differentiating ``build_cov4d_rot(...)[:, 3, 3]``."""
    ql, qr, s = _random_rot4d_params(64, seed=13)
    grads = []
    for fn in (temporal_marginal_from_rot4d, lambda a, b, c: build_cov4d_rot(a, b, c)[:, 3, 3]):
        a, b, c = ql.clone().requires_grad_(True), qr.clone().requires_grad_(True), s.clone().requires_grad_(True)
        fn(a, b, c).square().sum().backward()
        grads.append((a.grad, b.grad, c.grad))
    for lhs, rhs in zip(*grads):
        assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-5)


def test_temporal_marginal_std_carries_the_render_epsilon():
    """``std_t_marg == sqrt(cov_t_marg + TEMPORAL_VARIANCE_EPS)``, and floors a collapsed primitive."""
    ql, qr, s = _random_rot4d_params(64, seed=17)
    cov_t = temporal_marginal_from_rot4d(ql, qr, s)
    assert torch.allclose(temporal_marginal_std(cov_t), torch.sqrt(cov_t + TEMPORAL_VARIANCE_EPS))

    collapsed = torch.zeros(4)
    assert torch.allclose(temporal_marginal_std(collapsed), torch.full((4,), TEMPORAL_SIGMA_FLOOR))


def test_temporal_accessors_agree_across_all_three_call_shapes(random_model):
    """math / params-dict / model accessor are the same number, and the model form is ``[N]``."""
    splats = random_model.splats
    from_math = temporal_marginal_from_rot4d(splats["quats_l"], splats["quats_r"], splats["scales_4d_log"])
    from_params = ops_marginal_temporal_variance(splats)
    from_model = random_model.marginal_temporal_variance()

    assert from_model.shape == (splats["means"].shape[0],)
    assert torch.allclose(from_math, from_params)
    assert torch.allclose(from_math, from_model)
    assert torch.allclose(random_model.marginal_temporal_std(), ops_marginal_temporal_std(splats))
    assert torch.allclose(random_model.marginal_temporal_std(), temporal_marginal_std(from_model))


def test_rot4d_inverse_round_trips_through_build_rotation_4d():
    """rot4d_from_matrix inverts build_rotation_4d; rot4d_from_cov4d refactors a covariance and keeps axis 3 temporal."""
    torch.manual_seed(11)
    ql, qr = torch.randn(500, 4, dtype=torch.float64), torch.randn(500, 4, dtype=torch.float64)
    R = build_rotation_4d(ql, qr)
    assert torch.allclose(build_rotation_4d(*rot4d_from_matrix(R)), R, atol=1e-12)

    scales = torch.randn(500, 4, dtype=torch.float64) - 2.0
    cov = build_cov4d_rot(ql, qr, scales)
    q_l, q_r, s = rot4d_from_cov4d(cov)
    assert torch.allclose(build_cov4d_rot(q_l, q_r, s), cov, rtol=1e-10, atol=1e-14)
    time_row = build_rotation_4d(q_l, q_r)[:, 3, :].abs()
    assert (time_row.argmax(-1) == 3).all(), "axis 3 must be the eigen-axis most aligned with time"


@pytest.mark.parametrize("time_factor", [False, True])
def test_shrink_conditional_4d_keeps_velocity_and_scales_shape_and_time(time_factor: bool):
    """Velocity is unchanged, the exact Schur conditional scales by k² and Σ_tt by c² (c = 1 when not given)."""
    torch.manual_seed(12)
    n = 500
    ql, qr = torch.randn(n, 4, dtype=torch.float64), torch.randn(n, 4, dtype=torch.float64)
    scales = torch.randn(n, 4, dtype=torch.float64) - 2.0
    k = 0.5 + 0.5 * torch.rand(n, dtype=torch.float64)
    c = 0.2 + 0.8 * torch.rand(n, dtype=torch.float64) if time_factor else None
    before = build_cov4d_rot(ql, qr, scales)
    after = build_cov4d_rot(*shrink_conditional_4d(ql, qr, scales, k, c))

    def exact_conditional(cov: torch.Tensor) -> torch.Tensor:
        return cov[:, :3, :3] - cov[:, :3, 3:] * cov[:, 3:, :3] / cov[:, 3:, 3:]

    c2 = torch.ones(n, dtype=torch.float64) if c is None else c**2
    assert torch.allclose(after[:, 3, 3], c2 * before[:, 3, 3], rtol=1e-10)
    v_before, v_after = before[:, :3, 3] / before[:, 3:, 3], after[:, :3, 3] / after[:, 3:, 3]
    assert torch.allclose(v_after, v_before, rtol=1e-7, atol=1e-12)
    target = k[:, None, None] ** 2 * exact_conditional(before)
    assert torch.allclose(exact_conditional(after), target, rtol=1e-6, atol=1e-14)

    # An axis below float64 eigh resolution must not collapse to a zero scale (an eigh-based solver returned -354).
    thin = torch.tensor([-22.0, 0.5, 0.5, 0.5], dtype=torch.float64).repeat(64, 1)
    k_thin = torch.full((64,), 0.8, dtype=torch.float64)
    *_, s_thin = shrink_conditional_4d(ql[:64], qr[:64], thin, k_thin)
    assert s_thin.min() >= -22.0 + math.log(0.8) - 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device workspace check")
def test_shrink_conditional_4d_device_memory_is_linear_in_rows():
    """One MCMC refine relocates ~30 k rows; a per-matrix solver workspace ran a 16 GB GPU out of memory."""
    n = 30_000
    ql, qr = torch.randn(n, 4, device="cuda"), torch.randn(n, 4, device="cuda")
    scales, k = torch.randn(n, 4, device="cuda") - 3.0, torch.full((n,), 0.8, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    shrink_conditional_4d(ql, qr, scales, k)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base < 64 * 2**20
