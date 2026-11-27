"""4D covariance math for the rot_4d parameterization. Quaternions are wxyz and normalized at use."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from anycam4d.math.temporal import TEMPORAL_VARIANCE_EPS


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def schur_condition(
    cov_4d: Float[Tensor, "N 4 4"],
    tau: float | Float[Tensor, ""],
    mu_t: Float[Tensor, "N"],
    *,
    eps: float = 1e-6,
) -> tuple[Float[Tensor, "N 3"], Float[Tensor, "N 3 3"]]:
    """Condition a 4D Gaussian on time: spatial mean correction and conditional 3D covariance.

        cov_t_marg_safe = cov_4d[3, 3] + TEMPORAL_VARIANCE_EPS
        mean_3d_corr    = cov_xt / cov_t_marg_safe * (tau - mu_t)
        cov_3d_cond     = cov_3d_marg - cov_xt cov_xtᵀ / cov_t_marg_safe + eps·I

    ``TEMPORAL_VARIANCE_EPS`` is shared with the temporal gate; ``eps`` (``cov_eps``) is a spatial PSD floor.

    Returns:
        ``(mean_3d_corr [N, 3], cov_3d_cond [N, 3, 3])``.
    """
    cov_3d_marg = cov_4d[:, :3, :3]  # [N, 3, 3]
    cov_xt = cov_4d[:, :3, 3]  # [N, 3]
    cov_t_marg_safe = cov_4d[:, 3, 3] + TEMPORAL_VARIANCE_EPS  # [N]

    dt = tau - mu_t  # [N]

    # Conditional mean correction
    v = cov_xt / cov_t_marg_safe[:, None]  # [N, 3]
    mean_3d_corr = v * dt[:, None]  # [N, 3]

    # Rank-1 covariance correction (outer product)
    rank1_corr = cov_xt[:, :, None] * cov_xt[:, None, :] / cov_t_marg_safe[:, None, None]  # [N, 3, 3]

    # Conditional covariance with PSD floor
    eye = torch.eye(3, dtype=cov_4d.dtype, device=cov_4d.device).unsqueeze(0)
    cov_3d_cond = cov_3d_marg - rank1_corr + eps * eye  # [N, 3, 3]

    return mean_3d_corr, cov_3d_cond


def build_rotation_4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 4 4"]:
    """4D rotation ``(M_l(q_l) @ M_r(q_r)).flip([1, 2])``, matching 4DGS ``build_rotation_4d``; always in SO(4)."""
    q_l = F.normalize(quats_l, dim=-1)  # [N, 4] wxyz
    q_r = F.normalize(quats_r, dim=-1)  # [N, 4] wxyz

    a, b, c, d = q_l[:, 0], q_l[:, 1], q_l[:, 2], q_l[:, 3]
    p, q_, r_, s = q_r[:, 0], q_r[:, 1], q_r[:, 2], q_r[:, 3]

    # Left-isoclinic matrix M_l (left Hamilton multiplication by q_l).
    M_l = torch.stack(
        [a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a],
        dim=-1,
    ).reshape(-1, 4, 4)

    # M_r = Q_right(q_r)^T; right-multiplies by conjugate of q_r.
    M_r = torch.stack(
        [p, q_, r_, s, -q_, p, -s, r_, -r_, s, p, -q_, -s, -r_, q_, p],
        dim=-1,
    ).reshape(-1, 4, 4)

    # Product then reversal permutation (matches original's A.flip(1, 2))
    A = torch.bmm(M_l, M_r)
    return A.flip([1, 2])


def cov4d_factor_from_rot4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 4 4"]:
    """Factor ``L = R_4d @ diag(exp(scales_4d_log))`` with ``L Lᵀ = cov_4d``.

    Prefer it to factoring the assembled covariance, which can be indefinite in float32.
    """
    R_4d = build_rotation_4d(quats_l, quats_r)  # [N, 4, 4]
    s = torch.exp(scales_4d_log)  # [N, 4], always positive
    return R_4d * s.unsqueeze(1)  # [N, 4, 4] — column k scaled by s_k, i.e. R_4d @ diag(s)


def intrinsic_spatial_factor_from_rot4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 3 3"]:
    """Factor ``R_4d[:3, :3] @ diag(exp(scales_4d_log[:, :3]))`` of the intrinsic spatial covariance."""
    R_4d = build_rotation_4d(quats_l, quats_r)  # [N, 4, 4]
    s_xyz = torch.exp(scales_4d_log[:, :3])  # [N, 3]
    return R_4d[:, :3, :3] * s_xyz.unsqueeze(1)  # [N, 3, 3] — column k scaled by s_k


def intrinsic_spatial_from_rot4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 3 3"]:
    """Intrinsic spatial covariance ``B Bᵀ``; independent of the temporal scale."""
    B = intrinsic_spatial_factor_from_rot4d(quats_l, quats_r, scales_4d_log)  # [N, 3, 3]
    return torch.bmm(B, B.transpose(-1, -2))  # [N, 3, 3]


def build_cov4d_rot(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 4 4"]:
    """4D covariance ``R_4d diag(exp(scales_4d_log))² R_4dᵀ``, matching 4DGS ``build_scaling_rotation_4d``."""
    L = cov4d_factor_from_rot4d(quats_l, quats_r, scales_4d_log)  # [N, 4, 4]
    return torch.bmm(L, L.transpose(-2, -1))  # [N, 4, 4]


def temporal_marginal_from_rot4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
) -> Float[Tensor, "N"]:
    """Marginal temporal variance ``cov_4d[3, 3] = Σ_k R_4d[3, k]² exp(2 s_k)``, without epsilon.

    A convex combination of all four scales, so bounding ``scales_4d_log[:, 3]`` does not bound it.
    """
    R_row_t = time_row_from_rot4d(quats_l, quats_r)  # [N, 4] — last row of R_4d
    var = torch.exp(2.0 * scales_4d_log)  # [N, 4] = diag(exp(2·s))
    return torch.einsum("nk,nk->n", R_row_t * R_row_t, var)  # [N]


def time_row_from_rot4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
) -> Float[Tensor, "N 4"]:
    """Row 3 of :func:`build_rotation_4d` without forming the 4x4."""
    a, b, c, d = F.normalize(quats_l, dim=-1).unbind(-1)
    p, q, r, s = F.normalize(quats_r, dim=-1).unbind(-1)
    return torch.stack(
        [
            a * s - b * r + c * q - d * p,
            a * r + b * s - c * p - d * q,
            a * q - b * p - c * s + d * r,
            a * p + b * q + c * r + d * s,
        ],
        dim=-1,
    )


def quat_multiply(a: Float[Tensor, "N 4"], b: Float[Tensor, "N 4"]) -> Float[Tensor, "N 4"]:
    """Hamilton product ``a * b`` of wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_from_rotation_3d(R: Float[Tensor, "N 3 3"]) -> Float[Tensor, "N 4"]:
    """Unit wxyz quaternion of a 3D rotation (Shepperd: largest of four candidates, so never a near-zero divide)."""
    m = R
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    candidates = torch.stack(
        [
            torch.stack([1 + trace, m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], -1),
            torch.stack(
                [
                    m[:, 2, 1] - m[:, 1, 2],
                    1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                    m[:, 0, 1] + m[:, 1, 0],
                    m[:, 0, 2] + m[:, 2, 0],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[:, 0, 2] - m[:, 2, 0],
                    m[:, 0, 1] + m[:, 1, 0],
                    1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                    m[:, 1, 2] + m[:, 2, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[:, 1, 0] - m[:, 0, 1],
                    m[:, 0, 2] + m[:, 2, 0],
                    m[:, 1, 2] + m[:, 2, 1],
                    1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
                ],
                -1,
            ),
        ],
        dim=1,
    )  # [N, 4 candidates, 4]; candidate i is q scaled by 4 q_i
    best = torch.stack(
        [
            1 + trace,
            1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
        ],
        -1,
    ).argmax(-1)
    q = candidates[torch.arange(len(m), device=m.device), best]
    return F.normalize(q, dim=-1)


def rot4d_from_matrix(R_4d: Float[Tensor, "N 4 4"]) -> tuple[Float[Tensor, "N 4"], Float[Tensor, "N 4"]]:
    """Inverse of :func:`build_rotation_4d` for ``R_4d`` in SO(4): ``(quats_l, quats_r)``, up to a joint sign.

    ``build_rotation_4d`` acts as ``x ↦ q_l x q̄_r`` (after the flip). With ``u = q_l q̄_r`` (the image of 1),
    ``x ↦ A(x) ū = q_l x q̄_l`` is the 3D rotation of ``q_l``, and ``q_r = ū q_l``.
    """
    A = R_4d.flip([1, 2])
    u_conj = A[:, :, 0] * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=A.dtype, device=A.device)
    B = torch.stack([quat_multiply(A[:, :, j], u_conj) for j in range(1, 4)], dim=-1)  # [N, 4, 3]
    quats_l = quat_from_rotation_3d(B[:, 1:, :])
    quats_r = quat_multiply(u_conj, quats_l)
    return quats_l, quats_r


def rot4d_from_cov4d(
    cov_4d: Float[Tensor, "N 4 4"],
) -> tuple[Float[Tensor, "N 4"], Float[Tensor, "N 4"], Float[Tensor, "N 4"]]:
    """Factor a 4D covariance into ``(quats_l, quats_r, scales_4d_log)``.

    The eigen-axis most aligned with time becomes axis 3, so ``scales_4d_log[:, 3]`` keeps its meaning as the
    temporal scale; one spatial axis is flipped when needed to stay in SO(4).
    """
    # On CPU: cusolver's batched eigh allocates ~0.5 MB of workspace per 4x4 matrix (16 GB for one refine).
    evals, evecs = torch.linalg.eigh(cov_4d.cpu())  # ascending, columns
    evals, evecs = evals.to(cov_4d.device), evecs.to(cov_4d.device)
    t_axis = evecs[:, 3, :].abs().argmax(-1)  # [N]
    order = torch.arange(4, device=cov_4d.device).expand(len(cov_4d), 4).clone()
    order = torch.cat([order[order != t_axis[:, None]].view(-1, 3), t_axis[:, None]], dim=1)
    evecs = torch.gather(evecs, 2, order[:, None, :].expand(-1, 4, -1))
    evals = torch.gather(evals, 1, order)
    flip = torch.linalg.det(evecs) < 0
    evecs[flip, :, 0] = -evecs[flip, :, 0]
    quats_l, quats_r = rot4d_from_matrix(evecs)
    scales_4d_log = 0.5 * torch.log(evals.clamp_min(torch.finfo(evals.dtype).tiny))
    return quats_l, quats_r, scales_4d_log


def shrink_conditional_4d(
    quats_l: Float[Tensor, "N 4"],
    quats_r: Float[Tensor, "N 4"],
    scales_4d_log: Float[Tensor, "N 4"],
    k: Float[Tensor, "N"],
    c: Float[Tensor, "N"] | None = None,
) -> tuple[Float[Tensor, "N 4"], Float[Tensor, "N 4"], Float[Tensor, "N 4"]]:
    """Scale the drawn (Schur-conditional) spatial shape by ``k`` and the temporal extent by ``c``; velocity is kept.

    Applies ``x ↦ k (x - v t) + c v t, t ↦ c t`` to the 4D Gaussian: ``Σ_cond → k² Σ_cond``, ``cov_tt → c² cov_tt``,
    ``v`` unchanged. ``c = None`` keeps the temporal extent. Computed in float64, returned in the input dtype.
    """
    dtype = scales_4d_log.dtype
    L = cov4d_factor_from_rot4d(quats_l.double(), quats_r.double(), scales_4d_log.double())
    cov = torch.bmm(L, L.transpose(-2, -1))
    v = cov[:, :3, 3] / cov[:, 3, 3:]
    k = k.double()
    c = torch.ones_like(k) if c is None else c.double()
    T = torch.eye(4, dtype=cov.dtype, device=cov.device).repeat(len(cov), 1, 1)
    T[:, :3, :3] *= k[:, None, None]
    T[:, :3, 3] = (c - k)[:, None] * v
    T[:, 3, 3] = c
    L_new = torch.bmm(T, L)
    q_l, q_r, s = rot4d_from_cov4d(torch.bmm(L_new, L_new.transpose(-2, -1)))
    # An axis thinner than float64 eigh resolves (~1e-16 of the largest variance) comes back ~0; no axis may end
    # thinner than the parent's thinnest shrunk by k (or c), or it underflows in float32 and stops getting gradient.
    floor = scales_4d_log.double().min(dim=-1, keepdim=True).values + torch.log(torch.minimum(k, c))[:, None]
    s = torch.maximum(s, floor)
    return q_l.to(dtype), q_r.to(dtype), s.to(dtype)
