"""Densification kernels shared by the strategies: split, prune, relocate, sample-add, noise.

Every parameter mutation goes through ``_update_param_with_optimizer`` so optimizer state follows it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _multinomial_sample
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.strategy.ops import remove
from torch import nn

from anycam4d.math.covariance import build_cov4d_rot
from anycam4d.math.covariance import cov4d_factor_from_rot4d
from anycam4d.math.covariance import intrinsic_spatial_factor_from_rot4d
from anycam4d.math.covariance import intrinsic_spatial_from_rot4d
from anycam4d.math.covariance import shrink_conditional_4d
from anycam4d.math.covariance import temporal_marginal_from_rot4d
from anycam4d.primitives.fourdgaussians.conditioning import temporal_marginal_std


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.trainer.optimizers.group import OptimizerGroupProto


def _append_zero_state(
    state: dict[str, Any],
    n_new: int,
    device: torch.device,
    rest: Int[Tensor, "R"] | None = None,
) -> None:
    """Append ``n_new`` zero rows to every per-Gaussian state tensor, first keeping only ``rest`` if given."""
    for name, v in state.items():
        if isinstance(v, torch.Tensor):
            base = v if rest is None else v[rest]
            state[name] = torch.cat([base, torch.zeros((n_new, *v.shape[1:]), device=device)])


def _zero_state_rows(
    state: dict[str, Any],
    idx: Int[Tensor, "K"],
) -> None:
    """Zero the given rows of every per-Gaussian state accumulator (relocated source rows)."""
    for v in state.values():
        if isinstance(v, torch.Tensor):
            v[idx] = 0


def _selection_stable_randn(
    n_children: int,
    sel: Int[Tensor, "M"],
    n_total: int,
    trailing: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "C M ..."]:
    """Split offsets ``[n_children, M, *trailing]`` gathered from a population-sized draw.

    Drawing only for the selected rows would let one borderline selection change (atomicAdd noise)
    shift the RNG stream for every other primitive. This bounds the effect to its own children; it
    does not make training reproducible.
    """
    pool = torch.randn(n_children, n_total, *trailing, device=device, dtype=dtype)
    return pool[:, sel]


def accumulate_spatial_grad(
    state: dict[str, Any],
    info: dict[str, Any],
    n_gaussians: int,
    packed: bool = False,
    grad_scale: float = 1.0,
) -> None:
    """Accumulate pixel-scaled ``|means2d.grad|`` per Gaussian into ``state["grad2d"]`` (EWA only).

    ``grad_scale`` undoes the ``loss / accum_renders`` scaling so the signal, and therefore
    ``grow_grad2d``, keeps its batch-1 meaning at any accumulation depth.
    """
    grads = info["means2d"].grad.clone() * grad_scale
    grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
    grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

    if state["grad2d"] is None:
        state["grad2d"] = torch.zeros(n_gaussians, device=grads.device)
    if state["count"] is None:
        state["count"] = torch.zeros(n_gaussians, device=grads.device)

    if packed:
        gs_ids = info["gaussian_ids"]
    else:
        sel = (info["radii"] > 0.0).all(dim=-1)
        gs_ids = torch.where(sel)[1]
        grads = grads[sel]

    state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
    state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))


def visible_views(info: dict[str, Any], n_gaussians: int, packed: bool) -> Bool[Tensor, "C N"]:
    """Which of this render's cameras saw each Gaussian (``radii > 0``: frustum and temporal gate)."""
    if not packed:
        return (info["radii"] > 0.0).all(dim=-1)
    seen = torch.zeros(info["n_cameras"], n_gaussians, dtype=torch.bool, device=info["gaussian_ids"].device)
    seen[info["camera_ids"], info["gaussian_ids"]] = True
    return seen


def accumulate_param_spatial_grad(
    state: dict[str, Any],
    info: dict[str, Any],
    n_gaussians: int,
    packed: bool = False,
    grad_scale: float = 1.0,
) -> None:
    """Accumulate the 3DGRUT signal ``|∂L/∂μ| · ‖μ_cond - camera‖ / 2`` into ``state["grad3d"]``, per visible view.

    The distance (to the conditioned mean at this render's τ) removes scene scale and depth, so the 3DGS threshold
    applies (``threedgrut/strategy/gs.py``). Exact at C = 1. With C > 1 the norm is taken after the cameras' gradients
    sum, so opposing views cancel and the signal reads low. ``grad_scale`` undoes ``loss / accum_renders``.
    """
    means_grad = info.get("means_grad")
    if means_grad is None:
        return
    seen = visible_views(info, n_gaussians, packed)  # [C, N]
    views = seen.sum(dim=0).float()
    viewmats = info["viewmats"]
    centers = -torch.einsum("cji,cj->ci", viewmats[:, :3, :3], viewmats[:, :3, 3])  # [C, 3], -Rᵀt
    dist = (info["means_3d_cond"][None] - centers[:, None]).norm(dim=-1)  # [C, N]
    mean_dist = (dist * seen).sum(dim=0) / views.clamp_min(1)
    g = means_grad.norm(dim=-1) * mean_dist / 2.0 * grad_scale * info["n_cameras"]
    if state["grad3d"] is None:
        state["grad3d"] = torch.zeros(n_gaussians, device=g.device)
        state["count3d"] = torch.zeros(n_gaussians, device=g.device)
    state["grad3d"] += torch.where(views > 0, g, 0.0)
    state["count3d"] += views


def intrinsic_spatial_scales(params: dict[str, nn.Parameter]) -> Float[Tensor, "N 3"]:
    """Spatial scale parameters ``exp(scales_4d_log[:, :3])``, the basis operators write back to; not the rendered extent."""
    return torch.exp(params["scales_4d_log"][:, :3])


def marginal_temporal_variance(params: dict[str, nn.Parameter]) -> Float[Tensor, "N"]:
    """Marginal temporal variance ``cov_4d[3, 3]``, the width the temporal gate divides by."""
    return temporal_marginal_from_rot4d(params["quats_l"], params["quats_r"], params["scales_4d_log"])


def marginal_temporal_std(params: dict[str, nn.Parameter]) -> Float[Tensor, "N"]:
    """Return ``[N]`` marginal temporal width ``std_t_marg``, with the renderer's epsilon included."""
    return temporal_marginal_std(marginal_temporal_variance(params))


def primitive_speed(params: dict[str, nn.Parameter]) -> Float[Tensor, "N"]:
    """``|v|`` with ``v = cov_xt / cov_tt``, the velocity of each primitive's conditioned mean."""
    cov = build_cov4d_rot(params["quats_l"], params["quats_r"], params["scales_4d_log"])
    return (cov[:, :3, 3] / cov[:, 3, 3:].clamp_min(torch.finfo(cov.dtype).tiny)).norm(dim=-1)


def percentile_rank(values: Float[Tensor, "N"]) -> Float[Tensor, "N"]:
    """Fraction of ``values`` strictly below each entry, in ``[0, 1)``; ties share the lowest rank."""
    ordered = torch.sort(values).values
    return torch.searchsorted(ordered, values.contiguous()).float() / values.numel()


def spatial_move_factor(
    quats_l: Float[Tensor, "M 4"],
    quats_r: Float[Tensor, "M 4"],
    scales_4d_log: Float[Tensor, "M 4"],
    *,
    shrink: float | None = None,
) -> Float[Tensor, "M 3 3"]:
    """Closed-form factor ``F`` of the intrinsic spatial covariance for a displacement ``F ε``.

    ``shrink`` (child scale ``c``) scales ``F`` by ``sqrt(1 - c²)`` so split children keep the parent's marginal.
    """
    factor = intrinsic_spatial_factor_from_rot4d(quats_l, quats_r, scales_4d_log)  # [M, 3, 3]
    if shrink is not None:
        factor = factor * math.sqrt(max(1.0 - shrink**2, 0.0))
    return factor


@torch.no_grad()
def split_spatial(
    params: dict[str, nn.Parameter],
    group: OptimizerGroupProto,
    state: dict[str, Any],
    mask: Bool[Tensor, "N"],
    N: int = 2,
) -> None:
    """Split selected Gaussians in space; temporal parameters are copied verbatim.

    Offsets come from the intrinsic spatial covariance, the only choice independent of ``scale_t``:
    with children inheriting time, ``cov_3d_marg - cov_3d_marg_child = (1 - c²) cov_3d_intrinsic``.
    Shrinking spatial scales still changes the conditional drift, so children never inherit the
    parent's motion exactly.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    M = len(sel)
    shrink = 1.0 / (0.8 * N)

    factor = spatial_move_factor(
        params["quats_l"][sel], params["quats_r"][sel], params["scales_4d_log"][sel], shrink=shrink
    )  # [M, 3, 3]

    K = factor.shape[-1]
    factor_exp = factor.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * M, 3, K)
    eps = _selection_stable_randn(N, sel, mask.numel(), (K,), device, factor.dtype).reshape(N * M, K)
    offsets_xyz = torch.bmm(factor_exp, eps.unsqueeze(-1)).squeeze(-1)  # [N*M, 3]

    new_scales_xyz_log = torch.log(torch.exp(params["scales_4d_log"][:, :3][sel]) * shrink).repeat(N, 1)

    def param_fn(name: str, p: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        p_ch = p[sel].repeat([N] + [1] * (p.dim() - 1))
        if name == "means":
            p_ch = p[sel].repeat(N, 1) + offsets_xyz
        elif name == "scales_4d_log":
            p_ch = torch.cat([new_scales_xyz_log, p[sel, 3:4].repeat(N, 1)], dim=-1)
        return nn.Parameter(torch.cat([p[rest], p_ch]), requires_grad=p.requires_grad)

    def optimizer_fn(_key: str, v: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.cat([v[rest], torch.zeros_like(v[sel]).repeat([N] + [1] * (v.dim() - 1))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, group.optimizers)
    _append_zero_state(state, N * M, device, rest=rest)


def _split_full_4d_offsets_fourdgauss(
    params: dict[str, nn.Parameter],
    L_4d: Float[Tensor, "M 4 4"],
    sel: Int[Tensor, "M"],
    N: int,
    M: int,
    device: torch.device,
    n_total: int,
    sigma_t_floor: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Offsets ``L_4d ε`` (a draw from ``N(0, cov_4d)``) and all four log-scales shrunk by ``log(0.8·N)``.

    A uniform shrink scales the whole 4D covariance, so the marginal temporal variance shrinks by the
    same ratio. ``sigma_t_floor`` floors only the temporal column.
    """
    scales_4d_log = params["scales_4d_log"][sel]  # [M, 4], log space
    eps = _selection_stable_randn(N, sel, n_total, (4,), device, scales_4d_log.dtype)  # [N, M, 4]
    L_exp = L_4d.unsqueeze(0).expand(N, -1, -1, -1).reshape(N * M, 4, 4)
    offsets_4d = torch.bmm(L_exp, eps.reshape(N * M, 4, 1)).squeeze(-1)
    shrunk = scales_4d_log - math.log(0.8 * N)  # [M, 4]
    if sigma_t_floor > 0.0:
        shrunk = torch.cat([shrunk[:, :3], shrunk[:, 3:].clamp_min(math.log(sigma_t_floor))], dim=-1)
    new_scales = {"scales_4d_log": shrunk.repeat(N, 1)}  # [N*M, 4]
    return offsets_4d, new_scales


@torch.no_grad()
def split_full_4d(
    params: dict[str, nn.Parameter],
    group: OptimizerGroupProto,
    state: dict[str, Any],
    mask: Bool[Tensor, "N"],
    N: int = 2,
    *,
    sigma_t_floor: float = 0.0,
) -> None:
    """Split with 4D-correlated offsets in ``means`` and ``mu_t`` (published 4DGS ``rot_4d=True``)."""
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    M = len(sel)
    L_4d = cov4d_factor_from_rot4d(
        params["quats_l"][sel], params["quats_r"][sel], params["scales_4d_log"][sel]
    )  # [M, 4, 4]

    offsets_4d, new_scales = _split_full_4d_offsets_fourdgauss(
        params, L_4d, sel, N, M, device, mask.numel(), sigma_t_floor
    )

    def param_fn(name: str, p: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        p_ch = p[sel].repeat([N] + [1] * (p.dim() - 1))
        if name == "means":
            p_ch = p[sel].repeat(N, 1) + offsets_4d[:, :3]
        elif name == "mu_t":
            p_ch = p[sel].repeat(N) + offsets_4d[:, 3]
        elif name in new_scales:
            p_ch = new_scales[name]
        return nn.Parameter(torch.cat([p[rest], p_ch]), requires_grad=p.requires_grad)

    def optimizer_fn(_key: str, v: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.cat([v[rest], torch.zeros_like(v[sel]).repeat([N] + [1] * (v.dim() - 1))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, group.optimizers)
    _append_zero_state(state, N * M, device, rest=rest)


def prune_opacity_and_scale(
    params: dict[str, nn.Parameter],
    group: OptimizerGroupProto,
    state: dict[str, Any],
    opa_thr: float,
    spatial_scale_thr: float,
    scene_scale: float,
    after_first_reset: bool,
) -> int:
    """Prune by opacity and, after the first reset, by spatial scale; returns the number removed.

    Removes ``sigmoid(opacity) < opa_thr`` and parameter scale above ``spatial_scale_thr · scene_scale``.
    """
    is_prune = torch.sigmoid(params["opacities"].flatten()) < opa_thr

    if after_first_reset:
        s_xyz_max = intrinsic_spatial_scales(params).max(dim=-1).values
        is_prune = is_prune | (s_xyz_max > spatial_scale_thr * scene_scale)

    n_prune = int(is_prune.sum().item())
    if n_prune > 0:
        remove(params=params, optimizers=group.optimizers, state=state, mask=is_prune)
    return n_prune


def _op_sigmoid(x: Float[Tensor, "..."], k: float = 100.0, x0: float = 0.995) -> Float[Tensor, "..."]:
    """Steep sigmoid gate used in MCMC noise injection (matches gsplat MCMCStrategy)."""
    return 1.0 / (1.0 + torch.exp(-k * (x - x0)))


# Noise is skipped for rows whose (near-binary) gate is below this; the RNG draw stays full-N.
NOISE_GATE_EPS = 1e-6


def _relocated_shapes(
    params: dict[str, nn.Parameter],
    idx: Int[Tensor, "M"],
    opacities: Float[Tensor, "M"],
    ratios: Int[Tensor, "M"],
    binoms: Float[Tensor, "n_max n_max"],
    birth_opacity: float,
    temporal_partition: bool = False,
) -> tuple[Float[Tensor, "M"], dict[str, Float[Tensor, "M ..."]]]:
    """MCMC relocation of ``idx``: new opacity and new shape rows (``quats_l``, ``quats_r``, ``scales_4d_log``).

    ``compute_relocation``'s spatial scale factor is applied to the drawn conditional shape, not to the intrinsic
    spatial scales. Copies keep their parent's velocity; their temporal extent is the parent's, or ``1 / ratio`` of
    it under ``temporal_partition``.
    """
    ones = torch.ones(len(idx), 3, device=opacities.device)
    new_opacities, k = compute_relocation(opacities=opacities, scales=ones, ratios=ratios, binoms=binoms)
    eps = torch.finfo(torch.float32).eps
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=birth_opacity)
    c = 1.0 / ratios.float() if temporal_partition else None
    q_l, q_r, s = shrink_conditional_4d(
        params["quats_l"][idx], params["quats_r"][idx], params["scales_4d_log"][idx], k[:, 0], c
    )
    return new_opacities, {"quats_l": q_l, "quats_r": q_r, "scales_4d_log": s}


def _partition_offsets(
    params: dict[str, nn.Parameter],
    idx: Int[Tensor, "M"],
    ratios: Int[Tensor, "M"],
) -> dict[str, Float[Tensor, "M ..."]]:
    """Independent temporal offsets that tile a parent's footprint among its ``ratio`` copies.

    ``Δ ~ N(0, (1 - 1/ratio²) cov_tt)`` keeps the group's temporal marginal; the spatial mean moves by ``v Δ`` so
    each piece stays on the parent's trajectory. A piece never lands farther outside the normalized sequence
    ``[-0.5, 0.5]`` than its parent: there it is drawn at no timestamp and still counts as alive.
    """
    cov = build_cov4d_rot(params["quats_l"][idx], params["quats_r"][idx], params["scales_4d_log"][idx])
    cov_tt = cov[:, 3, 3]
    v = cov[:, :3, 3] / cov_tt[:, None].clamp_min(torch.finfo(cov.dtype).tiny)
    var = (1.0 - ratios.float().pow(-2)) * cov_tt
    dt = torch.randn_like(cov_tt) * var.clamp_min(0).sqrt()
    mu_t = params["mu_t"][idx].flatten().to(dt.dtype)
    dt = torch.minimum(torch.maximum(dt, (-0.5 - mu_t).clamp_max(0)), (0.5 - mu_t).clamp_min(0))
    return {"mu_t": dt, "means": v * dt[:, None]}


def _apply_offset(p: Float[Tensor, "N ..."], rows: Int[Tensor, "M"], offset: Float[Tensor, "M ..."]) -> None:
    """``p[rows] += offset`` for a parameter stored as ``[N]`` or ``[N, 1]``; ``rows`` must be unique."""
    p[rows] += offset.view(-1, *p.shape[1:])


@torch.no_grad()
def relocate_spatial(
    params: dict[str, nn.Parameter],
    group: OptimizerGroupProto,
    state: dict[str, Any],
    mask: Bool[Tensor, "N"],
    binoms: Float[Tensor, "n_max n_max"],
    birth_opacity: float = 0.005,
    temporal_partition: bool = False,
    parent_weights: Float[Tensor, "N"] | None = None,
) -> None:
    """Move dead Gaussians onto sampled live parents (∝ ``parent_weights``, default opacity), in place.

    Opacity and the drawn spatial shape follow ``compute_relocation``; velocity is the parent's, and ``mu_t`` and
    temporal width too unless ``temporal_partition`` splits the parent's footprint among the group. Every tensor
    in ``state`` is treated as a per-Gaussian accumulator; its parent and relocated rows are zeroed.
    """
    opacities = torch.sigmoid(params["opacities"])
    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)
    if n == 0:
        return

    weights = (opacities.flatten() if parent_weights is None else parent_weights)[alive_indices]
    # multinomial does not validate weights; a NaN would surface later as an opaque CUDA assert.
    if not torch.isfinite(weights).all():
        raise RuntimeError(
            f"relocation weights are not finite ({int((~torch.isfinite(weights)).sum())} of "
            f"{weights.numel()} live primitives); opacities have gone NaN."
        )
    sampled_idxs = alive_indices[_multinomial_sample(weights, n)]
    counts = torch.bincount(sampled_idxs)
    ratios = counts[sampled_idxs] + 1
    if temporal_partition:
        parents = sampled_idxs.unique()
        offsets_parent = _partition_offsets(params, parents, counts[parents] + 1)
        offsets_child = _partition_offsets(params, sampled_idxs, ratios)
    new_opacities, shapes = _relocated_shapes(
        params, sampled_idxs, opacities[sampled_idxs].flatten(), ratios, binoms, birth_opacity, temporal_partition
    )

    def param_fn(name: str, p: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities).view(-1, *p.shape[1:])
        elif name in shapes:
            p[sampled_idxs] = shapes[name]
        p[dead_indices] = p[sampled_idxs]
        if temporal_partition and name in offsets_child:
            _apply_offset(p, dead_indices, offsets_child[name])
            _apply_offset(p, parents, offsets_parent[name])
        return nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(_key: str, v: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        v[sampled_idxs] = 0
        v[dead_indices] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, params, group.optimizers)
    _zero_state_rows(state, sampled_idxs)
    _zero_state_rows(state, dead_indices)


def sample_add_spatial(
    params: dict[str, nn.Parameter],
    group: OptimizerGroupProto,
    state: dict[str, Any],
    n: int,
    binoms: Float[Tensor, "n_max n_max"],
    birth_opacity: float = 0.005,
    temporal_partition: bool = False,
    parent_weights: Float[Tensor, "N"] | None = None,
) -> None:
    """Append ``n`` Gaussians sampled from parents ∝ ``parent_weights`` (default opacity), as in relocation."""
    if n <= 0:
        return
    opacities = torch.sigmoid(params["opacities"])
    weights = opacities.flatten() if parent_weights is None else parent_weights
    sampled_idxs = _multinomial_sample(weights, n)
    counts = torch.bincount(sampled_idxs)
    ratios = counts[sampled_idxs] + 1
    if temporal_partition:
        parents = sampled_idxs.unique()
        offsets_parent = _partition_offsets(params, parents, counts[parents] + 1)
        offsets_child = _partition_offsets(params, sampled_idxs, ratios)
    new_opacities, shapes = _relocated_shapes(
        params, sampled_idxs, opacities.flatten()[sampled_idxs], ratios, binoms, birth_opacity, temporal_partition
    )

    def param_fn(name: str, p: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities).view(-1, *p.shape[1:])
        elif name in shapes:
            p[sampled_idxs] = shapes[name]
        children = p[sampled_idxs].clone()
        if temporal_partition and name in offsets_child:
            children += offsets_child[name].view(-1, *p.shape[1:])
            _apply_offset(p, parents, offsets_parent[name])
        return nn.Parameter(torch.cat([p, children]), requires_grad=p.requires_grad)

    def optimizer_fn(_key: str, v: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return torch.cat([v, torch.zeros_like(v[sampled_idxs])])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, group.optimizers)
    _append_zero_state(state, n, params["means"].device)


def _add_spatial_noise(
    params: dict[str, nn.Parameter],
    gate: Float[Tensor, "N"],
    noise: Float[Tensor, "N 3"],
    scaler: float,
) -> None:
    """Add ``gate * scaler * cov_3d_intrinsic @ noise`` to ``means`` on the gated rows only."""
    idx = (gate > NOISE_GATE_EPS).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    build = intrinsic_spatial_from_rot4d
    cov_3d = build(params["quats_l"][idx], params["quats_r"][idx], params["scales_4d_log"][idx])  # [M, 3, 3]
    delta = gate[idx].unsqueeze(-1) * scaler * torch.einsum("bij,bj->bi", cov_3d, noise[idx])
    params["means"].data[idx] += delta


@torch.no_grad()
def inject_noise_spatial(
    params: dict[str, nn.Parameter],
    state: dict[str, Any],  # noqa: ARG001
    scaler: float,
) -> None:
    """Langevin noise on ``means``, gated to low-opacity (< ~0.005) primitives.

    Writes ``means.data`` directly without touching optimizer state.
    """
    opacities = torch.sigmoid(params["opacities"].flatten())
    gate = _op_sigmoid(1.0 - opacities)  # [N]
    noise = torch.randn_like(params["means"])  # [N, 3] full-N draw: stream independent of the gate
    _add_spatial_noise(params, gate, noise, scaler)
