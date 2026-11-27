"""``OGFourDGSStrategy`` against transcriptions of the published 4DGS densifier.

The oracles follow https://github.com/fudan-zvg/4d-gaussian-splatting in its own idiom (``build_rotation_4d``,
``densify_and_clone``/``densify_and_split``, ``densify_and_prune``, ``reset_opacity``,
``add_densification_stats``) so they diff line by line against it. ``ref_build_rotation_4d`` is reproduced from
that repository (MIT License, Copyright (c) 2024 Fudan Zhang Vision Group).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from anycam4d.enums import SpatialTrigger
from anycam4d.math.covariance import build_rotation_4d
from anycam4d.math.covariance import cov4d_factor_from_rot4d
from anycam4d.strategy._ops import accumulate_spatial_grad
from anycam4d.strategy._ops import intrinsic_spatial_scales
from anycam4d.strategy._ops import prune_opacity_and_scale
from anycam4d.strategy.coupled import OGFourDGSStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup


# ref_build_rotation_4d is reproduced from utils/general_utils.py of https://github.com/fudan-zvg/4d-gaussian-splatting
# under the MIT License:
#
# Copyright (c) 2024 Fudan Zhang Vision Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
def ref_build_rotation_4d(l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:  # noqa: E741
    """utils/general_utils.py:build_rotation_4d, verbatim."""
    l_norm = torch.norm(l, dim=-1, keepdim=True)
    r_norm = torch.norm(r, dim=-1, keepdim=True)

    q_l = l / l_norm
    q_r = r / r_norm

    a, b, c, d = q_l.unbind(-1)
    p, q, r_, s = q_r.unbind(-1)

    M_l = torch.stack([a, -b, -c, -d, b, a, -d, c, c, d, a, -b, d, -c, b, a]).view(4, 4, -1).permute(2, 0, 1)
    M_r = torch.stack([p, q, r_, s, -q, p, -s, r_, -r_, s, p, -q, -s, -r_, q, p]).view(4, 4, -1).permute(2, 0, 1)
    A = M_l @ M_r
    return A.flip(1, 2)


def ref_select(
    grads: torch.Tensor,
    grad_threshold: float,
    scaling: torch.Tensor,
    percent_dense: float,
    scene_extent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Published clone/split selection: gradient ``>=`` threshold; clone on scale ``<=``, split on ``>``."""
    big = scaling.max(dim=1).values > percent_dense * scene_extent
    high = grads >= grad_threshold
    return high & ~big, high & big


def ref_split_children(
    xyzt: torch.Tensor,
    scaling_xyzt: torch.Tensor,
    rot_l: torch.Tensor,
    rot_r: torch.Tensor,
    samples_unit: torch.Tensor,
    n_children: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Published ``rot_4d`` split; ``samples_unit`` replaces the ``torch.normal(0, stds)`` draw."""
    stds = scaling_xyzt.repeat(n_children, 1)
    samples = samples_unit * stds
    rots = ref_build_rotation_4d(rot_l, rot_r).repeat(n_children, 1, 1)
    new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyzt.repeat(n_children, 1)
    new_scaling = torch.log(scaling_xyzt[:, :3].repeat(n_children, 1) / (0.8 * n_children))
    new_scaling_t = torch.log(scaling_xyzt[:, 3:].repeat(n_children, 1) / (0.8 * n_children))
    return new_xyzt, new_scaling, new_scaling_t


def ref_prune_mask(
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    extent: float,
    min_opacity: float,
    max_radii2d: torch.Tensor | None,
    max_screen_size: float | None,
) -> torch.Tensor:
    """densify_and_prune's prune half. ``max_screen_size`` is ``None`` before the first reset."""
    prune = opacity < min_opacity
    if max_screen_size:
        prune = prune | (max_radii2d > max_screen_size) | (scaling.max(dim=1).values > 0.1 * extent)
    return prune


def ref_reset_opacity(opacity_logit: torch.Tensor) -> torch.Tensor:
    """reset_opacity: ``inverse_sigmoid(min(sigmoid(x), 0.01))``."""
    return torch.logit(torch.clamp(torch.sigmoid(opacity_logit), max=0.01))


def ref_accumulate(ndc_grad: torch.Tensor) -> torch.Tensor:
    """add_densification_stats: the norm of the NDC screen-space gradient's first two components."""
    return ndc_grad[:, :2].norm(dim=-1)


N_CHILDREN = 2  # the published N=2; both sides hard-code it


def _population(n: int, seed: int = 0) -> dict[str, nn.Parameter]:
    """A toy population with genuinely anisotropic 4D geometry (identity quats hide the rotation)."""
    g = torch.Generator().manual_seed(seed)
    quats_l = torch.randn(n, 4, generator=g)
    quats_r = torch.randn(n, 4, generator=g)
    return {
        "means": nn.Parameter(torch.randn(n, 3, generator=g)),
        "mu_t": nn.Parameter(torch.rand(n, generator=g) - 0.5),
        "quats_l": nn.Parameter(torch.nn.functional.normalize(quats_l, dim=-1)),
        "quats_r": nn.Parameter(torch.nn.functional.normalize(quats_r, dim=-1)),
        "scales_4d_log": nn.Parameter(torch.randn(n, 4, generator=g) * 0.5 - 3.0),
        "opacities": nn.Parameter(torch.full((n,), 4.0)),
    }


def _group(params: dict[str, nn.Parameter]) -> DenseAdamGroup:
    return DenseAdamGroup({k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()})


def test_the_4d_rotation_matches_the_published_construction() -> None:
    """``build_rotation_4d`` reproduces the published matrix, ``flip(1, 2)`` included."""
    params = _population(64, seed=7)
    ours = build_rotation_4d(params["quats_l"], params["quats_r"])
    theirs = ref_build_rotation_4d(params["quats_l"], params["quats_r"])
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_the_split_offset_law_matches_the_published_sampler() -> None:
    """Given the same unit normal, ``L_4d @ eps`` equals the published ``R_4d @ (eps * stds)``."""
    params = _population(32, seed=3)
    scaling_xyzt = torch.exp(params["scales_4d_log"])
    eps = torch.randn(N_CHILDREN * 32, 4, generator=torch.Generator().manual_seed(11))

    factor = cov4d_factor_from_rot4d(params["quats_l"], params["quats_r"], params["scales_4d_log"])
    ours = torch.bmm(factor.repeat(N_CHILDREN, 1, 1), eps.unsqueeze(-1)).squeeze(-1)

    xyzt = torch.cat([params["means"], params["mu_t"].unsqueeze(-1)], dim=-1)
    theirs, _, _ = ref_split_children(xyzt, scaling_xyzt, params["quats_l"], params["quats_r"], eps, N_CHILDREN)
    theirs = theirs - xyzt.repeat(N_CHILDREN, 1)

    assert torch.allclose(ours, theirs, atol=1e-6)


def test_the_child_scales_match_the_published_shrink() -> None:
    """All four child scales are the parent's divided by ``0.8 N``."""
    params = _population(16, seed=5)
    scaling_xyzt = torch.exp(params["scales_4d_log"])
    eps = torch.zeros(N_CHILDREN * 16, 4)
    _, ref_s_xyz, ref_s_t = ref_split_children(
        torch.cat([params["means"], params["mu_t"].unsqueeze(-1)], dim=-1),
        scaling_xyzt,
        params["quats_l"],
        params["quats_r"],
        eps,
        N_CHILDREN,
    )
    ours = params["scales_4d_log"].repeat(N_CHILDREN, 1) - math.log(0.8 * N_CHILDREN)
    assert torch.allclose(ours[:, :3], ref_s_xyz, atol=1e-6)
    assert torch.allclose(ours[:, 3:], ref_s_t, atol=1e-6)


def test_the_split_consumes_its_parents() -> None:
    """A split replaces each parent with N children; the published code prunes the parent set."""
    n = 12
    params = _population(n, seed=9)
    group = _group(params)
    strategy = OGFourDGSStrategy(
        spatial_trigger=SpatialTrigger.SCREENSPACE,
        grow_grad2d=0.0,
        grow_scale3d=0.0,  # every primitive is "large" -> split, never clone
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100_000,
    )
    state = strategy.initialize_state(scene_scale=1.0)
    state["grad2d"] = torch.ones(n)
    state["count"] = torch.ones(n)
    n_dupli, n_split = strategy._grow_gs(params, group, state)
    assert (n_dupli, n_split) == (0, n)
    assert len(params["means"]) == n * N_CHILDREN


def test_the_clone_split_dispatch_matches_the_published_criteria() -> None:
    """Clone vs split uses the published signal, scale basis and comparisons."""
    n = 40
    params = _population(n, seed=13)
    group = _group(params)
    scene_scale, percent_dense, grad_thr = 1.1, 0.01, 2.0e-4

    # Straddle the scale threshold: half the population well under it, half well over.
    scales = params["scales_4d_log"].data
    scales[: n // 2, :3] = math.log(0.5 * percent_dense * scene_scale)
    scales[n // 2 :, :3] = math.log(2.0 * percent_dense * scene_scale)
    # Straddle the gradient threshold within each half.
    grads = torch.full((n,), grad_thr * 0.5)
    grads[::3] = grad_thr * 2.0

    strategy = OGFourDGSStrategy(
        spatial_trigger=SpatialTrigger.SCREENSPACE,
        grow_grad2d=grad_thr,
        grow_scale3d=percent_dense,
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100_000,
    )
    state = strategy.initialize_state(scene_scale=scene_scale)
    state["grad2d"] = grads.clone()
    state["count"] = torch.ones(n)

    ref_clone, ref_split = ref_select(grads, grad_thr, intrinsic_spatial_scales(params), percent_dense, scene_scale)
    n_dupli, n_split = strategy._grow_gs(params, group, state)

    assert n_dupli == int(ref_clone.sum())
    assert n_split == int(ref_split.sum())
    assert n_dupli > 0
    assert n_split > 0


def test_a_gradient_exactly_at_the_threshold_densifies() -> None:
    """A gradient exactly on the threshold densifies (``>=``); exact hits occur in practice."""
    n = 4
    params = _population(n, seed=17)
    group = _group(params)
    grad_thr = 2.0e-4
    strategy = OGFourDGSStrategy(
        spatial_trigger=SpatialTrigger.SCREENSPACE,
        grow_grad2d=grad_thr,
        grow_scale3d=0.0,  # all large -> split
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100_000,
    )
    state = strategy.initialize_state(scene_scale=1.0)
    state["grad2d"] = torch.full((n,), grad_thr)
    state["count"] = torch.ones(n)
    _, n_split = strategy._grow_gs(params, group, state)
    assert n_split == n


def test_the_screenspace_accumulator_reproduces_the_published_ndc_gradient() -> None:
    """The pixel-space accumulator equals the published ``|dL/d(ndc)|``."""
    n, width, height = 16, 640, 480
    pixel_grad = torch.randn(1, n, 2) * 1e-3
    means2d = torch.zeros(1, n, 2, requires_grad=True)
    means2d.grad = pixel_grad.clone()

    state: dict = {"grad2d": None, "count": None}
    info = {
        "means2d": means2d,
        "width": width,
        "height": height,
        "n_cameras": 1,
        "radii": torch.ones(1, n, 2),  # all visible
    }
    accumulate_spatial_grad(state, info, n, packed=False)

    ndc_grad = torch.stack([pixel_grad[0, :, 0] * width / 2.0, pixel_grad[0, :, 1] * height / 2.0], dim=-1)
    assert torch.allclose(state["grad2d"], ref_accumulate(ndc_grad), atol=1e-9)
    assert torch.equal(state["count"], torch.ones(n))


@pytest.mark.parametrize("after_first_reset", [False, True])
def test_the_prune_mask_matches_the_published_criteria(after_first_reset: bool) -> None:
    """Prune on opacity always, on world-space size only after the first opacity reset."""
    n = 24
    params = _population(n, seed=19)
    group = _group(params)
    scene_scale, min_opacity = 1.1, 0.005

    params["scales_4d_log"].data[:, :3] = math.log(0.02 * scene_scale)  # baseline: all well inside
    params["opacities"].data[:6] = torch.logit(torch.tensor(min_opacity / 2))  # below the cut
    params["scales_4d_log"].data[6:12, :3] = math.log(0.2 * scene_scale)  # above 0.1 * extent

    ref = ref_prune_mask(
        torch.sigmoid(params["opacities"]),
        intrinsic_spatial_scales(params),
        scene_scale,
        min_opacity,
        torch.zeros(n),  # no primitive is over the 20 px screen bound; the size arm here is world-space
        20.0 if after_first_reset else None,
    )
    n_prune = prune_opacity_and_scale(
        params,
        group,
        {},
        opa_thr=min_opacity,
        spatial_scale_thr=0.1,
        scene_scale=scene_scale,
        after_first_reset=after_first_reset,
    )
    assert n_prune == int(ref.sum())
    assert n_prune == (12 if after_first_reset else 6)


def test_the_opacity_reset_matches_the_published_clamp() -> None:
    """``reset_opa(prune_opa * 2)`` is the published ``min(opacity, 0.01)`` at the shipped threshold."""
    logits = torch.linspace(-6.0, 6.0, 32)
    ours = torch.clamp(logits, max=torch.logit(torch.tensor(0.005 * 2.0)).item())
    assert torch.allclose(ours, ref_reset_opacity(logits), atol=1e-5)
