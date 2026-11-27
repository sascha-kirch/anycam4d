"""Parameter-space spatial densification trigger: accumulator, capability gate, and agreement with screen space on EWA."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from anycam4d.camera.batched import BatchedCamera
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.data.normalization import compute_normalization
from anycam4d.enums import BackendType
from anycam4d.enums import SpatialTrigger
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.renderer.render_registry import check_spatial_trigger_capability
from anycam4d.strategy._ops import accumulate_param_spatial_grad
from anycam4d.strategy._ops import accumulate_spatial_grad
from anycam4d.strategy.default import DefaultSpatialStrategy
from anycam4d.trainer.optimizers.group import DenseAdamGroup


_needs_cuda = pytest.mark.skipif(not HAS_CUDA_EXT, reason="EWA render path runs on the CUDA kernels")


def _make_params(n: int) -> tuple[dict[str, nn.Parameter], DenseAdamGroup]:
    """Minimal FourDGaussians-schema params + optimizer group (no rendering)."""
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    params: dict[str, nn.Parameter] = {
        "means": nn.Parameter(torch.randn(n, 3)),
        "quats_l": nn.Parameter(quats.clone()),
        "quats_r": nn.Parameter(quats.clone()),
        "scales_4d_log": nn.Parameter(torch.full((n, 4), -3.0)),
        "opacities": nn.Parameter(torch.full((n,), 4.0)),  # alive → never pruned
    }
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}
    return params, DenseAdamGroup(optimizers)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two 1D tensors (Pearson on ranks)."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def _render_info(means: torch.Tensor, grad: torch.Tensor, visible: torch.Tensor, packed: bool) -> dict:
    """One C = 1 render's meta for the param-space accumulator; the camera sits at the origin looking down +z."""
    info = {"means_grad": grad, "means_3d_cond": means, "viewmats": torch.eye(4)[None], "n_cameras": 1}
    if packed:
        ids = torch.where(visible)[0]
        info["gaussian_ids"], info["camera_ids"] = ids, torch.zeros_like(ids)
    else:
        info["radii"] = visible[None, :, None].float().expand(1, len(means), 2)
    return info


@pytest.mark.parametrize("packed", [False, True])
def test_param_spatial_grad_is_mean_over_visible_views(packed: bool) -> None:
    """``grad3d / count3d`` averages the per-render signal over the renders that saw a Gaussian, never absent ones.

    A primitive outside its temporal window has ``radii == 0``; counting that render would dilute its mean by its
    temporal visibility fraction, which in 4D switches growth off.
    """
    n, grad_scale = 6, 3.0
    state: dict = {"grad3d": None, "count3d": None}
    gen = torch.Generator().manual_seed(0)
    total, seen = torch.zeros(n), torch.zeros(n)
    for _ in range(4):  # renders
        means = torch.randn(n, 3, generator=gen) + torch.tensor([0.0, 0.0, 5.0])
        grad = torch.randn(n, 3, generator=gen)
        visible = torch.rand(n, generator=gen) > 0.5
        grad = grad * visible[:, None]  # rendering gives an absent primitive no gradient
        accumulate_param_spatial_grad(state, _render_info(means, grad, visible, packed), n, packed, grad_scale)
        total += grad.norm(dim=-1) * means.norm(dim=-1) / 2 * grad_scale
        seen += visible.float()

    torch.testing.assert_close(state["count3d"], seen)
    torch.testing.assert_close(state["grad3d"] / state["count3d"].clamp_min(1), total / seen.clamp_min(1))


@pytest.mark.parametrize("depth", [1.0, 2.0, 20.0])
def test_param_spatial_grad_is_invariant_to_depth(depth: float) -> None:
    """The same image-plane error gives the same signal at any depth, hence at any scene scale (pinhole, on-axis).

    Oracle: autograd through a pinhole projection ``u = f x / z`` with a fixed ``∂L/∂u``, so the world gradient
    shrinks as ``1/z`` and the distance factor must cancel it exactly.
    """
    focal, dl_du = 500.0, 1e-3
    means = torch.tensor([[0.0, 0.0, depth]], requires_grad=True)
    u = focal * means[:, 0] / means[:, 2]
    (dl_du * u).sum().backward()

    state: dict = {"grad3d": None, "count3d": None}
    info = _render_info(means.detach(), means.grad, torch.ones(1, dtype=torch.bool), packed=False)
    accumulate_param_spatial_grad(state, info, 1)
    torch.testing.assert_close(state["grad3d"], torch.tensor([focal * dl_du / 2]))


def test_param_spatial_grad_none_when_no_grad() -> None:
    """A missing means gradient leaves the accumulator untouched (no crash), mirroring the temporal path."""
    state: dict = {"grad3d": None, "count3d": None}
    accumulate_param_spatial_grad(state, {}, 5)
    assert state["grad3d"] is None


def test_default_paramspace_runs_without_means2d_grad() -> None:
    """Paramspace refine steps densify from a render meta that carries no differentiable means2d."""
    n = 8
    params, opt = _make_params(n)
    strategy = DefaultSpatialStrategy(
        spatial_trigger=SpatialTrigger.PARAMSPACE,
        grow_grad3d=1e-6,  # tiny → any gradient fires
        grow_scale3d=1e9,  # everything "small" → clone
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100000,  # avoid opacity reset in the window
    )
    state = strategy.initialize_state(scene_scale=1.0)
    ray_meta: dict = {"n_cameras": 1}  # UT-style meta: radii, no means2d

    n_before = len(params["means"])
    for step in range(1, 4):
        strategy.step_pre_backward(params, opt, state, step, ray_meta)  # must not require means2d
        params["means"].grad = torch.ones(len(params["means"]), 3)  # strong world-space signal
        ray_meta["means_grad"] = params["means"].grad
        ray_meta["radii"] = torch.ones(1, len(params["means"]), 2)
        ray_meta["means_3d_cond"] = params["means"].detach() + torch.tensor([0.0, 0.0, 5.0])
        ray_meta["viewmats"] = torch.eye(4)[None]
        strategy.accumulate(params, state, step, ray_meta)
        strategy.step_post_backward(params, opt, state, step, ray_meta)
    assert len(params["means"]) > n_before, "paramspace trigger failed to densify on ray-style meta"
    assert state["grad2d"] is None, "screen-space state must never be touched on the paramspace path"


def test_default_screenspace_still_works_on_ewa() -> None:
    """Screenspace trigger accumulates grad2d (not grad3d) and densifies from means2d.grad."""
    n = 8
    params, opt = _make_params(n)
    means2d = torch.randn(1, n, 2, requires_grad=True)
    info = {
        "width": 64,
        "height": 64,
        "n_cameras": 1,
        "radii": torch.ones(1, n, 2),
        "gaussian_ids": None,
        "means2d": means2d,
    }
    strategy = DefaultSpatialStrategy(
        spatial_trigger=SpatialTrigger.SCREENSPACE,
        grow_grad2d=0.0,  # fire for all visible
        grow_scale3d=1e9,  # clone
        refine_start_iter=0,
        refine_every=1,
        refine_stop_iter=1000,
        reset_every=100000,
    )
    state = strategy.initialize_state(scene_scale=1.0)

    strategy.step_pre_backward(params, opt, state, 1, info)
    assert means2d.requires_grad
    (means2d.sum()).backward()
    strategy.accumulate(params, state, 1, info)
    strategy.step_post_backward(params, opt, state, 1, info)

    assert state["grad2d"] is not None, "screenspace path must populate grad2d"
    assert state["grad3d"] is None, "screenspace path must not touch grad3d"
    assert len(params["means"]) > n, "screenspace trigger failed to densify"


@pytest.mark.parametrize("backend", [BackendType.ANYCAM4D])
def test_screenspace_rejected_on_non_ewa(backend: BackendType) -> None:
    """screenspace + a non-EWA backend is rejected with an actionable message."""
    with pytest.raises(ValueError, match="paramspace"):
        check_spatial_trigger_capability(backend, SpatialTrigger.SCREENSPACE)


def test_screenspace_accepted_on_ewa() -> None:
    """screenspace on EWA is accepted (no raise)."""
    check_spatial_trigger_capability(BackendType.EWA, SpatialTrigger.SCREENSPACE)


@pytest.mark.parametrize("backend", [BackendType.EWA, BackendType.ANYCAM4D])
def test_paramspace_accepted_on_all_backends(backend: BackendType) -> None:
    """paramspace is never gated — it works on every backend."""
    check_spatial_trigger_capability(backend, SpatialTrigger.PARAMSPACE)


@_needs_cuda
def test_paramspace_matches_screenspace_growth_on_ewa(tmp_dataset: Path) -> None:
    """On an EWA pinhole scene both triggers rank-correlate and select near-equal candidate sets."""
    device = torch.device("cuda")
    train_ds = MultiViewSequenceDataset(str(tmp_dataset), split="train", scale_factor=1)
    norm = compute_normalization(train_ds)
    train_ds.apply_viewmat_transform(norm.T_denorm)

    torch.manual_seed(0)
    model = FourDGaussians.from_random(n_points=4000, sh_degree=1).to(device)
    n = len(model)

    backend = BackendFactory.create(RenderingConfig(backend=BackendType.EWA, packed=False))
    pipeline = RenderPipeline(backend=backend)

    state: dict = {"grad2d": None, "count": None, "grad3d": None, "count3d": None}
    passes = 4
    for _ in range(passes):
        for i in range(len(train_ds)):
            sample = train_ds[i]
            cam = BatchedCamera(
                sample.viewmat.unsqueeze(0).to(device),
                sample.K.unsqueeze(0).to(device),
                sample.camera_model,
                sample.image.shape[1],
                sample.image.shape[0],
                distortion=None,
            )
            out = pipeline.render(model, cam, float(sample.timestamp), model.sh_degree)
            out.meta["means2d"].retain_grad()
            target = sample.image.to(device).unsqueeze(0)
            model.zero_grad(set_to_none=True)
            (out.colors - target).abs().mean().backward()
            accumulate_spatial_grad(state, out.meta, n)
            out.meta["means_grad"] = model.splats["means"].grad
            accumulate_param_spatial_grad(state, out.meta, n)

    visible = state["count"] > 0
    assert int(visible.sum()) > 20, "too few visible Gaussians to assess the gate"

    mean2d = (state["grad2d"] / state["count"].clamp_min(1))[visible]
    mean3d = (state["grad3d"] / state["count3d"].clamp_min(1))[visible]

    rho = _spearman(mean2d, mean3d)

    # Matched selection percentile → equal candidate budget by construction; measure set overlap.
    q = 0.7
    cand2d = mean2d > torch.quantile(mean2d, q)
    cand3d = mean3d > torch.quantile(mean3d, q)
    inter = int((cand2d & cand3d).sum())
    union = int((cand2d | cand3d).sum())
    jaccard = inter / max(union, 1)
    count_ratio = int(cand3d.sum()) / max(int(cand2d.sum()), 1)

    print(
        f"visible={int(visible.sum())} spearman={rho:.3f} jaccard@q{q}={jaccard:.3f} "
        f"count_ratio={count_ratio:.3f} mean3d_med={float(mean3d.median()):.3e} "
        f"mean3d_q70={float(torch.quantile(mean3d, q)):.3e}"
    )

    assert rho > 0.5, f"param-space vs screen-space rank correlation too weak: {rho:.3f}"
    assert jaccard > 0.4, f"candidate-set overlap too low: {jaccard:.3f}"
    assert 0.7 < count_ratio < 1.3, f"candidate count out of band: {count_ratio:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
