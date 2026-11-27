"""PyTorchEWA backend: default render backend using gsplat EWA splatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gsplat import rasterization

from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.conditioning import condition_at_t
from anycam4d.renderer.backend import RenderOutput


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

    from tests.oracles.sh4d import Sh4dAppearanceModel as AppearanceModel
    from anycam4d.camera.batched import BatchedCamera
    from anycam4d.config.rendering import RenderingConfig
    from anycam4d.primitives.fourdgaussians.conditioning import ConditionedGaussians
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians


def render_gaussians(
    gaussians: ConditionedGaussians,
    viewmats: Float[Tensor, "C 4 4"],
    Ks: Float[Tensor, "C 3 3"],
    width: int,
    height: int,
    sh_degree: int,
    cfg: RenderingConfig,
    colors_preeval: Float[Tensor, "N C"] | None = None,
    backgrounds: Float[Tensor, "C 3"] | None = None,
) -> RenderOutput:
    """Render conditioned 3D Gaussians through ``gsplat.rasterization``.

    Exactly one of ``cov_3d_cond`` and ``quats`` is set. ``colors_preeval`` skips SH evaluation.
    """
    opacity_cond = gaussians.opacity_cond  # Already contains sigmoid(raw) * w_t from condition_at_t()

    if colors_preeval is not None:
        colors_input = colors_preeval
        sh_degree_call: int | None = None
    else:
        colors_input = gaussians.colors
        sh_degree_call = sh_degree

    render_colors, render_alphas, meta = rasterization(
        means=gaussians.means_3d_cond,
        quats=None,
        scales=None,
        opacities=opacity_cond,
        colors=colors_input,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree_call,
        packed=cfg.packed,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        eps2d=cfg.eps2d,
        render_mode=cfg.render_mode.value,
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=cfg.rasterize_mode.value,
        camera_model=CameraModelType.PINHOLE.value,
        covars=gaussians.cov_3d_cond,
    )

    # Composite background via alpha blending (avoids gsplat shape bug with packed mode)
    if backgrounds is not None:
        bg = backgrounds[:, None, None, :]  # [C, 1, 1, 3] → broadcasts to [C, H, W, 3]
        render_colors[..., :3] = render_colors[..., :3] + bg * (1.0 - render_alphas)

    return RenderOutput(
        colors=render_colors,
        alphas=render_alphas,
        meta=meta,
    )


class EWAPyTorchBackend:
    """Reference EWA backend: PyTorch conditioning, appearance, gsplat rasterization."""

    def __init__(
        self,
        cfg: RenderingConfig,
        appearance_model: AppearanceModel | None = None,
    ) -> None:
        """Initialize with rendering config and optional appearance model."""
        self._cfg = cfg
        self._appearance_model = appearance_model

    @property
    def cfg(self) -> RenderingConfig:
        """Rendering configuration used by this backend."""
        return self._cfg

    def render(
        self,
        model: FourDGaussians,
        camera: BatchedCamera,
        tau: float,
        sh_degree: int,
        *,
        backgrounds: Float[Tensor, "C 3"] | None = None,
        colors_override: Float[Tensor, "N 3"] | None = None,
    ) -> RenderOutput:
        """Render a PINHOLE camera batch; other models raise ``ValueError``."""
        if camera.model != CameraModelType.PINHOLE:
            raise ValueError(f"EWAPyTorchBackend supports only PINHOLE, got {camera.model}")

        cond = condition_at_t(
            model,
            tau,
            prefilter_threshold=model.prefilter_threshold,
        )

        colors_preeval = colors_override
        if colors_preeval is None and self._appearance_model is not None:
            colors_preeval = self._appearance_model.eval_colors(
                model.splats, cond.means_3d_cond, camera.viewmats, tau, sh_degree
            )

        # Unpack the carrier only here, at the gsplat-API boundary (render_gaussians → rasterization).
        return render_gaussians(
            cond,
            camera.viewmats,
            camera.Ks,
            camera.width,
            camera.height,
            sh_degree,
            self._cfg,
            colors_preeval,
            backgrounds,
        )
