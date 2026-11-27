"""Factory for building models and appearance models from config."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import assert_never

from anycam4d.console import notice
from anycam4d.data.point_cloud import load_ply
from anycam4d.enums import ModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians


if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from anycam4d.config.model import ModelConfig
    from anycam4d.data.normalization import SceneNormalization


class ModelFactory:
    """Creates model and appearance model from config."""

    @staticmethod
    def create_model(
        ply_path: Path,
        norm: SceneNormalization,
        model_cfg: ModelConfig,
        device: torch.device,
    ) -> FourDGaussians:
        """Load point cloud, normalize, and construct the primitive selected by ``model_cfg.model_type``.

        Returns ``FourDGaussians`` on the requested device.
        """
        if not ply_path.exists():
            raise FileNotFoundError(f"No point cloud PLY found at {ply_path}")

        points, colors = load_ply(ply_path)
        points_norm = (points - norm.norm_center) / norm.norm_scale
        notice(f"  {len(points):,} points, norm_scale={norm.norm_scale:.4f}")

        match model_cfg.model_type:
            case ModelType.FOURD_GAUSSIANS:
                gaussians = FourDGaussians.from_points(
                    points_norm,
                    sh_degree=model_cfg.sh_degree,
                    init_opacity=model_cfg.init_opacity,
                    temporal_spread_init=model_cfg.temporal_spread_init,
                    appearance_model=model_cfg.appearance_model,
                    n_temporal_freqs=model_cfg.n_temporal_freqs,
                    cov_eps=model_cfg.cov_eps,
                    prefilter_threshold=model_cfg.prefilter_threshold,
                    mu_t_margin_sigmas=model_cfg.mu_t_margin_sigmas,
                    colors=colors,
                ).to(device)
                notice(f"  Model initialized: {len(gaussians):,} Gaussians")
                return gaussians
            case _:
                assert_never(model_cfg.model_type)

    @staticmethod
    def from_checkpoint(
        model_state: dict,
        model_type: ModelType,
        device: torch.device,
        *,
        source: str = "checkpoint",
    ) -> FourDGaussians:
        """Reconstruct the primitive selected by ``model_type``, in eval mode on ``device``.

        ``model_type`` is the declared primitive from the run's ``ModelConfig`` (persisted
        in the checkpoint config snapshot) — the same axis ``create_model`` selects on, so
        reconstruction does not couple to the render backend. Parameters and appearance
        metadata come from ``model_state`` (see ``state_dict_with_meta``). ``source`` names the file
        in error messages.
        """
        model: FourDGaussians
        match model_type:
            case ModelType.FOURD_GAUSSIANS:
                model = FourDGaussians.from_checkpoint(model_state, source=source)
            case _:
                assert_never(model_type)
        return model.to(device).eval()
