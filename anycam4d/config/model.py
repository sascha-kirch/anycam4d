"""Model config dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from anycam4d.enums import AppearanceType
from anycam4d.enums import ModelType


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class ModelConfig:
    """Configuration for the 4D primitive model."""

    model_type: ModelType = ModelType.FOURD_GAUSSIANS
    sh_degree: int = 3
    init_opacity: float = 0.1

    # -1.0: initial temporal scale of 5 frames (5 / n_frames).
    temporal_spread_init: float = -1.0

    # Bound on |mu_t| beyond the sequence, in units of the temporal width; -1 derives it from
    # prefilter_threshold. No off value: an unbounded mu_t random-walks out of the sequence.
    mu_t_margin_sigmas: float = -1.0

    prefilter_threshold: float = 0.05

    # Spatial PSD floor on the conditional covariance; sqrt(cov_eps) is the narrowest axis the renderer draws.
    cov_eps: float = 1e-6

    appearance_model: AppearanceType = AppearanceType.SH4D_ADDITIVE

    n_temporal_freqs: int = 4

    def __post_init__(self) -> None:
        # Every kernel divides by the diagonal of the covariance factor, which cov_eps bounds below by sqrt(cov_eps).
        if self.cov_eps <= 0:
            raise ValueError(f"model.cov_eps must be > 0, got {self.cov_eps}")

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> ModelConfig:
        """Create ModelConfig from OmegaConf DictConfig."""
        raw = OmegaConf.to_container(cfg, resolve=True)
        model_type_raw = raw.get("model_type", ModelType.FOURD_GAUSSIANS.value)
        return ModelConfig(
            model_type=ModelType(model_type_raw),
            sh_degree=raw["sh_degree"],
            init_opacity=raw["init_opacity"],
            temporal_spread_init=raw["temporal_spread_init"],
            mu_t_margin_sigmas=float(raw["mu_t_margin_sigmas"]),
            prefilter_threshold=raw["prefilter_threshold"],
            cov_eps=raw["cov_eps"],
            appearance_model=AppearanceType(raw["appearance_model"]),
            n_temporal_freqs=raw["n_temporal_freqs"],
        )
