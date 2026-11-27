"""FourDGaussians: the 4D Gaussian parameter container."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Self

import torch
from torch import nn

from anycam4d.enums import AppearanceType
from anycam4d.math.covariance import temporal_marginal_from_rot4d
from anycam4d.primitives._helpers import RGB2SH
from anycam4d.primitives._helpers import _knn_distances
from anycam4d.primitives.appearance import APPEARANCE_CUDA_ID
from anycam4d.primitives.appearance import APPEARANCE_PARAM
from anycam4d.primitives.appearance import MAX_TEMPORAL_DEGREE
from anycam4d.primitives.appearance import AppearanceInputs
from anycam4d.primitives.fourdgaussians.conditioning import temporal_marginal_std


if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class FourDGaussians(nn.Module):
    """4D Gaussians in a flat ``nn.ParameterDict`` (rot_4d parameterization); activations applied at use.

    Keys:
        means [N, 3], mu_t [N] (normalized time), quats_l / quats_r [N, 4] (wxyz, unnormalized),
        scales_4d_log [N, 4] (log sx, sy, sz, st), opacities [N] (logit), sh0 [N, 1, 3],
        shN [N, K-1, 3], sh_temporal [N, T, 3] (SH4D_ADDITIVE only, T = 2 * n_temporal_freqs).
    """

    def __init__(
        self,
        splats: nn.ParameterDict,
        *,
        sh_degree: int,
        appearance_model: str | AppearanceType = AppearanceType.SH4D_ADDITIVE,
        n_temporal_freqs: int = 4,
        cov_eps: float = 1e-6,
        prefilter_threshold: float = 0.05,
        mu_t_margin_sigmas: float = -1.0,
    ) -> None:
        """Wrap raw parameters.

        Args:
            splats: Raw Gaussian parameters.
            sh_degree: Maximum SH degree.
            appearance_model: ``SH`` or ``SH4D_ADDITIVE``.
            n_temporal_freqs: Temporal Fourier frequencies (SH4D_ADDITIVE only).
            cov_eps: Spatial PSD floor for Schur conditioning.
            prefilter_threshold: Temporal weight below which a Gaussian is zeroed.
            mu_t_margin_sigmas: ``clamp_temporal_mean`` margin in widths; ``-1`` derives it from the prefilter.
        """
        super().__init__()
        self.splats = splats
        self._sh_degree = sh_degree
        self._appearance_model = (
            AppearanceType(appearance_model) if isinstance(appearance_model, str) else appearance_model
        )
        self._n_temporal_freqs = n_temporal_freqs
        self.cov_eps = cov_eps
        self.prefilter_threshold = prefilter_threshold
        self.mu_t_margin_sigmas = mu_t_margin_sigmas

        # The kernel splits the multiplicative stride into MAX_TEMPORAL_DEGREE blocks, so a file with
        # any other row count would be read at the wrong offsets — wrong colours, no error.
        if self._appearance_model is AppearanceType.SH4D_MULTIPLICATIVE:
            expected = MAX_TEMPORAL_DEGREE * (sh_degree + 1) ** 2
            rows = splats["sh_spacetime"].shape[1]
            if rows != expected:
                raise ValueError(
                    f"sh_spacetime has {rows} rows; the multiplicative basis needs "
                    f"MAX_TEMPORAL_DEGREE * (sh_degree + 1)^2 = {expected} at sh_degree {sh_degree}."
                )

    @classmethod
    def from_points(
        cls,
        points: Float[Tensor, "N 3"],
        *,
        sh_degree: int = 3,
        init_opacity: float = 0.1,
        temporal_spread_init: float = 0.2,
        appearance_model: str | AppearanceType = AppearanceType.SH4D_ADDITIVE,
        n_temporal_freqs: int = 4,
        cov_eps: float = 1e-6,
        prefilter_threshold: float = 0.05,
        mu_t_margin_sigmas: float = -1.0,
        colors: Float[Tensor, "N 3"] | None = None,
    ) -> Self:
        """Initialize on CPU from a point cloud; scales from the mean distance to the three nearest neighbours."""
        appearance_enum = AppearanceType(appearance_model) if isinstance(appearance_model, str) else appearance_model

        points_cpu = points.detach().cpu().float()
        n = points_cpu.shape[0]
        n_sh = (sh_degree + 1) ** 2

        mean_dist = _knn_distances(points_cpu, k=4)[:, 1:].mean(dim=-1)

        sh0_init = RGB2SH(colors.cpu()).unsqueeze(1) if colors is not None else torch.zeros(n, 1, 3)

        scales_xyz_log = torch.log(mean_dist.clamp(min=1e-8)).unsqueeze(-1).expand(n, 3)
        scale_t_log = math.log(max(temporal_spread_init, 1e-8))
        scales_4d_log_init = torch.cat([scales_xyz_log, torch.full((n, 1), scale_t_log)], dim=-1)

        quats_l_init = torch.zeros(n, 4)
        quats_l_init[:, 0] = 1.0
        quats_l_init = quats_l_init + torch.randn(n, 4) * 1e-3
        quats_l_init = torch.nn.functional.normalize(quats_l_init, dim=-1)

        quats_r_init = torch.zeros(n, 4)
        quats_r_init[:, 0] = 1.0
        quats_r_init = quats_r_init + torch.randn(n, 4) * 1e-3
        quats_r_init = torch.nn.functional.normalize(quats_r_init, dim=-1)

        splats = nn.ParameterDict(
            {
                "means": nn.Parameter(points_cpu.clone()),
                "mu_t": nn.Parameter(torch.rand(n) - 0.5),
                "quats_l": nn.Parameter(quats_l_init),
                "quats_r": nn.Parameter(quats_r_init),
                "scales_4d_log": nn.Parameter(scales_4d_log_init),
                "opacities": nn.Parameter(torch.full((n,), torch.logit(torch.tensor(init_opacity)).item())),
                "sh0": nn.Parameter(sh0_init),
                "shN": nn.Parameter(torch.zeros(n, n_sh - 1, 3)),
            }
        )

        if appearance_enum == AppearanceType.SH4D_ADDITIVE:
            splats["sh_temporal"] = nn.Parameter(torch.zeros(n, 2 * n_temporal_freqs, 3))
        elif appearance_enum == AppearanceType.SH4D_MULTIPLICATIVE:
            splats["sh_spacetime"] = nn.Parameter(torch.zeros(n, MAX_TEMPORAL_DEGREE * n_sh, 3))

        return cls(
            splats,
            sh_degree=sh_degree,
            appearance_model=appearance_enum,
            n_temporal_freqs=n_temporal_freqs,
            cov_eps=cov_eps,
            prefilter_threshold=prefilter_threshold,
            mu_t_margin_sigmas=mu_t_margin_sigmas,
        )

    @classmethod
    def from_random(
        cls,
        n_points: int,
        sh_degree: int = 3,
        appearance_model: str | AppearanceType = AppearanceType.SH4D_ADDITIVE,
        n_temporal_freqs: int = 4,
        mu_t_margin_sigmas: float = -1.0,
    ) -> Self:
        """Create a FourDGaussians with random parameters (useful for testing)."""
        appearance_enum = AppearanceType(appearance_model) if isinstance(appearance_model, str) else appearance_model
        n_sh = (sh_degree + 1) ** 2

        quats_l_init = torch.zeros(n_points, 4)
        quats_l_init[:, 0] = 1.0
        quats_r_init = torch.zeros(n_points, 4)
        quats_r_init[:, 0] = 1.0
        splats = nn.ParameterDict(
            {
                "means": nn.Parameter(torch.randn(n_points, 3)),
                # Normalized time spans [-0.5, 0.5]; from_points uses the same range.
                "mu_t": nn.Parameter(torch.rand(n_points) - 0.5),
                "quats_l": nn.Parameter(quats_l_init),
                "quats_r": nn.Parameter(quats_r_init),
                "scales_4d_log": nn.Parameter(torch.full((n_points, 4), -3.0)),
                "opacities": nn.Parameter(torch.full((n_points,), torch.logit(torch.tensor(0.1)).item())),
                "sh0": nn.Parameter(torch.zeros(n_points, 1, 3)),
                "shN": nn.Parameter(torch.zeros(n_points, n_sh - 1, 3)),
            }
        )

        if appearance_enum == AppearanceType.SH4D_ADDITIVE:
            splats["sh_temporal"] = nn.Parameter(torch.zeros(n_points, 2 * n_temporal_freqs, 3))
        elif appearance_enum == AppearanceType.SH4D_MULTIPLICATIVE:
            splats["sh_spacetime"] = nn.Parameter(torch.zeros(n_points, MAX_TEMPORAL_DEGREE * n_sh, 3))

        return cls(
            splats,
            sh_degree=sh_degree,
            appearance_model=appearance_enum,
            n_temporal_freqs=n_temporal_freqs,
            mu_t_margin_sigmas=mu_t_margin_sigmas,
        )

    def __len__(self) -> int:
        """Number of 4D Gaussians."""
        return self.splats["means"].shape[0]

    @property
    def sh_degree(self) -> int:
        """Maximum spherical harmonics degree for color representation."""
        return self._sh_degree

    @property
    def appearance_model(self) -> AppearanceType:
        """Appearance model type."""
        return self._appearance_model

    @property
    def n_temporal_freqs(self) -> int:
        """Number of temporal Fourier frequencies."""
        return self._n_temporal_freqs

    def appearance_inputs(self, sh_degree_t: int = MAX_TEMPORAL_DEGREE) -> AppearanceInputs:
        """Fused-kernel appearance inputs at the active temporal degree; the spatial degree goes to the kernels."""
        kind_id = APPEARANCE_CUDA_ID[self.appearance_model]
        sh0, shN = self.splats["sh0"], self.splats["shN"]
        coeffs = self.splats[APPEARANCE_PARAM[self.appearance_model]].contiguous()

        if self.appearance_model is AppearanceType.SH4D_MULTIPLICATIVE:
            n_terms = min(sh_degree_t, MAX_TEMPORAL_DEGREE)
            return AppearanceInputs(sh0, shN, coeffs, kind_id, n_terms, coeffs.shape[1] * 3)

        rows = coeffs.shape[1]
        return AppearanceInputs(sh0, shN, coeffs, kind_id, rows // 2, rows * 3)

    def intrinsic_spatial_scales(self) -> Float[Tensor, "N 3"]:
        """Spatial scale parameters ``exp(scales_4d_log[:, :3])`` ``[N, 3]``; not the rendered extent."""
        return torch.exp(self.splats["scales_4d_log"][:, :3])

    def marginal_temporal_variance(self) -> Float[Tensor, "N"]:
        """Marginal temporal variance ``cov_4d[3, 3]`` ``[N]``, what the temporal gate divides by."""
        return temporal_marginal_from_rot4d(
            self.splats["quats_l"], self.splats["quats_r"], self.splats["scales_4d_log"]
        )

    def marginal_temporal_std(self) -> Float[Tensor, "N"]:
        """Marginal temporal width ``std_t_marg`` [N], with the renderer's epsilon included."""
        return temporal_marginal_std(self.marginal_temporal_variance())

    @torch.no_grad()
    def clamp_temporal_mean(self, margin_sigmas: float, prefilter_threshold: float) -> None:
        """Clamp ``mu_t`` to where the primitive can still pass the prefilter at some sequence timestamp, in place.

            |mu_t| <= 0.5 + k * std_t_marg,   k = sqrt(-2 ln(prefilter_threshold))

        Without it ``mu_t`` random-walks out of the sequence. The width is the guarded marginal the gate uses.

        Args:
            margin_sigmas: ``k``; ``-1`` derives it from ``prefilter_threshold``, ``0`` pins ``mu_t`` inside.
            prefilter_threshold: Temporal cutoff, used when ``margin_sigmas`` is ``-1``; non-positive makes
                this a no-op.
        """
        if margin_sigmas < 0.0:
            if prefilter_threshold <= 0.0 or prefilter_threshold >= 1.0:
                return
            margin_sigmas = math.sqrt(-2.0 * math.log(prefilter_threshold))
        limit = 0.5 + margin_sigmas * self.marginal_temporal_std()
        self.splats["mu_t"].data.clamp_(min=-limit, max=limit)

    def state_dict_with_meta(
        self,
        norm_scale: float,
        norm_center: "Float[Tensor, '3']",
        t_min: float,
        t_max: float,
    ) -> dict:
        """Full checkpoint state including scene normalization metadata."""
        return {
            "splats": {k: v.detach().cpu() for k, v in self.splats.items()},
            "sh_degree": self.sh_degree,
            "appearance_model": self.appearance_model.value,
            "n_temporal_freqs": self._n_temporal_freqs,
            "cov_eps": self.cov_eps,
            "prefilter_threshold": self.prefilter_threshold,
            "mu_t_margin_sigmas": self.mu_t_margin_sigmas,
            "norm_scale": norm_scale,
            "norm_center": norm_center.detach().cpu().tolist(),
            "t_min": t_min,
            "t_max": t_max,
        }

    @classmethod
    def from_checkpoint(cls, state_dict: dict, *, source: str = "checkpoint") -> Self:
        """Rebuild from ``checkpoint["model_state"]``; ``source`` names the file in errors."""
        required = {
            "splats",
            "sh_degree",
            "appearance_model",
            "n_temporal_freqs",
            "cov_eps",
            "prefilter_threshold",
            "mu_t_margin_sigmas",
        }
        missing = required - state_dict.keys()
        if missing:
            raise ValueError(f"{source} is not an anycam4d checkpoint (missing {sorted(missing)})")
        return cls(
            nn.ParameterDict({k: nn.Parameter(v) for k, v in state_dict["splats"].items()}),
            sh_degree=state_dict["sh_degree"],
            appearance_model=AppearanceType(state_dict["appearance_model"]),
            n_temporal_freqs=state_dict["n_temporal_freqs"],
            cov_eps=state_dict["cov_eps"],
            prefilter_threshold=state_dict["prefilter_threshold"],
            mu_t_margin_sigmas=state_dict["mu_t_margin_sigmas"],
        )

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"FourDGaussians(n_gaussians={len(self)}, n_params={n_params:,}, sh_degree={self.sh_degree})"
