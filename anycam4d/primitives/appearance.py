"""Per-appearance-model facts: which parameter holds the temporal coefficients, and the CUDA id.

CUDA-free, so config and checkpoint code can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from anycam4d.enums import AppearanceType


if TYPE_CHECKING:
    from torch import Tensor


# Highest temporal order the multiplicative basis carries; published 4DGS fixes it at 2 (sh_degree_t) and never
# varies it, so it is a constant rather than a knob. Must mirror MAX_TEMPORAL_DEGREE in cuda/csrc/common.h.
MAX_TEMPORAL_DEGREE = 2

# Colour leaf the kernels instantiate. Must mirror ``enum class AppearanceKind`` in cuda/csrc/common.h.
APPEARANCE_CUDA_ID: dict[AppearanceType, int] = {
    AppearanceType.SH4D_ADDITIVE: 0,
    AppearanceType.SH4D_MULTIPLICATIVE: 1,
}

# Parameter holding each model's temporal coefficients.
APPEARANCE_PARAM: dict[AppearanceType, str] = {
    AppearanceType.SH4D_ADDITIVE: "sh_temporal",
    AppearanceType.SH4D_MULTIPLICATIVE: "sh_spacetime",
}


@dataclass(frozen=True)
class AppearanceInputs:
    """What the fused kernels need to evaluate colour, for one render at one active degree.

    Attributes:
        sh0: DC SH coefficient ``[N, 1, 3]``.
        shN: Higher SH coefficients ``[N, K_max - 1, 3]``; the kernels read only the active degree.
        coeffs: Temporal coefficients ``[N, T, 3]``, empty when the model has none.
        kind_id: ``APPEARANCE_CUDA_ID`` of the leaf to instantiate.
        n_terms: Active terms of that leaf; 0 evaluates spatial SH alone.
        stride: Floats per Gaussian in ``coeffs``.
    """

    sh0: Tensor
    shN: Tensor
    coeffs: Tensor
    kind_id: int
    n_terms: int
    stride: int


def active_degrees(step: int, interval: int, max_sh_degree: int) -> tuple[int, int]:
    """Spatial and temporal SH degree in effect at ``step``.

    Published 4DGS raises one degree per ``sh_increase_interval`` and takes the spatial degree to
    its maximum **before** the temporal degree starts (``gaussian_model.py:oneupSHdegree``).

    Returns:
        ``(sh_degree, sh_degree_t)``, each clamped to its maximum.
    """
    raised = step // interval if interval > 0 else max_sh_degree + MAX_TEMPORAL_DEGREE
    return min(max_sh_degree, raised), min(MAX_TEMPORAL_DEGREE, max(0, raised - max_sh_degree))
