"""Published-4DGS densification: the spatial trigger with the coupled full-4D split (paper baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from anycam4d.strategy._ops import split_full_4d
from anycam4d.strategy.default import DefaultSpatialStrategy


if TYPE_CHECKING:
    from jaxtyping import Bool
    from torch import Tensor
    from torch import nn

    from anycam4d.trainer.optimizers.group import OptimizerGroupProto


@dataclass
class OGFourDGSStrategy(DefaultSpatialStrategy):
    """The published 4DGS densifier, kept as the baseline.

    Candidates are selected exactly as in ``DefaultSpatialStrategy`` (spatial trigger, clone vs split on the
    spatial scale), but the split uses the coupled full-4D kernel of the published ``densify_and_split``: the
    offset is drawn through the 4D rotation and all four log-scales shrink, so a purely spatial trigger also
    moves the children's ``mu_t`` and temporal scale. The published code accepts a temporal gradient in
    ``densify_and_clone``/``densify_and_split`` but never reads it, so neither does this strategy.

    Fidelity to the published code is pinned by ``tests/test_og4dgs_port_fidelity.py``:

    * The published candidate signal is the screen-space NDC gradient (``strategy.spatial_trigger=screenspace``,
      EWA only); the shipped recipe reads the backend-independent paramspace trigger instead.
    * The published prune arm ``max_radii2D > 20`` has no counterpart: ``densification_postfix`` zeroes
      ``max_radii2D`` before the prune reads it, so that arm never fires in the published code.
    """

    _split_event: ClassVar[str] = "split_full_4d"

    def _spatial_split(
        self,
        params: dict[str, nn.Parameter],
        optimizers: OptimizerGroupProto,
        state: dict[str, Any],
        mask: Bool[Tensor, "N"],
    ) -> None:
        """Split the masked Gaussians with the coupled full-4D kernel (mutates ``mu_t``/``scale_t``)."""
        split_full_4d(
            params,
            optimizers,
            state,
            mask,
            sigma_t_floor=self.sigma_t_floor,
        )
