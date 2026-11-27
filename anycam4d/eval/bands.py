"""Incidence-angle bands for native-space evaluation.

A region scalar is weighted by the pixel grid; binning by incidence angle is comparable across grids
and lenses. Bands are pooled (summed squared error over all scored pixels / sample count) so narrow
bands are not dominated by per-image noise; the region scalars stay means over views.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor


#: Band edges in incidence degrees: 10 deg to 70, 5 deg to the quarter turn, one bin past it.
DEFAULT_BAND_EDGES_DEG: tuple[float, ...] = (
    0.0,
    10.0,
    20.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    180.0,
)

POOLING_NOTE = (
    "pooled: summed squared error over every scored pixel of every image, divided by the total "
    "sample count -- not a mean of per-image PSNRs (the region scalars are a mean over views)"
)


def validate_band_edges(edges: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Check edges are strictly increasing and span ``[0, 180]`` degrees, so bands partition every angle.

    Raises:
        ValueError: If not.
    """
    e = tuple(float(x) for x in edges)
    if len(e) < 2:
        raise ValueError(f"band edges need at least two entries to define one band, got {e}")
    if any(b <= a for a, b in itertools.pairwise(e)):
        raise ValueError(f"band edges must be strictly increasing, got {e}")
    if e[0] != 0.0 or e[-1] < 180.0:
        raise ValueError(f"band edges must span [0, 180] degrees so no valid pixel is dropped, got {e}")
    return e


def band_labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    """``"lo-hi"`` label per band, in degrees, with trailing zeros trimmed (``"70-75"``)."""
    return tuple(f"{lo:g}-{hi:g}" for lo, hi in itertools.pairwise(edges))


def incidence_degrees(rays: Float[Tensor, "H W 3"]) -> Float[Tensor, "H W"]:
    """Per-pixel incidence angle ``atan2(hypot(x, y), z)`` in degrees; NaN rays stay NaN."""
    return torch.rad2deg(torch.atan2(torch.hypot(rays[..., 0], rays[..., 1]), rays[..., 2]))


def band_masks(
    rays: Float[Tensor, "H W 3"],
    valid: Bool[Tensor, "H W"],
    edges: tuple[float, ...],
) -> tuple[Bool[Tensor, "H W"], ...]:
    """Per-band masks ``[lo, hi)`` (last band closed), each intersected with ``valid``: an exact partition.

    Args:
        rays: Unit-sphere rays ``[H, W, 3]``.
        valid: Validity mask ``[H, W]``.
        edges: Validated band edges in degrees.

    Returns:
        One ``[H, W]`` mask per band.
    """
    inc = incidence_degrees(rays).to(valid.device)
    last = len(edges) - 2
    masks = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        upper = inc <= hi if i == last else inc < hi
        masks.append((inc >= lo) & upper & valid)
    return tuple(masks)


class BandAccumulator:
    """Pooled squared error and SSIM per band over an eval set.

    Args:
        edges: Band edges in degrees.
    """

    def __init__(self, edges: tuple[float, ...] | list[float] = DEFAULT_BAND_EDGES_DEG) -> None:
        """Validate the edges and zero every band's accumulator."""
        self.edges = validate_band_edges(edges)
        self.labels = band_labels(self.edges)
        n = len(self.labels)
        self._squared_error = [0.0] * n
        self._ssim = [0.0] * n
        self._pixels = [0] * n
        self._samples = [0] * n

    def add(
        self,
        render: Float[Tensor, "H W 3"],
        gt: Float[Tensor, "H W 3"],
        ssim_map: Float[Tensor, "1 C H W"],
        masks: tuple[Bool[Tensor, "H W"], ...],
        semantic: Bool[Tensor, "H W"],
    ) -> None:
        """Accumulate one image; ``semantic`` is ANDed with each band mask, as the region metrics do."""
        squared_error = ((render - gt) ** 2).sum(-1)
        ssim_sum = ssim_map.squeeze(0).sum(0)
        channels = render.shape[-1]
        for i, mask in enumerate(masks):
            scored = mask & semantic
            n = int(scored.sum().item())
            if n == 0:
                continue
            self._squared_error[i] += float((squared_error * scored).sum().item())
            self._ssim[i] += float((ssim_sum * scored).sum().item())
            self._pixels[i] += n
            self._samples[i] += n * channels

    def table(self) -> dict[str, dict[str, float | int | None]]:
        """Per-band ``psnr``, ``ssim`` and ``px``; an empty band reports ``None``, not 0."""
        out: dict[str, dict[str, float | int | None]] = {}
        for i, label in enumerate(self.labels):
            if self._pixels[i] == 0:
                out[label] = {"psnr": None, "ssim": None, "px": 0}
                continue
            mse = self._squared_error[i] / self._samples[i]
            psnr = float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)
            out[label] = {"psnr": psnr, "ssim": self._ssim[i] / self._samples[i], "px": self._pixels[i]}
        return out

    def to_dict(self) -> dict:
        """Payload with the edges (tables cut at different edges are not comparable), pooling note and table."""
        return {"edges_deg": list(self.edges), "aggregation": POOLING_NOTE, "bands": self.table()}

    def format(self) -> str:
        """Human-readable ``lo-hi deg: PSNR (SSIM) over N px`` lines, one per non-empty band."""
        lines = []
        for label, values in self.table().items():
            if values["px"] == 0:
                continue
            lines.append(f"{label:>10} deg: {values['psnr']:6.2f} ({values['ssim']:.4f}) over {values['px']} px")
        return "\n".join(lines) if lines else "(no band carried a scored pixel)"
