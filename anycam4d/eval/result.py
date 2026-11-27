"""EvalResult — aggregated evaluation metrics container."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class MetricSpec:
    """How one metric is read: its direction, and how many decimals it is printed with."""

    decimals: int
    lower_is_better: bool


# Keys are qualified by region and by colour correction (``full_psnr``, ``cc_lpips``); the qualifier
# never changes the direction or the scale, so the spec is keyed by the trailing base name.
_METRICS: dict[str, MetricSpec] = {
    "psnr": MetricSpec(decimals=2, lower_is_better=False),
    "ssim": MetricSpec(decimals=4, lower_is_better=False),
    "lpips": MetricSpec(decimals=4, lower_is_better=True),
    "dssim1": MetricSpec(decimals=4, lower_is_better=True),
    "dssim2": MetricSpec(decimals=4, lower_is_better=True),
}


def metric_spec(metric: str) -> MetricSpec:
    """Look up a metric's spec by its base name.

    Raises rather than guessing: a direction guessed wrong reports the worst value as the best one,
    and nothing downstream would notice.

    Raises:
        KeyError: The metric's base name is not registered in ``_METRICS``.
    """
    base = metric.rsplit("_", 1)[-1]
    if base not in _METRICS:
        raise KeyError(f"unknown metric {metric!r}; register it in anycam4d/eval/result.py:_METRICS")
    return _METRICS[base]


def _decimals(metric: str) -> int:
    """Decimal places for a metric (gsplat convention)."""
    return metric_spec(metric).decimals


def lower_is_better(metric: str) -> bool:
    """Whether smaller is better for ``metric``."""
    return metric_spec(metric).lower_is_better


@dataclass
class EvalResult:
    """Metrics aggregated over the camera x timestamp score matrix.

    Attributes:
        mean: Metric -> mean over timestamps; ``cc_*`` only with colour correction.
        std: Metric -> population std over timestamps.
        per_timestamp: Metric -> camera-averaged value per frame, in frame order.
        per_camera: Metric -> timestamp-averaged value per camera, aligned with ``camera_ids``;
            shows a single failing camera the timestamp marginal cannot.
        num_timesteps: Distinct timestamps evaluated.
        num_cameras: Cameras per timestamp.
        num_images: Images scored.
        timestamps: Normalized timestamp per ``per_timestamp`` entry.
        camera_ids: Camera id per ``per_camera`` entry.
        frame_ids: Capture ``frame_id`` per ``per_timestamp`` entry (frame identity, unlike timestamps).
    """

    mean: dict[str, float]
    std: dict[str, float]
    per_timestamp: dict[str, list[float]]
    num_timesteps: int
    num_cameras: int
    num_images: int
    per_camera: dict[str, list[float]] = field(default_factory=dict)
    camera_ids: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def worst_timestamp(self, metric: str) -> tuple[int, float]:
        """Direction-aware ``(frame_id, value)`` of the worst frame; raises on an unknown metric or empty curve."""
        values = self.per_timestamp[metric]
        # "Worst" inverts with the metric: the smallest PSNR, but the largest LPIPS.
        pick = max if lower_is_better(metric) else min
        idx = pick(range(len(values)), key=values.__getitem__)
        frame_id = self.frame_ids[idx] if idx < len(self.frame_ids) else idx
        return frame_id, values[idx]

    def worst_camera(self, metric: str) -> tuple[int, float]:
        """Direction-aware ``(camera_id, value)`` of the worst camera."""
        values = self.per_camera[metric]
        pick = max if lower_is_better(metric) else min
        idx = pick(range(len(values)), key=values.__getitem__)
        camera_id = self.camera_ids[idx] if idx < len(self.camera_ids) else idx
        return camera_id, values[idx]

    def format(self) -> str:
        """Human-readable ``METRIC: mean (std)`` lines, one per metric."""
        lines = []
        for key, m in self.mean.items():
            d = _decimals(key)
            lines.append(f"{key.upper()}: {m:.{d}f} ({self.std[key]:.{d}f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable dict of means, stds, both per-axis marginals, and counts."""
        return {
            "mean": self.mean,
            "std": self.std,
            "per_timestamp": self.per_timestamp,
            "per_camera": self.per_camera,
            "num_timesteps": self.num_timesteps,
            "num_cameras": self.num_cameras,
            "num_images": self.num_images,
            "camera_ids": self.camera_ids,
            "timestamps": self.timestamps,
            "frame_ids": self.frame_ids,
        }
