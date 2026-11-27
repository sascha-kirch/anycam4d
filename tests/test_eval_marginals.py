"""Both marginals of the eval ``camera x timestamp`` matrix.

A dead camera moves the mean but leaves the per-timestamp std untouched, so only the camera marginal shows it.
"""

from __future__ import annotations

import pytest

from anycam4d.eval.evaluator import _aggregate
from anycam4d.eval.result import EvalResult
from anycam4d.eval.result import lower_is_better
from anycam4d.eval.result import metric_spec
from anycam4d.trainer.diagnostics import log_eval_marginals


class _RecordingWriter:
    """Minimal SummaryWriter stand-in that records what a logger tried to write."""

    def __init__(self) -> None:
        self.scalars: dict[str, float] = {}
        self.histograms: list[str] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:  # noqa: ARG002
        self.scalars[tag] = value

    def add_histogram(self, tag: str, values: object, step: int) -> None:  # noqa: ARG002
        self.histograms.append(tag)


def _result_from_matrix(matrix: dict[int, list[float]], metric: str = "psnr") -> EvalResult:
    """Build an EvalResult from ``camera_id -> per-timestamp scores`` the way the evaluator does."""
    camera_ids = sorted(matrix)
    n_ts = len(next(iter(matrix.values())))
    per_ts = {metric: [sum(matrix[c][t] for c in camera_ids) / len(camera_ids) for t in range(n_ts)]}
    per_cam = {metric: {c: list(matrix[c]) for c in camera_ids}}
    return _aggregate(
        per_ts,
        num_timesteps=n_ts,
        num_cameras=len(camera_ids),
        num_images=n_ts * len(camera_ids),
        per_cam=per_cam,
        timestamps=[float(t) for t in range(n_ts)],
        frame_ids=[100 + t for t in range(n_ts)],
    )


def test_lower_is_better_only_inverts_the_perceptual_distances() -> None:
    """Direction is by metric family, and the colour-corrected variants follow their base."""
    assert lower_is_better("lpips")
    assert lower_is_better("cc_lpips")
    assert not lower_is_better("psnr")
    assert not lower_is_better("ssim")
    assert not lower_is_better("cc_psnr")


def test_camera_marginal_is_keyed_by_id_not_encounter_order() -> None:
    """Cameras are reduced per id and reported in sorted id order."""
    result = _result_from_matrix({7: [30.0, 30.0], 2: [20.0, 20.0], 5: [25.0, 25.0]})
    assert result.camera_ids == [2, 5, 7]
    assert result.per_camera["psnr"] == [20.0, 25.0, 30.0]


def test_a_dead_camera_is_invisible_in_the_timestamp_marginal() -> None:
    """Same mean, loss spread vs concentrated in one camera: identical timestamp curves, different camera marginals."""
    spread = _result_from_matrix({0: [29.0, 29.0], 1: [29.0, 29.0], 2: [29.0, 29.0]})
    concentrated = _result_from_matrix({0: [33.0, 33.0], 1: [21.0, 21.0], 2: [33.0, 33.0]})

    assert spread.mean["psnr"] == concentrated.mean["psnr"]
    assert spread.std["psnr"] == concentrated.std["psnr"]
    assert spread.per_timestamp["psnr"] == concentrated.per_timestamp["psnr"]

    # The camera marginal separates them, and names the offender.
    assert spread.worst_camera("psnr") == (0, 29.0)
    assert concentrated.worst_camera("psnr") == (1, 21.0)


def test_worst_timestamp_reports_a_frame_id_and_respects_metric_direction() -> None:
    """The worst frame is named by its capture ``frame_id``, never by its position."""
    result = _aggregate(
        {"psnr": [30.0, 22.0, 31.0], "lpips": [0.10, 0.40, 0.09]},
        num_timesteps=3,
        num_cameras=1,
        num_images=3,
        frame_ids=[100, 101, 102],
    )
    assert result.worst_timestamp("psnr") == (101, 22.0)
    # Lower LPIPS is better, so the worst frame is the largest -- and it is the same frame.
    assert result.worst_timestamp("lpips") == (101, 0.40)


def test_trainer_logs_both_marginals_with_ids() -> None:
    """The logger writes worst-value and worst-*id* scalars per metric, plus the per-frame curve."""
    result = _result_from_matrix({0: [33.0, 33.0], 1: [21.0, 21.0], 2: [33.0, 33.0]})
    writer = _RecordingWriter()
    log_eval_marginals(writer, result, step=7)

    assert writer.scalars["eval/psnr_worst_frame"] == 29.0
    assert writer.scalars["eval/psnr_worst_frame_id"] == 100
    assert writer.scalars["eval/psnr_worst_camera"] == 21.0
    assert writer.scalars["eval/psnr_worst_camera_id"] == 1
    assert writer.histograms == ["eval/psnr_per_frame"]


def test_camera_marginal_absent_is_not_an_error() -> None:
    """Without per-camera scores only the timestamp marginal is logged."""
    result = _aggregate(
        {"psnr": [30.0, 22.0]},
        num_timesteps=2,
        num_cameras=1,
        num_images=2,
        frame_ids=[5, 6],
    )
    assert result.per_camera == {}
    writer = _RecordingWriter()
    log_eval_marginals(writer, result, step=0)

    assert writer.scalars["eval/psnr_worst_frame_id"] == 6
    assert not any("camera" in tag for tag in writer.scalars)


def test_a_single_camera_split_logs_no_camera_marginal() -> None:
    """With one eval camera the camera marginal equals the mean, so it is not logged."""
    result = _result_from_matrix({7: [28.0, 30.0]})
    assert result.per_camera["psnr"] == [29.0]
    assert result.mean["psnr"] == 29.0  # the marginal IS the mean, which is the whole problem

    writer = _RecordingWriter()
    log_eval_marginals(writer, result, step=3)

    assert "eval/psnr_worst_camera" not in writer.scalars
    assert "eval/psnr_worst_camera_id" not in writer.scalars
    # The timestamp marginal still carries information with one camera, so it is still written.
    assert writer.scalars["eval/psnr_worst_frame"] == 28.0


@pytest.mark.parametrize(
    ("metric", "lower", "decimals"),
    [
        ("psnr", False, 2),
        ("lpips", True, 4),
        ("cc_lpips", True, 4),
        ("full_psnr", False, 2),
        ("peripheral_ssim", False, 4),
    ],
)
def test_metric_spec_resolves_qualified_names(metric: str, lower: bool, decimals: int) -> None:
    """Region and cc qualifiers keep the base metric's direction; a wrong direction would report the
    worst value as the best one."""
    assert lower_is_better(metric) is lower
    assert metric_spec(metric).decimals == decimals


def test_metric_spec_rejects_unknown_metric() -> None:
    """An unregistered metric raises instead of defaulting to higher-is-better."""
    with pytest.raises(KeyError, match="unknown metric"):
        metric_spec("psnr_but_not_really")


def test_an_all_nonfinite_curve_does_not_abort_eval_logging(tmp_path) -> None:
    """A metric undefined on every frame (e.g. no observed pixel) must not crash the real writer mid-training."""
    from torch.utils.tensorboard import SummaryWriter

    result = _aggregate(
        {"lpips": [float("nan")], "psnr": [30.0, 31.0]},
        num_timesteps=2,
        num_cameras=1,
        num_images=2,
        frame_ids=[5, 6],
    )
    writer = SummaryWriter(str(tmp_path))
    log_eval_marginals(writer, result, step=0)
    writer.close()
