"""Incidence-band and per-frame breakdown of the native-space eval."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import CameraModelType
from anycam4d.eval.bands import DEFAULT_BAND_EDGES_DEG
from anycam4d.eval.bands import BandAccumulator
from anycam4d.eval.bands import validate_band_edges
from anycam4d.eval.native_evaluator import NativeRegionEvaluator
from anycam4d.eval.native_evaluator import _native_camera_geometry
from anycam4d.eval.metrics import masked_psnr
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from tests.test_native_evaluator import _WIDE_FISHEYE_DIST
from tests.test_native_evaluator import _WIDE_FISHEYE_K
from tests.test_native_evaluator import _ut_pipeline


requires_cuda = pytest.mark.skipif(not HAS_CUDA_EXT, reason="render path runs on the CUDA kernels")

_CPU = torch.device("cpu")

# Integer edges, so ``band_{lo}_{hi}`` and ``"lo-hi"`` labels name the same interval.
_EDGES: tuple[float, ...] = (0.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 180.0)


def test_default_edges_span_the_whole_range() -> None:
    """The shipped default edges satisfy the partition requirement they are validated against."""
    assert validate_band_edges(DEFAULT_BAND_EDGES_DEG) == DEFAULT_BAND_EDGES_DEG


@pytest.mark.parametrize(
    "edges",
    [
        (10.0, 90.0, 180.0),  # does not start at 0: pixels near the axis would vanish
        (0.0, 90.0),  # stops at 90: a fisheye's own observed pixels past the quarter turn vanish
        (0.0, 90.0, 90.0, 180.0),  # not strictly increasing: an empty band, and an ambiguous label
        (0.0,),  # no band at all
    ],
)
def test_edges_that_would_drop_pixels_raise(edges: tuple[float, ...]) -> None:
    """Edges are rejected at construction, so "no pixel is dropped" is structural, not incidental."""
    with pytest.raises(ValueError):
        BandAccumulator(edges)


def test_bands_partition_the_valid_set() -> None:
    """Band masks partition ``valid`` of a wide fisheye whose corners are unobserved."""
    geom = _native_camera_geometry(
        _WIDE_FISHEYE_K, _WIDE_FISHEYE_DIST, CameraModelType.FISHEYE_OPENCV, 64, 48, _CPU, None, _EDGES
    )
    assert len(geom.bands) == len(_EDGES) - 1
    membership = torch.stack(geom.bands).int().sum(0)
    assert torch.equal(membership, geom.valid.int())
    assert bool((~geom.valid).any()), "fixture must have unobserved pixels for this to mean anything"


def test_an_empty_band_reports_absent() -> None:
    """A band no pixel landed in reports `None`, not a number: absent and zero are different claims."""
    accumulator = BandAccumulator((0.0, 90.0, 180.0))
    render = torch.zeros(4, 4, 3)
    gt = torch.full((4, 4, 3), 0.5)
    ssim_map = torch.zeros(1, 3, 4, 4)
    occupied = torch.ones(4, 4, dtype=torch.bool)
    empty = torch.zeros(4, 4, dtype=torch.bool)
    accumulator.add(render, gt, ssim_map, (occupied, empty), occupied)

    table = accumulator.table()
    assert table["90-180"] == {"psnr": None, "ssim": None, "px": 0}
    assert table["0-90"]["px"] == 16
    assert table["0-90"]["psnr"] == pytest.approx(10.0 * math.log10(1.0 / 0.25))


def test_pooled_band_equals_masked_psnr() -> None:
    """One band over the whole mask reproduces `masked_psnr`: the same metric, a different subset."""
    generator = torch.Generator().manual_seed(3)
    render = torch.rand(8, 6, 3, generator=generator)
    gt = torch.rand(8, 6, 3, generator=generator)
    mask = torch.rand(8, 6, generator=generator) > 0.3

    accumulator = BandAccumulator((0.0, 180.0))
    accumulator.add(render, gt, torch.zeros(1, 3, 8, 6), (mask,), torch.ones(8, 6, dtype=torch.bool))

    assert accumulator.table()["0-180"]["psnr"] == pytest.approx(masked_psnr(render, gt, mask), abs=1e-5)


@requires_cuda
def test_pinhole_bands_are_finite_or_absent(tmp_dataset: Path) -> None:
    """A pinhole's bands past its ~39 degree field are absent, never NaN."""
    device = torch.device("cuda")
    dataset = MultiViewSequenceDataset(tmp_dataset, split="train")
    model = FourDGaussians.from_random(n_points=200, sh_degree=1).to(device).eval()

    report = NativeRegionEvaluator(device, lpips_net="alex", band_edges=_EDGES).evaluate(model, _ut_pipeline(), dataset)

    table = report.bands.table()
    for label, values in table.items():
        if values["px"] == 0:
            assert values["psnr"] is None and values["ssim"] is None
            continue
        assert not math.isnan(values["psnr"]), f"{label}: NaN PSNR"
        assert not math.isnan(values["ssim"]), f"{label}: NaN SSIM"
    assert table["40-50"]["px"] == 0, "a 50 mm-equivalent pinhole cannot see past ~39 degrees"
    assert table["0-20"]["px"] > 0


@requires_cuda
def test_per_frame_bands_are_keyed_by_frame_id(tmp_dataset: Path) -> None:
    """Per-frame keys are capture frame ids, and per-frame tables sum to the pooled one."""
    device = torch.device("cuda")
    dataset = MultiViewSequenceDataset(tmp_dataset, split="train")
    model = FourDGaussians.from_random(n_points=200, sh_degree=1).to(device).eval()

    report = NativeRegionEvaluator(device, lpips_net="alex", band_edges=_EDGES, per_frame_bands=True).evaluate(
        model, _ut_pipeline(), dataset
    )

    assert sorted(report.per_frame_bands) == dataset.frame_ids
    assert report.result.frame_ids == dataset.frame_ids
    assert len(report.result.frame_ids) == len(report.result.timestamps)

    pooled = report.bands.table()
    for label in report.bands.labels:
        per_frame_px = sum(acc.table()[label]["px"] for acc in report.per_frame_bands.values())
        assert per_frame_px == pooled[label]["px"], f"{label}: per-frame pixels do not sum to pooled"

    payload = report.to_dict()
    assert payload["edges_deg"] == list(_EDGES)
    assert sorted(payload["per_frame"]) == sorted(str(f) for f in dataset.frame_ids)
