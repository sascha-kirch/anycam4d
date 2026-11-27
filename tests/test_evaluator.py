"""Tests for the Evaluator core and EvalResult aggregation."""

from __future__ import annotations

import json
import math
import shutil
import statistics

import pytest
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from anycam4d.config.rendering import RenderingConfig
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import CameraModelType
from anycam4d.eval.evaluator import Evaluator
from anycam4d.eval.evaluator import _aggregate
from anycam4d.eval.evaluator import render_frame_batch
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="render path requires CUDA")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pipeline() -> RenderPipeline:
    backend = BackendFactory.create(RenderingConfig())
    return RenderPipeline(backend=backend)


def _eval_dataset(tmp_dataset) -> MultiViewSequenceDataset:
    # Train split = 2 frames x 3 cameras (uniform); good for per-timestamp aggregation.
    return MultiViewSequenceDataset(tmp_dataset, split="train")


@requires_cuda
def test_evaluate_returns_all_metrics(tmp_dataset, random_model) -> None:
    ev = Evaluator(_device(), lpips_net="alex")
    ds = _eval_dataset(tmp_dataset)
    result = ev.evaluate(random_model.eval(), _pipeline(), ds)
    for key in ("psnr", "ssim", "lpips"):
        assert key in result.mean
        assert key in result.std
    assert result.num_timesteps == 2
    assert result.num_cameras == 3
    assert result.num_images == 6
    assert "cc_psnr" not in result.mean


@requires_cuda
def test_cc_metrics_gated(tmp_dataset, random_model, monkeypatch) -> None:
    # color_correct needs a non-degenerate render; feed GT so the affine fit is well-posed.
    monkeypatch.setattr(
        Evaluator,
        "_render_batch",
        lambda self, model, batch, pipeline, sh, valid_mask=None: batch["images"].to(self.device),
    )
    ds = _eval_dataset(tmp_dataset)
    pipe = _pipeline()
    off = Evaluator(_device(), compute_cc=False).evaluate(random_model.eval(), pipe, ds, apply_mask=False)
    assert "cc_psnr" not in off.mean
    on = Evaluator(_device(), compute_cc=True).evaluate(random_model.eval(), pipe, ds, apply_mask=False)
    for key in ("cc_psnr", "cc_ssim", "cc_lpips"):
        assert key in on.mean
        assert key in on.std


def test_std_over_timestamps() -> None:
    per_ts = {"psnr": [28.0, 30.0, 32.0]}
    result = _aggregate(per_ts, num_timesteps=3, num_cameras=1, num_images=3)
    assert result.mean["psnr"] == pytest.approx(statistics.fmean(per_ts["psnr"]))
    assert result.std["psnr"] == pytest.approx(statistics.pstdev(per_ts["psnr"]))


def test_lpips_net_invalid_raises() -> None:
    with pytest.raises(ValueError):
        Evaluator(_device(), lpips_net="not_a_net")


def _half_invalid_pair(height: int = 64, width: int = 64):
    """Render/GT pair differing only on the observed left half; the unobserved half is shared background."""
    torch.manual_seed(0)
    gt = torch.rand(height, width, 3)
    render = gt.clone()
    render[:, : width // 2] = torch.rand(height, width // 2, 3)
    gt[:, width // 2 :] = 0.0
    render[:, width // 2 :] = 0.0
    validity = torch.zeros(height, width, dtype=torch.bool)
    validity[:, : width // 2] = True
    return render, gt, validity


def test_masked_metrics_equal_the_valid_region_reference() -> None:
    """Masked PSNR on a half-invalid frame equals PSNR computed on the observed crop alone."""
    device = _device()
    ev = Evaluator(device, lpips_net="alex")
    render, gt, validity = _half_invalid_pair()
    render, gt, validity = render.to(device), gt.to(device), validity.to(device)

    masked = ev._image_metrics(render, gt, validity)
    crop = ev._image_metrics(render[:, :32], gt[:, :32], None)

    assert masked["psnr"] == pytest.approx(crop["psnr"], rel=1e-5)


def test_dense_metrics_are_inflated_by_the_valid_fraction() -> None:
    """Dense PSNR equals masked PSNR plus exactly ``-10 log10(f)``."""
    device = _device()
    ev = Evaluator(device, lpips_net="alex")
    render, gt, validity = _half_invalid_pair()
    render, gt, validity = render.to(device), gt.to(device), validity.to(device)
    valid_fraction = validity.float().mean().item()

    dense = ev._image_metrics(render, gt, None)
    masked = ev._image_metrics(render, gt, validity)

    inflation = -10.0 * math.log10(valid_fraction)
    assert dense["psnr"] - masked["psnr"] == pytest.approx(inflation, abs=1e-4)
    assert dense["ssim"] > masked["ssim"], "zeroed regions score SSIM 1 and drag the dense mean up"


def test_validity_none_is_the_plain_torchmetrics_path() -> None:
    """``validity=None`` reproduces the unmasked torchmetrics scalars exactly."""
    device = _device()
    ev = Evaluator(device, lpips_net="alex")
    render, gt, _ = _half_invalid_pair()
    render, gt = render.to(device), gt.to(device)

    p = render.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
    t = gt.permute(2, 0, 1).unsqueeze(0)
    vals = ev._image_metrics(render, gt, None)

    # Reference SSIM from the stateful torchmetrics module the Evaluator avoids.
    ssim_mod = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True).to(device)

    assert vals["psnr"] == ev.psnr(p, t).item()
    assert vals["ssim"] == ssim_mod(p, t)[0].item()
    assert vals["lpips"] == ev.lpips(p, t).item()


def test_lpips_zeroing_is_symmetric() -> None:
    """Masked LPIPS is the full-frame LPIPS of the pair with both sides zeroed on ``~v``."""
    device = _device()
    ev = Evaluator(device, lpips_net="alex")
    render, gt, validity = _half_invalid_pair()
    render, gt, validity = render.to(device), gt.to(device), validity.to(device)

    w = validity.unsqueeze(-1).float()
    reference = ev._image_metrics(render * w, gt * w, None)["lpips"]

    assert ev._image_metrics(render, gt, validity)["lpips"] == pytest.approx(reference, abs=1e-6)


def test_scoring_holds_no_per_image_metric_state() -> None:
    """Scoring many pairs keeps no metric state that grows with the images seen."""
    device = _device()
    ev = Evaluator(device, lpips_net="alex")
    render, gt, validity = _half_invalid_pair()
    render, gt, validity = render.to(device), gt.to(device), validity.to(device)

    def state_numel() -> int:
        total = 0
        for metric in (ev.psnr, ev.lpips):
            for name in metric._defaults:
                value = getattr(metric, name)
                items = value if isinstance(value, list) else [value]
                total += sum(t.numel() for t in items)
        return total

    for _ in range(3):
        ev._image_metrics(render, gt, validity)
    after_three = state_numel()
    for _ in range(20):
        ev._image_metrics(render, gt, validity)

    assert state_numel() == after_three


@requires_cuda
def test_camera_marginal_is_keyed_by_the_manifest_id(tmp_dataset, random_model, tmp_path) -> None:
    """The camera marginal is keyed by manifest camera ids, tested on sparse ids."""
    root = tmp_path / "sparse_ids"
    shutil.copytree(tmp_dataset, root)
    manifest_ids = [6, 22, 38]
    remap = dict(zip(range(3), manifest_ids, strict=True))
    for split in ("train", "eval"):
        path = root / f"{split}_dataset.json"
        doc = json.loads(path.read_text())
        for cam in doc["cameras"]:
            cam["camera_id"] = remap[cam["camera_id"]]
        for cap in doc["captures"]:
            cap["camera_id"] = remap[cap["camera_id"]]
        path.write_text(json.dumps(doc))

    ds = MultiViewSequenceDataset(root, split="train")
    assert ds.camera_ids[:3] == [0, 1, 2], "the loader is expected to renumber; the test is moot otherwise"

    result = Evaluator(_device(), lpips_net="alex").evaluate(random_model.eval(), _pipeline(), ds)
    assert result.camera_ids == manifest_ids
    assert result.worst_camera("psnr")[0] in manifest_ids


def test_cameras_offset_in_time_are_each_rendered_at_their_own_timestamp() -> None:
    """One tau reaches the kernels, so a frame whose cameras disagree in time is rendered camera by camera."""

    class _TauPipeline:
        def render(self, model, camera, tau, sh_degree):  # noqa: ANN001, ANN202, ARG002
            colors = torch.full((camera.num_cameras, 2, 2, 3), float(tau))
            return type("Out", (), {"colors": colors})()

    c = 3
    batch = {
        "viewmats": torch.eye(4).expand(c, 4, 4).clone(),
        "Ks": torch.eye(3).expand(c, 3, 3).clone(),
        "distortion_params": torch.zeros(c, 12),
        "images": torch.zeros(c, 2, 2, 3),
        "camera_model": CameraModelType.PINHOLE,
        "timestamps": [0.1, 0.1, 0.25],
    }
    colors = render_frame_batch(None, batch, _TauPipeline(), 0, torch.device("cpu"))

    assert colors[:, 0, 0, 0].tolist() == pytest.approx([0.1, 0.1, 0.25])
