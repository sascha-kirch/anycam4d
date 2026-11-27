"""Backend-agnostic whole-frame metrics: per-image scores averaged over cameras per timestamp, then over time."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.valid_mask import ValidMaskCache
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import LpipsNet
from anycam4d.eval.color_correct import color_correct
from anycam4d.eval.metrics import literature_dssim
from anycam4d.eval.metrics import masked_psnr
from anycam4d.eval.metrics import masked_ssim
from anycam4d.eval.result import EvalResult


if TYPE_CHECKING:
    from jaxtyping import Bool
    from jaxtyping import Float
    from jaxtyping import Int
    from torch import Tensor

    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.pipeline import RenderPipeline


def _pstdev(values: list[float]) -> float:
    """Population std; 0.0 for fewer than two points or any non-finite value."""
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        return 0.0
    return statistics.pstdev(values)


def render_frame_batch(
    model: FourDGaussians,
    batch: dict,
    pipeline: RenderPipeline,
    sh_degree: int,
    device: torch.device,
    valid_mask: Int[Tensor, "C H W_words"] | None = None,
) -> Float[Tensor, "C H W 3"]:
    """Render one frame's cameras; shared by the evaluators.

    One pass when the cameras share a timestamp, else one render per camera at its own tau.
    """
    viewmats = batch["viewmats"].to(device)
    Ks = batch["Ks"].to(device)
    dist = batch["distortion_params"].to(device)
    images = batch["images"]
    camera = BatchedCamera(
        viewmats=viewmats,
        Ks=Ks,
        model=batch["camera_model"],
        width=images.shape[2],
        height=images.shape[1],
        distortion=dist,
        valid_mask=valid_mask,
    )
    taus = [float(t) for t in batch["timestamps"]]
    if max(taus) - min(taus) <= 1e-8:
        return pipeline.render(model, camera, taus[0], sh_degree).colors
    return torch.cat([pipeline.render(model, camera.select(i), tau, sh_degree).colors for i, tau in enumerate(taus)])


def _aggregate(
    per_ts: dict[str, list[float]],
    *,
    num_timesteps: int,
    num_cameras: int,
    num_images: int,
    per_cam: dict[str, dict[int, list[float]]] | None = None,
    timestamps: list[float] | None = None,
    frame_ids: list[int] | None = None,
) -> EvalResult:
    """Reduce per-timestamp values to mean and population std (non-finite values give std 0).

    ``per_cam`` is reduced to one value per manifest camera id, sorted by id.
    """
    mean = {k: statistics.fmean(v) for k, v in per_ts.items()}
    std = {k: _pstdev(v) for k, v in per_ts.items()}
    per_cam = per_cam or {}
    camera_ids = sorted({cam for by_cam in per_cam.values() for cam in by_cam})
    per_camera = {
        k: [statistics.fmean(by_cam[cam]) for cam in camera_ids if cam in by_cam] for k, by_cam in per_cam.items()
    }
    return EvalResult(
        mean=mean,
        std=std,
        per_timestamp={k: list(v) for k, v in per_ts.items()},
        per_camera=per_camera,
        num_timesteps=num_timesteps,
        num_cameras=num_cameras,
        num_images=num_images,
        camera_ids=camera_ids,
        timestamps=list(timestamps or []),
        frame_ids=list(frame_ids or []),
    )


class Evaluator:
    """Scores a model and pipeline over a dataset with PSNR, SSIM and LPIPS (``cc_*`` optional)."""

    def __init__(
        self,
        device: torch.device,
        *,
        lpips_net: str | LpipsNet = "alex",
        compute_cc: bool = False,
    ) -> None:
        """Build PSNR/SSIM/LPIPS metric modules on ``device``."""
        net = LpipsNet(lpips_net)  # raises ValueError on an unsupported net string
        self.device = device
        self.compute_cc = compute_cc
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net.value, normalize=True).to(device)

    def _ssim(self, p: Float[Tensor, "1 3 H W"], t: Float[Tensor, "1 3 H W"]) -> tuple[Tensor, Tensor]:
        """Whole-frame SSIM and its map via the functional API; the stateful module accumulates every map."""
        scalar, full = structural_similarity_index_measure(p, t, data_range=1.0, return_full_image=True)
        return scalar, full

    def _lpips_score(self, p: Float[Tensor, "1 3 H W"], t: Float[Tensor, "1 3 H W"]) -> float:
        """One pair's LPIPS, resetting the module's growing score state."""
        value = self.lpips(p, t).item()
        self.lpips.reset()
        return value

    def _score_pair(
        self,
        render: Float[Tensor, "H W 3"],
        gt: Float[Tensor, "H W 3"],
        validity: Bool[Tensor, "H W"] | None,
    ) -> dict[str, float]:
        """PSNR, SSIM, LPIPS and DSSIM₁/₂ over observed pixels; LPIPS zeroes both sides outside ``validity``."""
        p = render.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
        t = gt.permute(2, 0, 1).unsqueeze(0)
        ssim_scalar, ssim_map = self._ssim(p, t)
        dssim = literature_dssim(p, t, validity)
        if validity is None:
            return {
                "psnr": self.psnr(p, t).item(),
                "ssim": ssim_scalar.item(),
                "lpips": self._lpips_score(p, t),
            } | dssim
        w = validity.to(p.dtype)
        return {
            "psnr": masked_psnr(render.clamp(0, 1), gt, validity),
            "ssim": masked_ssim(ssim_map, validity),
            "lpips": self._lpips_score(p * w, t * w),
        } | dssim

    def _image_metrics(
        self,
        render: Float[Tensor, "H W 3"],
        gt: Float[Tensor, "H W 3"],
        validity: Bool[Tensor, "H W"] | None = None,
    ) -> dict[str, float]:
        """PSNR/SSIM/LPIPS (and optional ``cc_*``) for one ``[H, W, 3]`` render/GT pair."""
        vals = self._score_pair(render, gt, validity)
        if self.compute_cc:
            cc = self._score_pair(color_correct(render, gt), gt, validity)
            vals.update({f"cc_{k}": v for k, v in cc.items()})
        return vals

    def _validity_for(
        self,
        dataset: MultiViewSequenceDataset,
        camera_id: int,
        cache: dict[int, Bool[Tensor, "H W"]],
    ) -> Bool[Tensor, "H W"] | None:
        """One camera's observation mask on the metric device, memoized (it is static)."""
        mask = cache.get(camera_id)
        if mask is None:
            host = dataset.validity_for(camera_id)
            if host is None:
                return None
            mask = host.to(self.device)
            cache[camera_id] = mask
        return mask

    def _render_batch(
        self,
        model: FourDGaussians,
        batch: dict,
        pipeline: RenderPipeline,
        sh_degree: int,
        valid_mask: Int[Tensor, "C H W_words"] | None = None,
    ) -> Float[Tensor, "C H W 3"]:
        """Render one frame's cameras (see ``render_frame_batch``)."""
        return render_frame_batch(model, batch, pipeline, sh_degree, self.device, valid_mask)

    @torch.no_grad()
    def evaluate(
        self,
        model: FourDGaussians,
        pipeline: RenderPipeline,
        dataset: MultiViewSequenceDataset,
        *,
        sh_degree: int | None = None,
        apply_mask: bool = True,
    ) -> EvalResult:
        """Score every frame-aligned batch, keeping per-camera scores for both marginals."""
        sh = model.sh_degree if sh_degree is None else sh_degree
        # A per-call loader: the dataset is an argument, and num_workers=0 makes it free.
        loader = DataLoader(
            dataset,
            batch_sampler=FrameAlignedSampler(dataset, shuffle=False),
            collate_fn=anycam4d_collate,
            num_workers=0,
            # Its own RNG. A DataLoader iterator otherwise draws its base seed from the global CPU
            # stream even at num_workers=0, which shifts the training data order once per eval.
            # Nothing consumes this seed; see tests/test_eval_determinism.py.
            generator=torch.Generator().manual_seed(0),
        )
        per_ts: dict[str, list[float]] = defaultdict(list)
        per_cam: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        timestamps: list[float] = []
        frame_ids: list[int] = []
        num_images = 0
        valid_masks = ValidMaskCache(dataset, self.device)
        validity: dict[int, Bool[Tensor, "H W"]] = {}
        for batch in tqdm(loader, desc="Evaluating", unit="frame", leave=False, dynamic_ncols=True, colour="yellow"):
            images = batch["images"].to(self.device)  # [C, H, W, 3]
            packed = valid_masks.for_batch(batch["camera_ids"], images.shape[2], images.shape[1])
            renders = self._render_batch(model, batch, pipeline, sh, packed)  # [C, H, W, 3]
            masks = batch["masks"].to(self.device) if batch["masks"] is not None else None
            cam_vals: dict[str, list[float]] = defaultdict(list)
            for c in range(renders.shape[0]):
                render, gt = renders[c], images[c]
                if apply_mask and masks is not None:
                    render, gt = render * masks[c], gt * masks[c]
                cam_id = int(batch["camera_ids"][c])
                # Manifest id, so a logged camera can be looked up in the dataset.
                raw_id = dataset.raw_camera_id(cam_id)
                logged_id = cam_id if raw_id is None else raw_id
                v_c = None if dataset.fully_observed else self._validity_for(dataset, cam_id, validity)
                for k, v in self._image_metrics(render, gt, v_c).items():
                    cam_vals[k].append(v)
                    per_cam[k][logged_id].append(v)
                num_images += 1
            for k, v in cam_vals.items():
                per_ts[k].append(sum(v) / len(v))
            timestamps.append(sum(float(t) for t in batch["timestamps"]) / len(batch["timestamps"]))
            frame_ids.append(dataset.frame_id_for(batch["image_ids"][0]))
        return _aggregate(
            per_ts,
            num_timesteps=dataset.num_timesteps,
            num_cameras=dataset.cameras_per_timestep,
            num_images=num_images,
            per_cam=per_cam,
            timestamps=timestamps,
            frame_ids=frame_ids,
        )
