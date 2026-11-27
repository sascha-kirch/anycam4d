"""Native-space evaluation: full/central/peripheral PSNR/SSIM, full-frame LPIPS, incidence bands, per-frame curve.

Renders are already in each camera's native space. Central is what a reference pinhole of the same
camera images (the undistorted copy, so the target both a native and an undistorted model observed);
without a reference no central/peripheral split is scored. Geometry is cached per camera.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from anycam4d.camera.valid_mask import ValidMaskCache
from anycam4d.camera.validity import compute_validity
from anycam4d.data.samplers import FrameAlignedSampler
from anycam4d.data.samplers import anycam4d_collate
from anycam4d.enums import LpipsNet
from anycam4d.eval.bands import DEFAULT_BAND_EDGES_DEG
from anycam4d.eval.bands import BandAccumulator
from anycam4d.eval.bands import band_masks
from anycam4d.eval.evaluator import _aggregate
from anycam4d.eval.evaluator import render_frame_batch
from anycam4d.eval.metrics import masked_psnr
from anycam4d.eval.metrics import masked_ssim
from anycam4d.math.camera_rays import camera_rays
from anycam4d.math.region import PinholeField
from anycam4d.math.region import build_region_masks


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Bool
    from jaxtyping import Float
    from torch import Tensor

    from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
    from anycam4d.enums import CameraModelType
    from anycam4d.eval.result import EvalResult
    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.pipeline import RenderPipeline


@dataclass(frozen=True)
class CameraGeometry:
    """Cached per-camera region masks. Time-invariant: built once per camera, reused every timestep."""

    valid: Bool[Tensor, "H W"]
    central: Bool[Tensor, "H W"]
    peripheral: Bool[Tensor, "H W"]
    # One mask per incidence band, in edge order, each already intersected with ``valid``. An exact
    # partition of the valid set, so the bands account for every pixel the region scalars score.
    bands: tuple[Bool[Tensor, "H W"], ...]


def _native_camera_geometry(
    K: Float[Tensor, "3 3"],
    distortion: Float[Tensor, "P"],
    model: CameraModelType,
    width: int,
    height: int,
    device: torch.device,
    observed: Bool[Tensor, "h w"] | None = None,
    band_edges: tuple[float, ...] = DEFAULT_BAND_EDGES_DEG,
    field: PinholeField | None = None,
) -> CameraGeometry:
    """Valid, central, peripheral and band masks for one camera.

    ``valid`` is the lens footprint, intersected with pixels that have a ray (an undetermined angle would
    otherwise land in peripheral) and with the loader's ``observed`` mask (which covers reprojected
    pinholes whose border the source never saw).

    Args:
        K: Intrinsics.
        distortion: Canonical distortion.
        model: Camera model.
        width: Working width.
        height: Working height.
        device: Target device.
        observed: Loader-resolved validity, or ``None``.
        band_edges: Band edges in degrees.
        field: Reference pinhole of the central region; ``None`` scores no central/peripheral split.
    """
    rays, ray_valid = camera_rays(K, distortion, model, width, height)
    valid = compute_validity(model, K, distortion, width, height) & ray_valid  # CPU, like ``rays``
    if observed is not None:
        valid = valid & observed.to(valid.device, torch.bool)
    region = build_region_masks(rays, field, valid=valid)
    bands = band_masks(rays, region["valid"], band_edges)
    return CameraGeometry(
        valid=region["valid"].to(device),
        central=region["central"].to(device),
        peripheral=region["peripheral"].to(device),
        bands=tuple(b.to(device) for b in bands),
    )


@dataclass(frozen=True)
class NativeRegionReport:
    """One native-space eval pass.

    Attributes:
        result: ``{full,central,peripheral}_{psnr,ssim}`` and ``lpips``, means over views, per-frame curve by ``frame_id``.
        bands: Pooled per-band PSNR/SSIM.
        per_frame_bands: Band tables per frame; empty unless requested.
    """

    result: EvalResult
    bands: BandAccumulator
    per_frame_bands: dict[int, BandAccumulator]

    def to_dict(self) -> dict:
        """JSON-serialisable band payload: edges, pooling note, table, and the per-frame cross."""
        payload = self.bands.to_dict()
        if self.per_frame_bands:
            payload["per_frame"] = {str(fid): acc.table() for fid, acc in sorted(self.per_frame_bands.items())}
        return payload


def load_central_fields(data_dir: Path) -> dict[int, PinholeField]:
    """Read the reference pinholes of an undistorted dataset's held-out cameras, by rig camera id."""
    cameras = json.loads((data_dir / "eval_dataset.json").read_text())["cameras"]
    fields: dict[int, PinholeField] = {}
    for cam in cameras:
        if cam["model"] != "pinhole":
            raise ValueError(f"central reference {data_dir} camera {cam['camera_id']} is {cam['model']}, not pinhole")
        fx, fy, cx, cy = cam["params"][:4]
        fields[int(cam["camera_id"])] = PinholeField(fx, fy, cx, cy, float(cam["width"]), float(cam["height"]))
    return fields


class NativeRegionEvaluator:
    """Native-space region-split evaluator.

    Args:
        device: Device for rendering and metrics.
        lpips_net: LPIPS backbone.
        band_edges: Band edges in degrees, spanning ``[0, 180]``.
        per_frame_bands: Also keep a band table per frame.
        central_fields: Reference pinhole of the central region per rig camera id, or ``None``.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        lpips_net: str | LpipsNet = "alex",
        band_edges: tuple[float, ...] = DEFAULT_BAND_EDGES_DEG,
        per_frame_bands: bool = False,
        central_fields: dict[int, PinholeField] | None = None,
    ) -> None:
        """Build the LPIPS metric module on ``device`` (SSIM is computed functionally)."""
        net = LpipsNet(lpips_net)
        self.device = device
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net.value, normalize=True).to(device)
        # Validated here rather than at first use, so a bad edge list fails before any rendering.
        self.band_edges = BandAccumulator(band_edges).edges
        self.per_frame_bands = per_frame_bands
        self.central_fields = central_fields
        self._geometry_cache: dict[int, CameraGeometry] = {}

    def _field_for(self, dataset: MultiViewSequenceDataset, camera_id: int) -> PinholeField | None:
        """The reference pinhole of one camera, looked up by its rig id."""
        if self.central_fields is None:
            return None
        return self.central_fields[dataset.raw_camera_id(camera_id)]

    def _geometry_for(
        self,
        camera_id: int,
        camera_model: CameraModelType,
        K: Float[Tensor, "3 3"],
        distortion: Float[Tensor, "P"],
        width: int,
        height: int,
        observed: Bool[Tensor, "h w"] | None = None,
        field: PinholeField | None = None,
    ) -> CameraGeometry:
        """Look up (or build once) the geometry for one camera; static across timesteps."""
        cached = self._geometry_cache.get(camera_id)
        if cached is not None:
            return cached
        cached = _native_camera_geometry(
            K, distortion, camera_model, width, height, self.device, observed, self.band_edges, field
        )
        self._geometry_cache[camera_id] = cached
        return cached

    def _region_metrics(
        self,
        render: Float[Tensor, "H W 3"],
        gt: Float[Tensor, "H W 3"],
        geom: CameraGeometry,
        semantic: Bool[Tensor, "H W"],
    ) -> tuple[dict[str, float], Float[Tensor, "1 C H W"]]:
        """Region PSNR/SSIM and full-frame LPIPS (both sides zeroed outside ``valid``) for a clamped render.

        Returns:
            ``(metrics, ssim_map)``; the map is reused for the bands.
        """
        p = render.permute(2, 0, 1).unsqueeze(0)
        t = gt.permute(2, 0, 1).unsqueeze(0)
        # Functional, not the stateful ``StructuralSimilarityIndexMeasure``: with
        # ``return_full_image=True`` the module retains every ``[1, 3, H, W]`` map it has scored
        # in its ``image_return`` metric state, which accumulates the whole eval set on device.
        _, ssim_map = structural_similarity_index_measure(p, t, data_range=1.0, return_full_image=True)

        w = geom.valid.to(p.dtype)
        # Only the per-call value is used; ``forward`` also appends it to the module's
        # ``all_scores`` state, which would grow with every image scored — reset it away.
        lpips = self.lpips(p * w, t * w).item()
        self.lpips.reset()
        vals: dict[str, float] = {"lpips": lpips}
        # The shared target's perceptual score: both sides zeroed outside it, as ``lpips`` is outside ``valid``.
        if geom.central.any():
            c = geom.central.to(p.dtype)
            vals["central_lpips"] = self.lpips(p * c, t * c).item()
            self.lpips.reset()
        regions = {
            "full": geom.valid & semantic,
            "central": geom.central & semantic,
            "peripheral": geom.peripheral & semantic,
        }
        for name, region_mask in regions.items():
            vals[f"{name}_psnr"] = masked_psnr(render, gt, region_mask)
            vals[f"{name}_ssim"] = masked_ssim(ssim_map, region_mask)
        return vals, ssim_map

    def _score_frame(
        self,
        model: FourDGaussians,
        batch: dict,
        pipeline: RenderPipeline,
        sh_degree: int,
        dataset: MultiViewSequenceDataset,
        valid_masks: ValidMaskCache,
        apply_mask: bool,
        accumulators: tuple[BandAccumulator, ...],
    ) -> tuple[dict[str, list[float]], int]:
        """Render and score one frame's cameras; return their per-metric values and image count.

        The band accumulators are fed here rather than in the caller because the SSIM map and the
        native-space render/GT pair only exist inside this loop, and recomputing either per band
        would change what is being measured.
        """
        images = batch["images"].to(self.device)  # [C, H, W, 3]
        packed = valid_masks.for_batch(batch["camera_ids"], images.shape[2], images.shape[1])
        renders = render_frame_batch(model, batch, pipeline, sh_degree, self.device, packed)  # [C, H, W, 3]
        masks = batch["masks"].to(self.device) if (apply_mask and batch["masks"] is not None) else None
        camera_model = batch["camera_model"]
        width, height = images.shape[2], images.shape[1]
        cam_vals: dict[str, list[float]] = defaultdict(list)
        for c in range(renders.shape[0]):
            camera_id = batch["camera_ids"][c]
            geom = self._geometry_for(
                camera_id,
                camera_model,
                batch["Ks"][c],
                batch["distortion_params"][c],
                width,
                height,
                dataset.validity_for(camera_id),
                self._field_for(dataset, camera_id),
            )
            render = renders[c].clamp(0, 1)
            semantic = (
                masks[c][..., 0] > 0.5 if masks is not None else torch.ones_like(render[..., 0], dtype=torch.bool)
            )
            # Clamped once so every metric scores the same image; the training loss stays unclamped.
            vals, ssim_map = self._region_metrics(render, images[c], geom, semantic)
            for accumulator in accumulators:
                accumulator.add(render, images[c], ssim_map, geom.bands, semantic)
            for k, v in vals.items():
                if not math.isnan(v):
                    cam_vals[k].append(v)
        return cam_vals, int(renders.shape[0])

    @torch.no_grad()
    def evaluate(
        self,
        model: FourDGaussians,
        pipeline: RenderPipeline,
        dataset: MultiViewSequenceDataset,
        *,
        sh_degree: int | None = None,
        apply_mask: bool = True,
    ) -> NativeRegionReport:
        """Render every view and score it in its native space.

        Args:
            model: Model in eval mode.
            pipeline: Render pipeline.
            dataset: Dataset to score.
            sh_degree: SH degree; defaults to the model's.
            apply_mask: AND the per-frame foreground mask into every region.

        Returns:
            A :class:`NativeRegionReport`; a region key is absent if its mask was empty for every image.
        """
        sh = model.sh_degree if sh_degree is None else sh_degree
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
        timestamps: list[float] = []
        frame_ids: list[int] = []
        num_images = 0
        bands = BandAccumulator(self.band_edges)
        per_frame_bands: dict[int, BandAccumulator] = {}
        valid_masks = ValidMaskCache(dataset, self.device)
        for batch in tqdm(
            loader, desc="Native-region eval", unit="frame", leave=False, dynamic_ncols=True, colour="yellow"
        ):
            # The frame's identity is its capture ``frame_id``, read off any of its samples — not the
            # batch's position in the loader, and not its timestamp.
            frame_id = dataset.frame_id_for(batch["image_ids"][0])
            frame_bands = BandAccumulator(self.band_edges) if self.per_frame_bands else None
            accumulators = (bands,) if frame_bands is None else (bands, frame_bands)
            cam_vals, scored = self._score_frame(
                model, batch, pipeline, sh, dataset, valid_masks, apply_mask, accumulators
            )
            if frame_bands is not None:
                per_frame_bands[frame_id] = frame_bands
            for k, v in cam_vals.items():
                if v:
                    per_ts[k].append(sum(v) / len(v))
            num_images += scored
            timestamps.append(sum(float(t) for t in batch["timestamps"]) / len(batch["timestamps"]))
            frame_ids.append(frame_id)
        result = _aggregate(
            per_ts,
            num_timesteps=dataset.num_timesteps,
            num_cameras=dataset.cameras_per_timestep,
            num_images=num_images,
            timestamps=timestamps,
            frame_ids=frame_ids,
        )
        return NativeRegionReport(result=result, bands=bands, per_frame_bands=per_frame_bands)
