"""The host observation mask, bitpacked into the world rasterizer.

Analytic lens gates cannot see reprojection provenance or on-disk masks; uploading the host mask
makes the compositing gate and the loss mask the same object.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.valid_mask import ValidMaskCache
from anycam4d.camera.valid_mask import pack_validity
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.cuda.ext import ProjectionFamily
from anycam4d.data.multiview_sequence_dataset import MultiViewSequenceDataset
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline


requires_cuda = pytest.mark.skipif(not HAS_CUDA_EXT, reason="render path runs on the CUDA kernels")

TILE = 16


def unpack_validity(packed: torch.Tensor, width: int) -> torch.Tensor:
    """Inverse of ``pack_validity``."""
    shifts = torch.arange(32, device=packed.device, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :width].bool()


@pytest.mark.parametrize("width", [1, 5, 31, 32, 33, 64, 70])
def test_pack_unpack_round_trip(width: int) -> None:
    """Packing and unpacking is the identity for any width, ragged last word included."""
    torch.manual_seed(width)
    mask = torch.rand(7, width) > 0.5
    packed = pack_validity(mask)

    assert packed.dtype == torch.int32
    assert packed.shape == (7, (width + 31) // 32)
    assert torch.equal(unpack_validity(packed, width), mask)


def test_pack_padding_bits_are_zero() -> None:
    """The bits past ``width`` in a ragged last word are 0, never a copy of the last pixel."""
    width = 33
    packed = pack_validity(torch.ones(2, width, dtype=torch.bool))
    bits = unpack_validity(packed, packed.shape[1] * 32)

    assert bits[:, :width].all()
    assert not bits[:, width:].any()


def test_pack_all_true_last_word_is_negative() -> None:
    """A full 32-bit word is -1 in int32 (two's complement) — the packing must not overflow."""
    packed = pack_validity(torch.ones(1, 32, dtype=torch.bool))
    assert int(packed[0, 0]) == -1


def _probe_scene(device: torch.device, W: int, H: int, C: int, *, n_dist: int):
    """Scene + tile structure for ``C`` identical cameras staring at one huge Gaussian."""
    from gsplat.cuda._wrapper import isect_offset_encode
    from gsplat.cuda._wrapper import isect_tiles

    tile_h, tile_w = math.ceil(H / TILE), math.ceil(W / TILE)
    means_3d_cond = torch.tensor([[0.0, 0.0, 1.0]], device=device)
    factors = torch.tensor([[1e2, 0.0, 0.0, 1e2, 0.0, 1e2]], device=device)  # Σ = 1e4·I
    opacities = torch.tensor([0.9], device=device)
    colors = torch.ones(C, 1, 3, device=device)

    means2d = torch.zeros(C, 1, 2, device=device)
    depths = torch.ones(C, 1, device=device)
    radii = torch.full((C, 1, 2), max(W, H) * 4, dtype=torch.int32, device=device)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, TILE, tile_w, tile_h, packed=False, n_images=C)
    offsets = isect_offset_encode(isect_ids, C, tile_w, tile_h).reshape(C, tile_h, tile_w)
    dist = torch.zeros((C, n_dist), device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(C, 1, 1).contiguous()
    return means_3d_cond, factors, opacities, colors, offsets.contiguous(), flatten_ids.contiguous(), dist, viewmats


def _probe_render(
    K: torch.Tensor,
    family: ProjectionFamily,
    W: int,
    H: int,
    *,
    n_dist: int,
    masks: list[torch.Tensor] | None,
    C: int = 1,
):
    """Render the probe scene; returns ``(render_colors, render_alphas, last_ids)``."""
    from anycam4d.cuda.ext import rasterize_from_world_fwd

    device = torch.device("cuda")
    means_3d_cond, factors, opac, colors, offsets, flatten_ids, dist, viewmats = _probe_scene(
        device, W, H, C, n_dist=n_dist
    )
    Ks = K.to(device).unsqueeze(0).repeat(C, 1, 1).contiguous()
    packed = None
    if masks is not None:
        packed = torch.stack([pack_validity(m.to(device)) for m in masks]).contiguous()
    return rasterize_from_world_fwd(
        means_3d_cond,
        factors,
        opac,
        colors,
        viewmats,
        Ks,
        dist,
        offsets,
        flatten_ids,
        W,
        H,
        int(family),
        packed,
    )


def _composited(alphas: torch.Tensor, last_ids: torch.Tensor) -> torch.Tensor:
    """``[H, W]`` bool: pixels that actually accumulated a contributor."""
    return (alphas > 0.0) & (last_ids >= 0)


_FAMILY_CASES = [
    (ProjectionFamily.PINHOLE_FAMILY, 12, torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])),
    (ProjectionFamily.FISHEYE_FAMILY, 4, torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])),
]


@requires_cuda
@pytest.mark.parametrize(("family", "n_dist", "K"), _FAMILY_CASES)
def test_mask_is_honoured_per_family(family: ProjectionFamily, n_dist: int, K: torch.Tensor) -> None:
    """Every family composites exactly the pixels the mask marks observed."""
    W = H = 48
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[8:40, 5:37] = True  # an off-centre box, so a wrong stride cannot pass by symmetry

    _, alphas, last = _probe_render(K, family, W, H, n_dist=n_dist, masks=[mask])

    assert torch.equal(_composited(alphas[0], last[0]).cpu(), mask)


@requires_cuda
@pytest.mark.parametrize(("family", "n_dist", "K"), _FAMILY_CASES)
def test_null_mask_equals_all_ones_mask(family: ProjectionFamily, n_dist: int, K: torch.Tensor) -> None:
    """The dense path (no mask) and an all-observed mask render bit-identically."""
    W = H = 48
    ones = torch.ones(H, W, dtype=torch.bool)

    dense = _probe_render(K, family, W, H, n_dist=n_dist, masks=None)
    masked = _probe_render(K, family, W, H, n_dist=n_dist, masks=[ones])

    for a, b in zip(dense, masked):
        assert torch.equal(a, b)


@requires_cuda
def test_unobserved_pixel_is_exact_background() -> None:
    """An unobserved pixel keeps alpha exactly 0, colour exactly 0 and no last contributor."""
    W = H = 48
    K = torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[:, : W // 2] = True

    colors, alphas, last = _probe_render(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, masks=[mask])

    unobserved = ~mask.to(colors.device)
    assert torch.all(alphas[0][unobserved] == 0.0)
    assert torch.all(colors[0][unobserved] == 0.0)
    assert torch.all(last[0][unobserved] < 0)
    assert torch.all(alphas[0][~unobserved] > 0.0), "the observed half must still render"


@requires_cuda
def test_each_camera_reads_its_own_mask() -> None:
    """At C > 1 each camera gates by its own mask (pins the per-camera stride)."""
    W = H = 48
    K = torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]])
    left = torch.zeros(H, W, dtype=torch.bool)
    left[:, : W // 2] = True
    top = torch.zeros(H, W, dtype=torch.bool)
    top[: H // 2, :] = True

    _, alphas, last = _probe_render(K, ProjectionFamily.PINHOLE_FAMILY, W, H, n_dist=12, masks=[left, top], C=2)

    assert torch.equal(_composited(alphas[0], last[0]).cpu(), left)
    assert torch.equal(_composited(alphas[1], last[1]).cpu(), top)


@requires_cuda
def test_no_gradient_flows_from_a_masked_pixel() -> None:
    """A loss supported only on unobserved pixels produces exactly zero parameter gradient."""
    from anycam4d.cuda.autograd import RasterizeWorldFunction

    device = torch.device("cuda")
    W = H = 48
    means_3d_cond, factors, opac, colors, offsets, flatten_ids, dist, viewmats = _probe_scene(
        device, W, H, 1, n_dist=12
    )
    means_3d_cond = means_3d_cond.clone().requires_grad_(True)
    colors = colors.clone().requires_grad_(True)
    opac = opac.clone().requires_grad_(True)
    K = torch.tensor([[40.0, 0.0, 24.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]], device=device).unsqueeze(0).contiguous()

    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    mask[:, : W // 2] = True
    packed = pack_validity(mask).unsqueeze(0).contiguous()

    render_colors, render_alphas = RasterizeWorldFunction.apply(
        means_3d_cond, factors, opac, colors, viewmats, K, dist, offsets, flatten_ids, W, H, 0, packed
    )
    # Support the loss only where the camera observed nothing.
    loss = (render_colors[:, :, W // 2 :, :] ** 2).sum() + (render_alphas[:, :, W // 2 :] ** 2).sum()
    loss.backward()

    assert float(loss.detach()) == 0.0
    for name, grad in (("means_3d_cond", means_3d_cond.grad), ("colors", colors.grad), ("opacities", opac.grad)):
        assert grad is not None and torch.all(grad == 0.0), f"{name} received gradient from an unobserved pixel"


def _front_facing_model(n: int, device: torch.device, tau: float) -> FourDGaussians:
    torch.manual_seed(3)
    model = FourDGaussians.from_random(n, sh_degree=1).to(device)
    depths = torch.linspace(2.0, 6.0, n, device=device)
    angles = torch.linspace(-0.2, 0.2, n, device=device)
    means = torch.stack([torch.tan(angles) * depths, torch.tan(angles.flip(0)) * depths, depths], dim=-1)
    with torch.no_grad():
        model.splats["means"].copy_(means)
        model.splats["mu_t"].fill_(tau)
        model.splats["opacities"].fill_(2.0)
        model.splats["sh0"].normal_(0.0, 0.3)
    return model


def _cameras(device: torch.device, n_cams: int, W: int, H: int) -> BatchedCamera:
    K = torch.tensor([[40.0, 0.0, W / 2], [0.0, 40.0, H / 2], [0.0, 0.0, 1.0]], device=device)
    distortion = torch.zeros(n_cams, 12, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0).repeat(n_cams, 1, 1)
    for c in range(n_cams):
        viewmats[c, 0, 3] = -0.4 + 0.8 * c / max(1, n_cams - 1)
    mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    mask[4 : H - 4, 6:] = True
    valid_mask = torch.stack([pack_validity(mask)] * n_cams).contiguous()
    return BatchedCamera(
        viewmats.contiguous(),
        K.unsqueeze(0).repeat(n_cams, 1, 1).contiguous(),
        CameraModelType.PINHOLE,
        W,
        H,
        distortion=distortion,
        valid_mask=valid_mask,
    )


@requires_cuda
@pytest.mark.parametrize("n_cams", [1, 3])
def test_dense_matches_packed_with_mask(n_cams: int) -> None:
    """UT renders the same image packed or dense once the mask is carried."""
    device = torch.device("cuda")
    W = H = 32
    tau = 0.3
    cameras = _cameras(device, n_cams, W, H)
    model = _front_facing_model(24, device, tau)

    outs = {}
    for packed in (False, True):
        pipeline = RenderPipeline(
            backend=BackendFactory.create(RenderingConfig(backend=BackendType.ANYCAM4D, packed=packed))
        )
        with torch.no_grad():
            outs[packed] = pipeline.render(model, cameras, tau, model.sh_degree)

    torch.testing.assert_close(outs[True].colors, outs[False].colors, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(outs[True].alphas, outs[False].alphas, rtol=1e-4, atol=1e-5)
    unobserved = unpack_validity(cameras.valid_mask[0], W).logical_not()
    assert torch.all(outs[True].alphas[:, unobserved] == 0.0)
    assert torch.all(outs[False].alphas[:, unobserved] == 0.0)


def _write_dataset(tmp_path: Path, camera_spec: dict, *, width: int, height: int) -> Path:
    """A one-camera, two-frame MultiViewSequence dataset with the given camera block."""
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(5)
    cameras = [{**camera_spec, "camera_id": 0}]
    captures = []
    for frame_idx, t in enumerate([0.0, 1.0 / 30.0]):
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        name = f"cam00_{frame_idx:04d}"
        Image.fromarray(rgb, mode="RGB").save(tmp_path / "images" / f"{name}.png")
        captures.append(
            {
                "image_path": f"images/{name}.png",
                "camera_id": 0,
                "frame_id": frame_idx,
                "timestamp": t,
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 3.0],
            }
        )
    schema = {
        "schema_version": 1,
        "metadata": {
            "world_coordinate_system": "right_handed",
            "camera_coordinate_system": "opencv",
            "transform_convention": "camera_to_world",
        },
        "cameras": cameras,
        "captures": captures,
    }
    with open(tmp_path / "train_dataset.json", "w") as f:
        json.dump(schema, f)
    return tmp_path


_W = _H = 48


def _pipeline() -> RenderPipeline:
    return RenderPipeline(backend=BackendFactory.create(RenderingConfig(backend=BackendType.ANYCAM4D, packed=False)))


def _render_dataset_camera(data_dir: Path, *, with_mask: bool):
    """Render camera 0 of ``data_dir``; returns ``(alphas [H, W], validity [H, W])``."""
    device = torch.device("cuda")
    ds = MultiViewSequenceDataset(data_dir, split="train")
    assert not ds.fully_observed, "fixture must have a non-trivial footprint"
    validity = ds.validity_for(0)
    cache = ValidMaskCache(ds, device)
    packed = cache.for_batch([0], _W, _H) if with_mask else None

    sample = ds[0]
    camera = BatchedCamera.from_single_view(
        sample.viewmat.to(device),
        sample.K.to(device),
        sample.camera_model,
        _W,
        _H,
        distortion=sample.distortion_params.to(device),
    )
    camera = BatchedCamera(
        camera.viewmats, camera.Ks, camera.model, _W, _H, distortion=camera.distortion, valid_mask=packed
    )
    model = _front_facing_model(64, device, 0.0)
    with torch.no_grad():
        out = _pipeline().render(model, camera, 0.0, model.sh_degree)
    return out.alphas[0], validity.to(device)


@pytest.fixture()
def reprojected_pinhole_dataset(tmp_path: Path) -> Path:
    """A pinhole (f = 15) rectified from a narrower source (f = 60): its periphery was never observed."""
    spec = {
        "model": "pinhole",
        "width": _W,
        "height": _H,
        "params": [15.0, 15.0, (_W - 1) / 2.0, (_H - 1) / 2.0],
        "reproject_source": {
            "model": "pinhole",
            "width": _W,
            "height": _H,
            "params": [60.0, 60.0, (_W - 1) / 2.0, (_H - 1) / 2.0],
        },
    }
    return _write_dataset(tmp_path, spec, width=_W, height=_H)


@pytest.fixture()
def on_disk_override_dataset(tmp_path: Path) -> Path:
    """A plain pinhole whose footprint is an arbitrary hand-authored raster in ``valid/``."""
    spec = {
        "model": "pinhole",
        "width": _W,
        "height": _H,
        "params": [40.0, 40.0, (_W - 1) / 2.0, (_H - 1) / 2.0],
    }
    data_dir = _write_dataset(tmp_path, spec, width=_W, height=_H)
    raster = np.zeros((_H, _W), dtype=np.uint8)
    raster[6:30, 9:41] = 255  # no lens produces this shape
    (data_dir / "valid").mkdir()
    Image.fromarray(raster, mode="L").save(data_dir / "valid" / "0.png")
    return data_dir


@requires_cuda
@pytest.mark.parametrize(
    "fixture_name",
    ["reprojected_pinhole_dataset", "on_disk_override_dataset"],
)
def test_provenance_footprint_renders_exact_background(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Every validity branch gates the render; without the mask the unobserved region gets content."""
    data_dir = request.getfixturevalue(fixture_name)

    alphas, validity = _render_dataset_camera(data_dir, with_mask=True)
    unobserved = ~validity

    assert unobserved.any(), "fixture must have unobserved pixels"
    assert torch.all(alphas[unobserved] == 0.0)
    assert torch.any(alphas[validity] > 0.0), "the observed region must still render content"

    ungated, _ = _render_dataset_camera(data_dir, with_mask=False)
    assert torch.any(ungated[unobserved] > 0.0), "without the mask this footprint is not gated at all"


class _MixedRig:
    """Dataset stand-in: camera 0 a circular fisheye (masked, 8x8), camera 1 a pinhole (fully observed, 6x8)."""

    camera_ids = [0, 1]

    def __init__(self) -> None:
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        self._validity = {0: (yy - 3.5) ** 2 + (xx - 3.5) ** 2 < 16, 1: torch.ones(6, 8, dtype=torch.bool)}

    def validity_for(self, camera_id: int) -> torch.Tensor:
        return self._validity[camera_id]


def test_mixed_resolution_rig_masks_only_partially_observed_cameras() -> None:
    """A fully observed camera takes the dense path at its own resolution; a masked one keeps its mask."""
    rig = _MixedRig()
    cache = ValidMaskCache(rig, torch.device("cpu"))

    assert cache.for_batch([1], 8, 6) is None
    assert torch.equal(unpack_validity(cache.for_batch([0], 8, 8)[0], 8), rig.validity_for(0))
    with pytest.raises(ValueError, match="camera 0"):
        cache.for_batch([0], 8, 6)
