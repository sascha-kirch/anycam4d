"""Lateral chromatic aberration: the one-pass kernel against single-lens renders, ray for ray."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType


pytestmark = pytest.mark.skipif(not HAS_CUDA_EXT, reason="chromatic aberration runs on the CUDA UT kernels")

if HAS_CUDA_EXT:
    from anycam4d.camera.valid_mask import pack_validity
    from anycam4d.renderer import chromatic_aberration as ca
    from anycam4d.renderer.chromatic_aberration import BLUE_NM
    from anycam4d.renderer.chromatic_aberration import GREEN_NM
    from anycam4d.renderer.chromatic_aberration import RED_NM
    from anycam4d.renderer.chromatic_aberration import dispersion_lenses
    from anycam4d.renderer.chromatic_aberration import magnification
    from anycam4d.renderer.chromatic_aberration import render_chromatic_aberration
    from anycam4d.renderer.ut_fused_backend import UTFusedBackend
    from tests.test_rolling_shutter import _camera
    from tests.test_rolling_shutter import _look
    from tests.test_rolling_shutter import _moving_model

CFG = RenderingConfig(backend=BackendType.ANYCAM4D)
MODELS = [CameraModelType.PINHOLE, CameraModelType.FISHEYE_OPENCV]


def _lens(lenses, i: int, valid_mask):  # noqa: ANN001, ANN202
    return dataclasses.replace(
        lenses,
        viewmats=lenses.viewmats[i : i + 1],
        Ks=lenses.Ks[i : i + 1],
        distortion=lenses.distortion[i : i + 1],
        valid_mask=valid_mask,
    )


def test_dispersion_curve() -> None:
    """Green is unmagnified and red minus blue is the strength."""
    assert magnification(GREEN_NM, 0.03) == pytest.approx(1.0)
    assert magnification(RED_NM, 0.03) - magnification(BLUE_NM, 0.03) == pytest.approx(0.03)


@pytest.mark.parametrize(("strength", "samples", "masked"), [(0.0, 1, True), (0.08, 1, False), (0.08, 2, True)])
@pytest.mark.parametrize("camera_model", MODELS)
def test_one_pass_equals_single_lens_renders(
    camera_model: CameraModelType, strength: float, samples: int, masked: bool
) -> None:
    """Channel c is the mean of the plain renders through its wavelengths' lenses: same rays, own transmittance,
    same observation mask. At zero strength every lens is the camera itself."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=3)
    camera = _camera(device, camera_model, _look(device, 0.0, 0.0))
    if masked:  # left half observed
        cols = torch.arange(camera.width, device=device) < camera.width // 2
        camera = dataclasses.replace(camera, valid_mask=pack_validity(cols.expand(camera.height, -1))[None])
    lenses = dispersion_lenses(camera, strength, samples)
    with torch.no_grad():
        colors, alphas = render_chromatic_aberration(model, camera, 0.07, 3, CFG, strength=strength, samples=samples)
        refs = [
            UTFusedBackend(CFG).render(model, _lens(lenses, i, camera.valid_mask), 0.07, 3) for i in range(3 * samples)
        ]
    for c in range(3):
        group = refs[c * samples : (c + 1) * samples]
        ref_c = torch.stack([r.colors[..., c] for r in group]).mean(0)
        ref_a = torch.stack([r.alphas[..., 0] for r in group]).mean(0)
        assert (colors[..., c] - ref_c).abs().max() <= 1e-5
        assert (alphas[..., c] - ref_a).abs().max() <= 1e-5
    assert alphas.max() > 0.5
    if strength > 0:  # the channels really differ, so the test can fail
        assert (colors[..., 0] - colors[..., 2]).abs().max() > 0.05
    if masked:
        assert alphas[..., camera.width // 2 :, :].abs().max() == 0.0


@pytest.mark.parametrize("camera_model", MODELS)
def test_union_box_contains_every_lens(camera_model: CameraModelType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Growing every tile box changes nothing: the union already holds each wavelength's drawn pixels."""
    device = torch.device("cuda")
    model = _moving_model(device, "sh4d_additive", sh_degree=3)
    camera = _camera(device, camera_model, _look(device, 0.0, 0.0))
    with torch.no_grad():
        tight, _ = render_chromatic_aberration(model, camera, 0.07, 3, CFG, strength=0.08, samples=2)
        project = ca.project

        def padded(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            centre, depth, half, *rest = project(*args, **kwargs)
            return (centre, depth, torch.where(half > 0, half + 32, half).int(), *rest)

        monkeypatch.setattr(ca, "project", padded)
        loose, _ = render_chromatic_aberration(model, camera, 0.07, 3, CFG, strength=0.08, samples=2)
    assert (tight - loose).abs().max() == 0.0
