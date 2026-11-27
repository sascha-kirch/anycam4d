"""Camera models at the boundaries: manifest parsing, distortion layout, the capability gate."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from pydantic import ValidationError

from anycam4d.camera.batched import BatchedCamera
from anycam4d.camera.registry import CAMERA_REGISTRY
from anycam4d.camera.registry import FAMILY_CUDA_ID
from anycam4d.camera.registry import ProjectionFamily
from anycam4d.camera.validity import intrinsics_from_params
from anycam4d.config.rendering import RenderingConfig
from anycam4d.cuda.ext import HAS_CUDA_EXT
from anycam4d.data.schema import CameraSchema
from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.renderer.capability import check_camera_capability
from anycam4d.renderer.pipeline import RenderPipeline


_INTR = [50.0, 50.0, 32.0, 24.0]
_MAX_PARAMS = {
    CameraModelType.PINHOLE: 4,
    CameraModelType.OPENCV: 9,
    CameraModelType.OPENCV_FULL: 16,
    CameraModelType.FISHEYE_EQUIDISTANT: 4,
    CameraModelType.FISHEYE_OPENCV: 8,
}


def _camera(model: CameraModelType, distortion: torch.Tensor | None = None) -> BatchedCamera:
    viewmat = torch.eye(4).unsqueeze(0)
    viewmat[0, 2, 3] = 3.0
    k = torch.tensor([[[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]])
    return BatchedCamera(viewmat, k, model, 32, 32, distortion)


@pytest.mark.parametrize("model", list(CameraModelType))
def test_schema_accepts_any_distortion_prefix_up_to_the_model_max(model: CameraModelType) -> None:
    """Intrinsics are mandatory; distortion is any positional prefix of the model's coefficients."""
    hi = _MAX_PARAMS[model]
    CameraSchema(camera_id=0, model=model, width=10, height=10, params=[1.0] * 4)
    CameraSchema(camera_id=0, model=model, width=10, height=10, params=[1.0] * hi)
    for bad in (3, hi + 1):
        with pytest.raises(ValidationError, match="params"):
            CameraSchema(camera_id=0, model=model, width=10, height=10, params=[1.0] * bad)


@pytest.mark.parametrize("name", ["ortho", "ftheta", "not_a_camera"])
def test_schema_rejects_unknown_models(name: str) -> None:
    with pytest.raises(ValidationError):
        CameraSchema(camera_id=0, model=name, width=10, height=10, params=_INTR)


@pytest.mark.parametrize(
    ("model", "coeffs", "expected"),
    [
        (CameraModelType.PINHOLE, [], [0.0] * 12),
        (CameraModelType.OPENCV, [0.1, -0.02], [0.1, -0.02] + [0.0] * 10),
        (CameraModelType.OPENCV, [0.0, 0.0, 0.005, -0.003], [0.0, 0.0, 0.005, -0.003] + [0.0] * 8),
        (CameraModelType.OPENCV_FULL, [0.01 * (i + 1) for i in range(12)], [0.01 * (i + 1) for i in range(12)]),
        (CameraModelType.FISHEYE_OPENCV, [0.1, -0.02, 0.003, -0.0004], [0.1, -0.02, 0.003, -0.0004]),
        (CameraModelType.FISHEYE_EQUIDISTANT, [], [0.0] * 4),
    ],
)
def test_distortion_fills_the_canonical_family_slots(
    model: CameraModelType, coeffs: list[float], expected: list[float]
) -> None:
    """Slot order is OpenCV's ``distCoeffs`` order, zero-filled to the family width (12 or 4)."""
    k, dist = intrinsics_from_params(model, [*_INTR, *coeffs], 1.0)
    assert torch.equal(dist, torch.tensor(expected, dtype=torch.float32))
    assert k[0, 0] == 50.0 and k[0, 2] == 32.0


def test_scale_factor_divides_intrinsics_not_distortion() -> None:
    coeffs = [0.1, -0.02, 0.001, -0.002, 0.003]
    k, dist = intrinsics_from_params(CameraModelType.OPENCV, [*_INTR, *coeffs], 2.0)
    assert k[0, 0] == pytest.approx(25.0)
    assert k[0, 2] == pytest.approx(16.0)
    assert torch.equal(dist[:5], torch.tensor(coeffs, dtype=torch.float32))


def test_batched_camera_rejects_a_foreign_distortion_width() -> None:
    with pytest.raises(AssertionError):
        _camera(CameraModelType.FISHEYE_OPENCV, torch.zeros(1, 5))
    _camera(CameraModelType.FISHEYE_OPENCV, torch.zeros(1, 4))


@pytest.mark.parametrize("model", list(CameraModelType))
@pytest.mark.parametrize("backend", [BackendType.EWA, BackendType.ANYCAM4D])
def test_capability_gate_admits_every_model_on_both_backends(model: CameraModelType, backend: BackendType) -> None:
    """EWA linearizes each lens at the mean, UT samples it: both render every model."""
    check_camera_capability(model, backend, None)


def test_capability_gate_rejects_a_foreign_distortion_width() -> None:
    with pytest.raises(ValueError):
        check_camera_capability(CameraModelType.FISHEYE_OPENCV, BackendType.ANYCAM4D, torch.zeros(1, 5))


class _SpyBackend:
    def __init__(self) -> None:
        self.cfg = RenderingConfig(backend=BackendType.EWA)

    def render(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("backend.render must not be reached when the gate rejects")


def test_pipeline_gates_before_the_backend_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = CAMERA_REGISTRY[CameraModelType.FISHEYE_OPENCV]
    monkeypatch.setitem(
        CAMERA_REGISTRY,
        CameraModelType.FISHEYE_OPENCV,
        dataclasses.replace(spec, backends=frozenset({BackendType.ANYCAM4D})),
    )
    pipeline = RenderPipeline(backend=_SpyBackend())
    with pytest.raises(NotImplementedError):
        pipeline.render(None, _camera(CameraModelType.FISHEYE_OPENCV), timestamp=0.0, sh_degree=0)


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="requires the CUDA extension")
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize(
    ("model", "coeffs"),
    [
        (CameraModelType.OPENCV, [-0.2, 0.05, 0.01, -0.01, 0.0]),
        (CameraModelType.FISHEYE_OPENCV, [0.1, -0.02, 0.0, 0.0]),
    ],
)
def test_fused_ewa_renders_through_the_camera_distortion(
    random_model: FourDGaussians, model: CameraModelType, coeffs: list[float], packed: bool
) -> None:
    """The distortion reaches the kernel: the render differs from the same camera with zeroed distortion."""
    from anycam4d.renderer.ewa_fused_backend import FusedEWABackend

    device = next(random_model.parameters()).device
    dist = torch.zeros(1, 12 if model is CameraModelType.OPENCV else 4)
    dist[0, : len(coeffs)] = torch.tensor(coeffs)
    backend = FusedEWABackend(RenderingConfig(packed=packed))
    with torch.no_grad():
        lens = backend.render(random_model, _camera(model, dist).to(device), 0.0, random_model.sh_degree)
        flat = backend.render(
            random_model, _camera(model, torch.zeros_like(dist)).to(device), 0.0, random_model.sh_degree
        )
    assert lens.alphas.sum() > 0
    assert (lens.alphas - flat.alphas).abs().max() > 1e-2


@pytest.mark.skipif(not HAS_CUDA_EXT, reason="requires the CUDA extension")
def test_family_ids_mirror_the_extension_enum() -> None:
    """``FAMILY_CUDA_ID`` must equal the C++ ``ProjectionFamily`` values, which the kernels dispatch on."""
    from anycam4d.cuda.ext import ProjectionFamily as CudaProjectionFamily

    assert FAMILY_CUDA_ID == {
        ProjectionFamily.PINHOLE_FAMILY: int(CudaProjectionFamily.PINHOLE_FAMILY),
        ProjectionFamily.FISHEYE_FAMILY: int(CudaProjectionFamily.FISHEYE_FAMILY),
    }
