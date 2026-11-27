"""Prep stages: Resize and ToPinhole."""

from __future__ import annotations


import cv2
import numpy as np
import pytest

from anycam4d.camera.validity import compute_validity
from anycam4d.camera.validity import intrinsics_from_params
from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.stages.resize import Resize
from anycam4d.data.prep.stages.to_pinhole import ToPinhole
from anycam4d.data.schema import ReprojectSourceSchema
from anycam4d.enums import CameraModelType


def _make_view(
    cam_id: int = 0,
    frame_id: int = 0,
    timestamp: float = 0.0,
    model: CameraModelType = CameraModelType.PINHOLE,
    params: list[float] | None = None,
    mask: np.ndarray | None = None,
    h: int = 32,
    w: int = 24,
) -> CalibratedView:
    rng = np.random.default_rng(cam_id * 100 + frame_id)
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    K = np.array([[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = cam_id * 0.5
    return CalibratedView(
        cam_id=cam_id,
        frame_id=frame_id,
        timestamp=timestamp,
        image=image,
        mask=mask,
        model=model,
        K=K,
        params=params if params is not None else [100.0, 100.0, w / 2.0, h / 2.0],
        c2w=c2w,
    )


def _binary_mask(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([0, 255], dtype=np.uint8), size=(h, w))


def test_resize_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Resize()
    with pytest.raises(ValueError, match="exactly one"):
        Resize(factor=2.0, target_wh=(10, 10))


def test_resize_factor_scales_intrinsics_and_dims() -> None:
    h, w = 32, 24
    mask = _binary_mask(h, w, seed=1)
    params = [100.0, 100.0, 12.0, 16.0, 0.1, 0.2]  # opencv: 4 intrinsics + 2 distortion coeffs
    view = _make_view(model=CameraModelType.OPENCV, params=params, mask=mask, h=h, w=w)

    out = Resize(factor=2.0)(view)

    assert out.image.shape[:2] == (h // 2, w // 2)
    assert out.mask.shape == (h // 2, w // 2)
    assert out.K[0, 0] == pytest.approx(view.K[0, 0] / 2.0)
    assert out.K[1, 1] == pytest.approx(view.K[1, 1] / 2.0)
    assert out.K[0, 2] == pytest.approx(view.K[0, 2] / 2.0)
    assert out.K[1, 2] == pytest.approx(view.K[1, 2] / 2.0)
    assert out.params[0] == pytest.approx(params[0] / 2.0)
    assert out.params[2] == pytest.approx(params[2] / 2.0)
    assert out.params[4:] == pytest.approx(params[4:])  # distortion coeffs untouched
    assert set(np.unique(out.mask).tolist()) <= {0, 255}
    assert np.array_equal(out.c2w, view.c2w)


def test_resize_target_wh_uses_per_axis_ratio() -> None:
    h, w = 32, 20
    view = _make_view(h=h, w=w)
    new_w, new_h = 10, 8  # deliberately non-uniform sx != sy

    out = Resize(target_wh=(new_w, new_h))(view)

    sx, sy = new_w / w, new_h / h
    assert sx != sy
    assert out.image.shape[:2] == (new_h, new_w)
    assert out.K[0, 0] == pytest.approx(view.K[0, 0] * sx)
    assert out.K[1, 1] == pytest.approx(view.K[1, 1] * sy)
    assert out.K[0, 2] == pytest.approx(view.K[0, 2] * sx)
    assert out.K[1, 2] == pytest.approx(view.K[1, 2] * sy)


def test_to_pinhole_opencv_zero_distortion() -> None:
    h, w = 40, 30
    params = [80.0, 80.0, 15.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    view = _make_view(model=CameraModelType.OPENCV, params=params, h=h, w=w)

    out = ToPinhole()(view)

    assert out.model is CameraModelType.PINHOLE
    assert len(out.params) == 4
    assert out.image.shape == view.image.shape
    np.testing.assert_allclose(out.K, view.K, atol=2.0)
    diff = np.abs(out.image.astype(np.int16) - view.image.astype(np.int16))
    assert diff.mean() < 5.0


def test_to_pinhole_returns_pinhole_view_unchanged() -> None:
    view = _make_view(model=CameraModelType.PINHOLE)
    out = ToPinhole()(view)
    assert out is view
    # A native pinhole was never reprojected: no provenance, and validity stays trivially all-True.
    assert out.reproject_source is None


@pytest.mark.parametrize(
    ("model", "params", "h", "w"),
    [
        (CameraModelType.OPENCV, [80.0, 80.0, 15.0, 20.0, 0.05, -0.02, 0.001, 0.001, 0.0], 40, 30),
        (CameraModelType.FISHEYE_OPENCV, [60.0, 60.0, 24.0, 24.0, 0.01, -0.005, 0.001, 0.0], 48, 48),
    ],
)
def test_to_pinhole_records_native_lens_provenance(model: CameraModelType, params: list[float], h: int, w: int) -> None:
    view = _make_view(model=model, params=params, h=h, w=w)

    out = ToPinhole()(view)

    assert out.reproject_source == {"model": model.value, "width": w, "height": h, "params": params}


@pytest.mark.parametrize(
    ("model", "params", "partial"),
    [
        # A mild Brown-Conrady undistorts within its own frame -> fully observed, no invalid border.
        (CameraModelType.OPENCV, [80.0, 80.0, 24.0, 24.0, -0.3, 0.1, 0.0, 0.0, 0.0], False),
        # A wide Kannala-Brandt fisheye cannot fit its FoV on a pinhole plane -> real invalid ring.
        (CameraModelType.FISHEYE_OPENCV, [30.0, 30.0, 24.0, 24.0, 0.01, -0.005, 0.001, 0.0], True),
    ],
)
def test_to_pinhole_provenance_reproduces_the_remap_footprint(
    model: CameraModelType, params: list[float], partial: bool
) -> None:
    """Validity replayed from the emitted provenance == the pixels the rectify maps actually sourced."""
    h = w = 48
    view = _make_view(model=model, params=params, h=h, w=w)
    stage = ToPinhole()

    out = stage(view)

    K, distortion = intrinsics_from_params(out.model, out.params, 1.0)
    valid = compute_validity(out.model, K, distortion, w, h, ReprojectSourceSchema(**out.reproject_source)).numpy()

    _, map1, map2 = stage._rectify_maps(view, w, h)  # noqa: SLF001 - the maps are the ground truth here
    src_x, src_y = cv2.convertMaps(map1, map2, cv2.CV_32FC1)  # fixed-point maps -> float source coords
    in_bounds = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)

    # The two are computed independently (OpenCV's rectify maps vs. unproject+project), so allow
    # disagreement only for pixels sampling within a pixel of the source border, where the
    # fixed-point map's rounding decides the side.
    on_border = (
        (np.abs(src_x) <= 1) | (np.abs(src_x - (w - 1)) <= 1) | (np.abs(src_y) <= 1) | (np.abs(src_y - (h - 1)) <= 1)
    )
    assert in_bounds.all() != partial
    assert np.array_equal(valid[~on_border], in_bounds[~on_border])


def test_to_pinhole_builds_rectify_maps_once_per_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """The expensive map build runs once per camera, not once per frame (the perf fix)."""
    import anycam4d.data.prep.stages.to_pinhole as tp

    calls: list[int] = []
    real = tp.undistort_pinhole_family

    def counting(K, params, w, h):  # noqa: ANN001, ANN202 - test spy
        calls.append(1)
        return real(K, params, w, h)

    monkeypatch.setattr(tp, "undistort_pinhole_family", counting)

    params = [80.0, 80.0, 15.0, 20.0, 0.05, -0.02, 0.001, 0.001, 0.0]
    stage = ToPinhole()
    for frame_id in range(4):  # 4 frames x 2 cameras = 8 calls, but only 2 map builds
        for cam_id in range(2):
            stage(_make_view(cam_id=cam_id, frame_id=frame_id, model=CameraModelType.OPENCV, params=params))

    assert len(calls) == 2


def test_interpolation_guard_never_introduces_intermediate_mask_values() -> None:
    h, w = 40, 30
    mask = _binary_mask(h, w, seed=3)
    params = [80.0, 80.0, 15.0, 20.0, 0.05, -0.02, 0.001, 0.001, 0.0]
    view = _make_view(model=CameraModelType.OPENCV, params=params, mask=mask, h=h, w=w)

    resized = Resize(factor=2.0)(view)
    undistorted = ToPinhole()(view)

    assert set(np.unique(resized.mask).tolist()) <= {0, 255}
    assert set(np.unique(undistorted.mask).tolist()) <= {0, 255}
