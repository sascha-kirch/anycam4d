"""Headless unit tests for the viewer render adapter (camera math, caches, count throttle)."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch

from anycam4d.data.normalization import normalize_viewmat

from anycam4d.enums import CameraModelType
from anycam4d.math.lens_domain import fisheye_domain_angle
from anycam4d.math.lens_domain import fisheye_theta_d
from anycam4d.renderer.backend import RenderOutput
from anycam4d.viewer.panels.render_settings import RenderSettings
from anycam4d.viewer.render_adapter import RenderAdapter
from anycam4d.viewer.viewer import _RENDERED_COUNT_THROTTLE
from anycam4d.viewer.viewer import Anycam4DViewer


def _random_c2w() -> torch.Tensor:
    """A valid camera-to-world: rotation from QR + translation."""
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    c2w = torch.eye(4)
    c2w[:3, :3] = q
    c2w[:3, 3] = torch.randn(3)
    return c2w


def test_viewmat_without_t_denorm_is_plain_inverse() -> None:
    """With no normalisation transform, viewmat is the inverse of c2w."""
    c2w = _random_c2w()
    viewmat = RenderAdapter.viewmat_from_c2w(c2w, None)
    assert torch.allclose(viewmat, torch.linalg.inv(c2w), atol=1e-5)


def test_viewmat_applies_t_denorm_when_present() -> None:
    """T_denorm is applied iff present, the same way training normalizes (covers the export-divergence root cause)."""
    c2w = _random_c2w()
    t_denorm = torch.eye(4)
    t_denorm[:3, :3] *= 2.5
    t_denorm[:3, 3] = torch.tensor([0.1, -0.2, 0.3])
    viewmat = RenderAdapter.viewmat_from_c2w(c2w, t_denorm)
    assert torch.allclose(viewmat, normalize_viewmat(torch.linalg.inv(c2w), t_denorm), atol=1e-5)


def test_k_from_fov_focal_and_principal_point() -> None:
    """Intrinsics from a vertical FOV: focal from height, principal point at center."""
    fov, w, h = np.pi / 2.0, 64, 48
    k = RenderAdapter.K_from_fov(fov, w, h, torch.device("cpu"))
    expected_focal = h / 2.0 / np.tan(fov / 2.0)
    assert k[0, 0].item() == k[1, 1].item()
    assert abs(k[0, 0].item() - expected_focal) < 1e-4
    assert k[0, 2].item() == w / 2.0
    assert k[1, 2].item() == h / 2.0


class _CapturePipeline:
    """Captures the BatchedCamera handed to render()."""

    def __init__(self) -> None:
        self.last_camera = None
        self.last_colors_override = None
        self.fill = 0.0

    def render(self, model, camera, tau, sh_degree, *, backgrounds, colors_override):  # noqa: ANN001
        self.last_camera = camera
        self.last_colors_override = colors_override
        h, w = camera.height, camera.width
        return RenderOutput(torch.full((1, h, w, 3), self.fill), torch.full((1, h, w, 1), self.fill), {})


def _stub_viewer(camera_model: CameraModelType, t_denorm: torch.Tensor | None = None) -> types.SimpleNamespace:
    """Namespace with the viewer attributes the adapter reads; applying the Lens panel is a no-op."""
    return types.SimpleNamespace(
        _device=torch.device("cpu"),
        _T_denorm=t_denorm,
        _model=types.SimpleNamespace(sh_degree=0),
        _pipeline=_CapturePipeline(),
        _postprocess=lambda out: out.colors[0, ..., :3].clamp(0.0, 1.0).cpu().numpy(),
        _apply_lens=lambda device: None,  # noqa: ARG005
        render_settings=RenderSettings(active_camera_model=camera_model),
    )


def test_export_honors_active_camera_model() -> None:
    """render_at builds the camera with the viewer's active model, not a hardcoded pinhole."""
    viewer = _stub_viewer(CameraModelType.FISHEYE_OPENCV)
    adapter = RenderAdapter(viewer)
    adapter.render_at(c2w=_random_c2w().numpy(), fov_rad=np.pi / 2.0, width=8, height=6, tau=0.0)
    assert viewer._pipeline.last_camera.model == CameraModelType.FISHEYE_OPENCV


def test_render_at_applies_camera_panels_before_reading_settings() -> None:
    """``render_at`` applies the Lens panel state before reading render settings, like ``render_fn``."""
    calls: list[str] = []
    viewer = _stub_viewer(CameraModelType.PINHOLE)
    viewer._apply_lens = lambda device: calls.append("lens")  # noqa: ARG005
    adapter = RenderAdapter(viewer)
    adapter.render_at(c2w=_random_c2w().numpy(), fov_rad=np.pi / 2.0, width=8, height=6, tau=0.0)
    assert calls == ["lens"]


def test_render_at_uses_shared_viewmat_math() -> None:
    """The camera viewmat built by render_at matches the shared helper."""
    t_denorm = _random_c2w()
    viewer = _stub_viewer(CameraModelType.PINHOLE, t_denorm=t_denorm)
    adapter = RenderAdapter(viewer)
    c2w = _random_c2w()
    adapter.render_at(c2w=c2w.numpy(), fov_rad=np.pi / 2.0, width=8, height=6, tau=0.0)
    expected = RenderAdapter.viewmat_from_c2w(c2w, t_denorm)
    assert torch.allclose(viewer._pipeline.last_camera.viewmats[0], expected, atol=1e-5)


def test_background_tensor_reused_until_color_changes() -> None:
    """The background tensor is cached and rebuilt only when the color changes."""
    viewer = _stub_viewer(CameraModelType.PINHOLE)
    adapter = RenderAdapter(viewer)
    first = adapter._background_tensor()
    assert adapter._background_tensor() is first
    viewer.render_settings.background_color = (1.0, 0.0, 0.0)
    assert adapter._background_tensor() is not first


def _temporal_viewer(colormap: str = "turbo") -> types.SimpleNamespace:
    """Stub viewer in temporal render mode with a mu_t-carrying model."""
    model = types.SimpleNamespace(splats={"mu_t": torch.linspace(-0.5, 0.5, 16)})
    return types.SimpleNamespace(
        render_settings=RenderSettings(render_mode="temporal", colormap=colormap), _model=model
    )


def test_temporal_colormap_built_once_and_reused() -> None:
    """An unchanged (model, colormap, mode) reuses the cached temporal colors."""
    viewer = _temporal_viewer()
    adapter = RenderAdapter(viewer)
    first = adapter._temporal_colors_override(0.0)
    assert first is not None
    assert adapter._temporal_colors_override(0.0) is first


def test_temporal_colormap_rebuilt_on_colormap_change() -> None:
    """A colormap change invalidates the cached temporal colors."""
    viewer = _temporal_viewer()
    adapter = RenderAdapter(viewer)
    first = adapter._temporal_colors_override(0.0)
    viewer.render_settings.colormap = "viridis"
    assert adapter._temporal_colors_override(0.0) is not first


def test_temporal_colormap_rebuilt_on_model_update() -> None:
    """on_model_update clears the cached temporal colors."""
    viewer = _temporal_viewer()
    adapter = RenderAdapter(viewer)
    first = adapter._temporal_colors_override(0.0)
    adapter.on_model_update()
    assert adapter._temporal_colors_override(0.0) is not first


def test_temporal_colors_none_outside_temporal_mode() -> None:
    """No temporal colors are produced when the viewer is not in temporal mode."""
    viewer = _temporal_viewer()
    viewer.render_settings.render_mode = "rgb"
    adapter = RenderAdapter(viewer)
    assert adapter._temporal_colors_override(0.0) is None


def _full_temporal_model(n: int = 16) -> types.SimpleNamespace:
    """A model carrying every parameter the std mode needs (deterministic)."""
    torch.manual_seed(0)
    quats_l = torch.randn(n, 4)
    quats_r = torch.randn(n, 4)
    scales_4d_log = torch.randn(n, 4)
    return types.SimpleNamespace(
        splats={
            "mu_t": torch.linspace(-0.5, 0.5, n),
            "quats_l": quats_l,
            "quats_r": quats_r,
            "scales_4d_log": scales_4d_log,
        }
    )


def _full_temporal_viewer(mode: str = "mean") -> types.SimpleNamespace:
    """Temporal-mode stub viewer with a fully-populated model and the given color mode."""
    return types.SimpleNamespace(
        render_settings=RenderSettings(render_mode="temporal", temporal_color_mode=mode),
        _model=_full_temporal_model(),
        _apply_lens=lambda device: None,  # noqa: ARG005
    )


def test_distance_mode_depends_on_tau() -> None:
    """distance override changes with tau and is never cached across calls."""
    viewer = _full_temporal_viewer("distance")
    adapter = RenderAdapter(viewer)
    a = adapter._temporal_colors_override(-0.5)
    b = adapter._temporal_colors_override(0.5)
    assert not torch.allclose(a, b)
    # distance bypasses the static cache: a fresh tensor each call.
    assert adapter._temporal_colors_override(0.5) is not b


def test_mean_and_std_are_tau_independent() -> None:
    """mean/std overrides do not change with tau."""
    for mode in ("mean", "std"):
        viewer = _full_temporal_viewer(mode)
        adapter = RenderAdapter(viewer)
        assert torch.allclose(adapter._temporal_colors_override(-0.4), adapter._temporal_colors_override(0.4))


def test_std_mode_derives_from_sigma_tt() -> None:
    """std colors derive from sqrt(cov_4d[:, 3, 3]) min-max normalized, then colormap."""
    from anycam4d.math.covariance import build_cov4d_rot
    from nerfview import apply_float_colormap

    viewer = _full_temporal_viewer("std")
    adapter = RenderAdapter(viewer)
    got = adapter._temporal_colors_override(0.0)

    splats = viewer._model.splats
    sigma_4d = build_cov4d_rot(splats["quats_l"], splats["quats_r"], splats["scales_4d_log"])
    sigma_t = sigma_4d[:, 3, 3].clamp_min(0.0).sqrt()
    lo = sigma_t.min()
    normalized = ((sigma_t - lo) / (sigma_t.max() - lo + 1e-8)).unsqueeze(-1)
    expected = apply_float_colormap(normalized, "turbo")
    assert torch.allclose(got, expected)


def test_mean_mode_colors_mu_t_through_turbo() -> None:
    """The mean override is mu_t (shifted to [0, 1]) through the turbo colormap."""
    from nerfview import apply_float_colormap

    viewer = _full_temporal_viewer("mean")
    adapter = RenderAdapter(viewer)
    got = adapter._temporal_colors_override(0.123)

    mu_t = viewer._model.splats["mu_t"]
    expected = apply_float_colormap((mu_t + 0.5).clamp(0, 1).unsqueeze(-1), "turbo")
    assert torch.equal(got, expected)


def test_mode_change_invalidates_static_cache() -> None:
    """Switching temporal_color_mode rebuilds the cached static override."""
    viewer = _full_temporal_viewer("mean")
    adapter = RenderAdapter(viewer)
    first = adapter._temporal_colors_override(0.0)
    viewer.render_settings.temporal_color_mode = "std"
    assert adapter._temporal_colors_override(0.0) is not first


def test_export_passes_temporal_override_to_pipeline() -> None:
    """render_at hands the same temporal override the active mode/τ would use (was None)."""
    viewer = _full_temporal_viewer("mean")
    viewer._device = torch.device("cpu")
    viewer._T_denorm = None
    viewer._model.sh_degree = 0
    viewer._pipeline = _CapturePipeline()
    viewer._postprocess = lambda out: out.colors[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()

    adapter = RenderAdapter(viewer)
    expected = adapter._temporal_colors_override(0.25)
    adapter.render_at(c2w=_random_c2w().numpy(), fov_rad=np.pi / 2.0, width=8, height=6, tau=0.25)
    assert viewer._pipeline.last_colors_override is not None
    assert torch.equal(viewer._pipeline.last_colors_override, expected)


def test_export_runs_postprocess() -> None:
    """render_at routes through viewer._postprocess (so alpha/temporal export honors it)."""
    viewer = _stub_viewer(CameraModelType.PINHOLE)
    viewer._postprocess = lambda out: out.alphas[0, ..., :1].cpu().numpy()  # marker postprocess
    adapter = RenderAdapter(viewer)
    img = adapter.render_at(c2w=_random_c2w().numpy(), fov_rad=np.pi / 2.0, width=8, height=6, tau=0.0)
    # the captured alphas are zeros → uint8 zeros, shape [H, W, 1] from the marker postprocess.
    assert img.shape == (6, 8, 1)
    assert img.dtype == np.uint8


def _viewer_for_count() -> Anycam4DViewer:
    """A bare viewer instance (no GUI/server) with just the count machinery wired."""
    viewer = Anycam4DViewer.__new__(Anycam4DViewer)
    viewer._frame_counter = 0
    viewer._info_panel = types.SimpleNamespace(rendered_primitives=types.SimpleNamespace(value=0))
    return viewer


def test_rendered_count_updates_only_on_throttle_schedule() -> None:
    """The count display is written only every Nth frame, never on the intervening frames."""
    viewer = _viewer_for_count()
    meta = {"gaussian_ids": torch.arange(7)}
    for _ in range(_RENDERED_COUNT_THROTTLE - 1):
        viewer._update_rendered_count(meta)
    assert viewer._info_panel.rendered_primitives.value == 0
    viewer._update_rendered_count(meta)
    assert viewer._info_panel.rendered_primitives.value == 7


def test_count_rendered_does_not_use_unique_on_packed_ids() -> None:
    """gaussian_ids count is read from the tensor (numel), no .unique() sync."""
    viewer = _viewer_for_count()
    # Duplicated ids would inflate a numel count but be collapsed by .unique();
    # for the single-camera viewer there are no duplicates, so numel is correct.
    assert viewer._count_rendered({"gaussian_ids": torch.tensor([0, 1, 2, 3])}) == 4


@pytest.mark.parametrize(("circular", "width", "height"), [(True, 800, 600), (True, 600, 800), (False, 800, 600)])
def test_fisheye_image_circle_fits_frame_or_diagonal(circular: bool, width: int, height: int) -> None:
    """The lens field's rim lands on the shorter side (circular, in either orientation) or the diagonal."""
    K = RenderAdapter.fisheye_K(torch.zeros(4), 180.0, circular, width, height)
    radius = min(width, height) / 2 if circular else math.hypot(width, height) / 2
    assert torch.isclose(K[1, 1] * math.pi / 2, torch.tensor(radius))
    assert K[0, 2] == width / 2 and K[1, 2] == height / 2


def test_fisheye_image_circle_stops_at_the_fold() -> None:
    """A lens that folds inside its field puts the fold, not the field, on the rim."""
    dist = torch.tensor([-0.3, 0.0, 0.0, 0.0])
    K = RenderAdapter.fisheye_K(dist, 180.0, True, 800, 600)
    d = dist.double().numpy()
    fold = fisheye_domain_angle(d)
    assert fold < math.pi / 2
    assert torch.isclose(K[1, 1] * float(fisheye_theta_d(np.array(fold), d)), torch.tensor(300.0))


def test_pixels_outside_the_fisheye_field_show_the_background() -> None:
    """A circular 180° fisheye blanks the corners to the background and keeps the centre."""
    viewer = _stub_viewer(CameraModelType.FISHEYE_EQUIDISTANT)
    viewer._pipeline.fill = 1.0
    viewer.render_settings.background_color = (0.0, 0.0, 1.0)
    rgb = RenderAdapter(viewer).render_at(c2w=np.eye(4), fov_rad=1.0, width=40, height=20, tau=0.0)
    assert rgb[0, 0].tolist() == [0, 0, 255]
    assert rgb[10, 20].tolist() == [255, 255, 255]


def test_plain_pinhole_is_not_masked() -> None:
    """A distortion-free pinhole observes every pixel, so nothing is blanked."""
    viewer = _stub_viewer(CameraModelType.PINHOLE)
    viewer._pipeline.fill = 1.0
    rgb = RenderAdapter(viewer).render_at(c2w=np.eye(4), fov_rad=1.0, width=40, height=20, tau=0.0)
    assert (rgb == 255).all()
