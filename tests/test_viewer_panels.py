"""GUI panels: typed handles and callback wiring (skipped without a viser server)."""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch

from anycam4d.enums import CameraModelType

from anycam4d.viewer.nerfview_adapter import RenderTabView
from anycam4d.viewer.panels.camera_path_panel import CameraPathPanel
from anycam4d.viewer.panels.handles import CameraPathHandles
from anycam4d.viewer.panels.handles import ExportHandles
from anycam4d.viewer.panels.handles import PreviewHandles
from anycam4d.viewer.panels.handles import handle_values
from anycam4d.viewer.panels.lens_panel import LensPanel
from anycam4d.viewer.panels.render_settings import RenderSettings
from anycam4d.viewer.panels.rendering_panel import RenderingPanel


@pytest.fixture()
def server():  # noqa: ANN201
    """A bare viser server, or skip if one cannot start."""
    viser = pytest.importorskip("viser")
    try:
        srv = viser.ViserServer(host="127.0.0.1", port=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"viser server unavailable: {exc}")
    yield srv
    srv.stop()


class _StubTemporal:
    """Temporal controller stand-in recording preview playback calls."""

    num_timesteps = 100

    def __init__(self) -> None:
        self.played = False
        self.stopped = False

    def play_preview(self, _tick, _period) -> None:  # noqa: ANN001
        self.played = True

    def stop_preview(self) -> None:
        self.stopped = True


def _stub_viewer() -> types.SimpleNamespace:
    """Viewer stand-in exposing the public surface CameraPathPanel reads."""
    return types.SimpleNamespace(
        fps=30.0,
        num_timesteps=100,
        temporal=_StubTemporal(),
        lock=None,
        render_frame=lambda **k: None,
    )


def _render_tab() -> RenderTabView:
    """A RenderTabView over a stub state carrying the fields the panel touches."""
    return RenderTabView(types.SimpleNamespace(viewer_res=2048, preview_time=0.0, preview_render=False))


def _build_panel(server, tmp_path) -> CameraPathPanel:  # noqa: ANN001
    """Construct a CameraPathPanel on a real server with a stub viewer."""
    folder = server.gui.add_folder("Rendering")
    return CameraPathPanel(server, folder, _render_tab(), _stub_viewer(), tmp_path)


def test_panels_expose_typed_handles(server, tmp_path) -> None:  # noqa: ANN001
    """Each sub-section exposes its handles as a typed dataclass with every field populated."""
    panel = _build_panel(server, tmp_path)
    assert isinstance(panel._path, CameraPathHandles)  # noqa: SLF001
    assert isinstance(panel._preview, PreviewHandles)  # noqa: SLF001
    assert isinstance(panel._export, ExportHandles)  # noqa: SLF001
    for container in (panel._path, panel._preview, panel._export):  # noqa: SLF001
        for value in handle_values(container):
            assert value is not None


def test_all_controls_covers_every_handle(server, tmp_path) -> None:  # noqa: ANN001
    """_all_controls enumerates duration + every path/preview/export handle (for export-disable)."""
    panel = _build_panel(server, tmp_path)
    expected = (
        1  # duration
        + len(dataclasses.fields(CameraPathHandles))
        + len(dataclasses.fields(PreviewHandles))
        + len(dataclasses.fields(ExportHandles))
    )
    assert len(panel._all_controls()) == expected  # noqa: SLF001


def test_tension_callback_targets_the_model(server, tmp_path) -> None:  # noqa: ANN001
    """The tension callback writes the camera-path model, not a shared dict."""
    panel = _build_panel(server, tmp_path)
    panel._path.tension_slider.value = 0.8  # noqa: SLF001
    panel._on_tension_changed(None)  # noqa: SLF001
    assert panel._camera_path.tension == pytest.approx(0.8)  # noqa: SLF001


def test_preview_play_pause_target_the_controller(server, tmp_path) -> None:  # noqa: ANN001
    """Preview play / pause callbacks drive the temporal controller."""
    panel = _build_panel(server, tmp_path)
    panel._on_preview_play(None)  # noqa: SLF001
    assert panel._viewer.temporal.played  # noqa: SLF001
    panel._on_preview_pause(None)  # noqa: SLF001
    assert panel._viewer.temporal.stopped  # noqa: SLF001


def test_loop_mode_disables_manual_transition(server, tmp_path) -> None:  # noqa: ANN001
    """Switching to Loop mode disables manual τ editing on the camera path."""
    panel = _build_panel(server, tmp_path)
    panel._export.temporal_mode_dd.value = "Loop"  # noqa: SLF001
    panel._on_temporal_mode_changed(None)  # noqa: SLF001
    assert panel._camera_path.tau_editable is False  # noqa: SLF001


def test_rendering_panel_writes_settings(server) -> None:  # noqa: ANN001
    """RenderingPanel callbacks write the shared RenderSettings the adapter reads."""
    settings = RenderSettings()
    rerenders: list[object] = []
    folder = server.gui.add_folder("Rendering")
    panel = RenderingPanel(server, folder, settings, _render_tab(), rerenders.append)

    # Setting a handle value fires its registered on_update; the panel writes the holder.
    panel.mode_dd.value = "alpha"
    assert settings.render_mode == "alpha"
    assert panel.cmap_dd.disabled is False  # colormap enabled outside RGB

    panel.bg_rgb.value = (255, 0, 0)
    assert settings.background_color == (1.0, 0.0, 0.0)
    assert len(rerenders) == 2  # one per callback


def test_rendering_panel_temporal_color_dropdown(server) -> None:  # noqa: ANN001
    """The Temporal Color By dropdown writes the mode and is gated to temporal render mode."""
    settings = RenderSettings()
    folder = server.gui.add_folder("Rendering")
    panel = RenderingPanel(server, folder, settings, _render_tab(), lambda _e: None)

    # Disabled outside temporal mode; enabled only in temporal mode.
    assert panel.temporal_color_dd.disabled is True
    panel.mode_dd.value = "temporal"
    assert panel.temporal_color_dd.disabled is False
    panel.mode_dd.value = "alpha"
    assert panel.temporal_color_dd.disabled is True

    # Selecting a readable label writes the RenderSettings mode value.
    panel.temporal_color_dd.value = "Distance |τ−μ_t|"
    assert settings.temporal_color_mode == "distance"
    panel.temporal_color_dd.value = "Std (σ_t)"
    assert settings.temporal_color_mode == "std"


def test_lens_panel_writes_the_selected_model_and_its_coefficients(server) -> None:  # noqa: ANN001
    """Only the selected model's coefficients are shown, and they land in their canonical slots."""
    settings = RenderSettings()
    panel = LensPanel(server, settings, lambda _e: None)
    assert not any(s.visible for sliders in panel.coeff_sliders.values() for s in sliders)

    panel.model_dd.value = "Pinhole + OpenCV distortion"
    p1 = panel.coeff_sliders[CameraModelType.OPENCV][2]
    assert p1.visible and not panel.coeff_sliders[CameraModelType.FISHEYE_OPENCV][0].visible
    p1.value = 0.01
    panel.apply(torch.device("cpu"))
    assert settings.active_camera_model is CameraModelType.OPENCV
    assert settings.active_distortion_params.shape == (12,)
    assert settings.active_distortion_params[2].item() == pytest.approx(0.01)

    panel.model_dd.value = "Pinhole"
    panel.apply(torch.device("cpu"))
    assert settings.active_distortion_params is None

    panel.model_dd.value = "Fisheye (equidistant)"
    panel.apply(torch.device("cpu"))
    assert settings.active_camera_model is CameraModelType.FISHEYE_EQUIDISTANT
    assert not settings.active_distortion_params.any()
    assert panel.field_slider.visible and not panel.fov_slider.visible
