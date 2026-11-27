"""LensPanel: camera model, field of view and distortion controls; writes :class:`RenderSettings` on apply."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from anycam4d.camera.registry import distortion_width
from anycam4d.enums import CameraModelType


if TYPE_CHECKING:
    from collections.abc import Callable

    import viser

    from anycam4d.viewer.panels.render_settings import RenderSettings


MODEL_LABELS: dict[str, CameraModelType] = {
    "Pinhole": CameraModelType.PINHOLE,
    "Pinhole + OpenCV distortion": CameraModelType.OPENCV,
    "Fisheye (equidistant)": CameraModelType.FISHEYE_EQUIDISTANT,
    "Fisheye + Kannala-Brandt": CameraModelType.FISHEYE_OPENCV,
}

# (name, canonical slot, slider half-range). Ranges shrink with the power of the term so every slider spans a
# visible warp without most of its travel folding the lens.
COEFFICIENTS: dict[CameraModelType, tuple[tuple[str, int, float], ...]] = {
    CameraModelType.OPENCV: (("k1", 0, 0.5), ("k2", 1, 0.2), ("p1", 2, 0.02), ("p2", 3, 0.02), ("k3", 4, 0.1)),
    CameraModelType.FISHEYE_OPENCV: (("k1", 0, 0.5), ("k2", 1, 0.2), ("k3", 2, 0.05), ("k4", 3, 0.01)),
}

_FISHEYE = (CameraModelType.FISHEYE_EQUIDISTANT, CameraModelType.FISHEYE_OPENCV)
_SLIDER_STEPS = 500


class LensPanel:
    """Builds the Lens folder and applies it into the render settings.

    Args:
        server: Viser server.
        settings: Render settings holder this panel writes on :meth:`apply`.
        rerender: Callback that requests a re-render on control change.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        settings: RenderSettings,
        rerender: Callable[[object], None],
    ) -> None:
        """Build the Lens folder and wire its callbacks."""
        self._server = server
        self._settings = settings
        self._rerender = rerender

        with server.gui.add_folder("Lens"):
            self.model_dd = server.gui.add_dropdown("Model", tuple(MODEL_LABELS), initial_value="Pinhole")
            self.fov_slider = server.gui.add_slider(
                "FOV (°)", min=10.0, max=150.0, step=1.0, initial_value=60.0, hint="Vertical field of view."
            )
            self.field_slider = server.gui.add_slider(
                "Lens Field (°)",
                min=60.0,
                max=180.0,
                step=1.0,
                initial_value=180.0,
                visible=False,
                hint="Angle the fisheye images; pixels outside it are not observed and show the background.",
            )
            self.circular_cb = server.gui.add_checkbox(
                "Circular Image",
                initial_value=True,
                visible=False,
                hint="On: the lens field fits inside the frame (circular fisheye). Off: it spans the diagonal.",
            )
            self.coeff_sliders: dict[CameraModelType, list[viser.GuiSliderHandle]] = {
                model: [
                    server.gui.add_slider(
                        name,
                        min=-half,
                        max=half,
                        step=2 * half / _SLIDER_STEPS,
                        initial_value=0.0,
                        marks=(-half, 0.0, half),
                        visible=False,
                    )
                    for name, _slot, half in coeffs
                ]
                for model, coeffs in COEFFICIENTS.items()
            }
            self.reset_btn = server.gui.add_button("Reset Distortion", visible=False)

        self.model_dd.on_update(self._on_model)
        self.fov_slider.on_update(self._on_fov)
        self.field_slider.on_update(rerender)
        self.circular_cb.on_update(rerender)
        for sliders in self.coeff_sliders.values():
            for slider in sliders:
                slider.on_update(rerender)
        self.reset_btn.on_click(self._on_reset)
        server.on_client_connect(self._push_fov)

    @property
    def model(self) -> CameraModelType:
        """The selected camera model."""
        return MODEL_LABELS[self.model_dd.value]

    def _on_model(self, event: viser.GuiEvent) -> None:
        """Show the controls the selected model reads and re-render."""
        model = self.model
        fisheye = model in _FISHEYE
        self.fov_slider.visible = not fisheye
        self.field_slider.visible = fisheye
        self.circular_cb.visible = fisheye
        for owner, sliders in self.coeff_sliders.items():
            for slider in sliders:
                slider.visible = owner == model
        self.reset_btn.visible = model in COEFFICIENTS
        self._rerender(event)

    def _push_fov(self, client: viser.ClientHandle) -> None:
        """Pinhole FOV is the browser camera's, so the scene overlays match the render."""
        client.camera.fov = math.radians(self.fov_slider.value)

    def _on_fov(self, _event: viser.GuiEvent) -> None:
        for client in self._server.get_clients().values():
            self._push_fov(client)

    def _on_reset(self, event: viser.GuiEvent) -> None:
        for slider in self.coeff_sliders.get(self.model, ()):
            slider.value = 0.0
        self._rerender(event)

    def apply(self, device: torch.device) -> None:
        """Write the model, its canonical distortion vector and the fisheye field into the render settings."""
        model = self.model
        settings = self._settings
        settings.active_camera_model = model
        settings.fisheye_field_deg = float(self.field_slider.value)
        settings.fisheye_circular = bool(self.circular_cb.value)
        if model is CameraModelType.PINHOLE:
            settings.active_distortion_params = None
            return
        dist = [0.0] * distortion_width(model)
        for (_name, slot, _half), slider in zip(COEFFICIENTS.get(model, ()), self.coeff_sliders.get(model, ())):
            dist[slot] = float(slider.value)
        settings.active_distortion_params = torch.tensor(dist, dtype=torch.float32, device=device)
