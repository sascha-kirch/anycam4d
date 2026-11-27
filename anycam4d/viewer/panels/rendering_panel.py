"""RenderingPanel — Viewer Res, Background, Render Mode, Colormap controls.

Built inside the nerfview rendering tab folder. Writes the live :class:`RenderSettings`
(read by the render adapter) and the render-tab viewer resolution, and triggers a re-render
on change. Replaces ``Anycam4DViewer._setup_render_controls_in_tab``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    import viser

    from anycam4d.viewer.nerfview_adapter import RenderTabView
    from anycam4d.viewer.panels.render_settings import RenderSettings


# Readable dropdown labels mapped to the RenderSettings.temporal_color_mode values.
_TEMPORAL_COLOR_LABELS = {
    "Mean time (μ_t)": "mean",
    "Distance |τ−μ_t|": "distance",  # noqa: RUF001
    "Std (σ_t)": "std",  # noqa: RUF001
}
_MEAN_LABEL = "Mean time (μ_t)"


class RenderingPanel:
    """Builds the render-settings controls inside the rendering tab.

    Args:
        server: Viser server.
        folder: The rendering tab folder to populate.
        settings: Render settings holder these controls write.
        render_tab: Typed view of the nerfview render-tab state (viewer resolution).
        rerender: Callback that requests a re-render (the nerfview ``rerender`` hook).
    """

    def __init__(
        self,
        server: viser.ViserServer,
        folder: viser.GuiFolderHandle,
        settings: RenderSettings,
        render_tab: RenderTabView,
        rerender: Callable[[object], None],
    ) -> None:
        """Build and wire the render-settings controls."""
        self._settings = settings
        self._render_tab = render_tab
        self._rerender = rerender

        with folder:
            self.viewer_res_slider = server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=2048,
                hint="Maximum resolution of the viewer rendered image.",
            )
            self.bg_rgb = server.gui.add_rgb(
                "Background",
                initial_value=(0, 0, 0),
                hint="Background color (shown where alpha < 1).",
            )
            self.mode_dd = server.gui.add_dropdown(
                "Render Mode",
                ("rgb", "alpha", "temporal"),
                initial_value="rgb",
                hint="RGB: color. Alpha: opacity. Temporal: Gaussians colored by mu_t.",
            )
            self.cmap_dd = server.gui.add_dropdown(
                "Colormap",
                ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                initial_value="turbo",
                disabled=True,
                hint="Colormap for alpha/temporal visualization.",
            )
            self.temporal_color_dd = server.gui.add_dropdown(
                "Temporal Color By",
                tuple(_TEMPORAL_COLOR_LABELS),
                initial_value=_MEAN_LABEL,
                disabled=True,
                hint="What temporal mode colors by: mean time, distance from the current time, or temporal spread.",
            )

        self.viewer_res_slider.on_update(self._on_viewer_res)
        self.bg_rgb.on_update(self._on_background)
        self.mode_dd.on_update(self._on_mode)
        self.cmap_dd.on_update(self._on_colormap)
        self.temporal_color_dd.on_update(self._on_temporal_color)

    def _on_viewer_res(self, event: viser.GuiEvent) -> None:
        """Push the viewer resolution into the render-tab state and re-render."""
        self._render_tab.viewer_res = int(self.viewer_res_slider.value)
        self._rerender(event)

    def _on_background(self, event: viser.GuiEvent) -> None:
        """Normalize the background color to [0, 1] and re-render."""
        r, g, b = self.bg_rgb.value
        self._settings.background_color = (r / 255.0, g / 255.0, b / 255.0)
        self._rerender(event)

    def _on_mode(self, event: viser.GuiEvent) -> None:
        """Update the render mode, gate the colormap / temporal-color controls, re-render."""
        self._settings.render_mode = self.mode_dd.value
        self.cmap_dd.disabled = self.mode_dd.value == "rgb"
        self.temporal_color_dd.disabled = self.mode_dd.value != "temporal"
        self._rerender(event)

    def _on_colormap(self, event: viser.GuiEvent) -> None:
        """Update the colormap and re-render."""
        self._settings.colormap = self.cmap_dd.value
        self._rerender(event)

    def _on_temporal_color(self, event: viser.GuiEvent) -> None:
        """Update the temporal color sub-mode and re-render."""
        self._settings.temporal_color_mode = _TEMPORAL_COLOR_LABELS[self.temporal_color_dd.value]
        self._rerender(event)
