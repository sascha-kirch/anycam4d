"""Anycam4DViewer: interactive viewer for trained 4D Gaussian splatting models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import viser
import viser.transforms as tf
from nerfview import Viewer
from nerfview import apply_float_colormap

from anycam4d.viewer.fair_lock import FairLock
from anycam4d.viewer.nerfview_adapter import RenderTabView
from anycam4d.viewer.nerfview_adapter import install_lock
from anycam4d.viewer.nerfview_adapter import new_render_tab_state
from anycam4d.viewer.panels.camera_path_panel import CameraPathPanel
from anycam4d.viewer.panels.info_panel import InfoPanel
from anycam4d.viewer.panels.lens_panel import LensPanel
from anycam4d.viewer.panels.render_settings import RenderSettings
from anycam4d.viewer.panels.rendering_panel import RenderingPanel
from anycam4d.viewer.panels.temporal_panel import TemporalPanel
from anycam4d.viewer.render_adapter import RenderAdapter
from anycam4d.viewer.temporal_controller import TemporalController


# Update the rendered-primitive count every Nth frame so the common per-frame path
# never forces a CUDA sync (the count reduction is the only sync it would incur).
_RENDERED_COUNT_THROTTLE = 10


if TYPE_CHECKING:
    from jaxtyping import Float
    from jaxtyping import UInt8
    from nerfview import CameraState
    from nerfview import RenderTabState
    from torch import Tensor

    from anycam4d.primitives.fourdgaussians.model import FourDGaussians
    from anycam4d.renderer.backend import RenderOutput
    from anycam4d.renderer.pipeline import RenderPipeline


class Anycam4DViewer(Viewer):
    """nerfview viewer with 4D camera paths, temporal playback and camera-model panels.

    Args:
        server: Viser server.
        model: Trained model on the target device.
        pipeline: Render pipeline.
        num_timesteps: Discrete timesteps on the slider.
        fps: Target playback rate.
        T_denorm: Optional ``[4, 4]`` normalized-to-world transform.
        update_every_n_steps: How often training pushes model updates.
        output_dir: Directory for trajectories and videos.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: FourDGaussians,
        pipeline: RenderPipeline,
        *,
        num_timesteps: int = 100,
        fps: float = 30.0,
        T_denorm: Float[Tensor, "4 4"] | None = None,
        update_every_n_steps: int = 50,
        output_dir: Path | None = None,
    ) -> None:
        """Set every attribute the rendering-tab overrides read before ``super().__init__()`` calls them."""
        self._model = model
        self._pipeline = pipeline
        self.update_every_n_steps = update_every_n_steps
        self.num_timesteps = num_timesteps
        self.fps = fps
        self._T_denorm: Float[Tensor, "4 4"] | None = T_denorm
        self._output_dir = output_dir or Path("./results")

        # Built before super().__init__: the render thread reads τ as soon as it starts.
        self._temporal = TemporalController(
            fps=fps,
            num_timesteps=num_timesteps,
            rerender=lambda: self.rerender(None),
        )
        self._frame_counter: int = 0

        # Single source of truth for the live render settings the adapter reads; the
        # rendering / UT-camera panels write it.
        self.render_settings = RenderSettings()

        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

        # Adapter owns the shared camera math + per-frame allocation caches. Built
        # before super().__init__ so render_fn can delegate to it.
        self._render_adapter = RenderAdapter(self)

        # ── super().__init__ calls _init_rendering_tab / _populate_rendering_tab ─
        super().__init__(
            server=server,
            render_fn=self._render_fn,
            output_dir=self._output_dir,
            mode="rendering",
        )

        install_lock(self, FairLock())
        self._info_panel = InfoPanel(server, len(model))

    @property
    def temporal(self) -> TemporalController:
        """The single owner of live τ / playback (panels wire callbacks to it)."""
        return self._temporal

    def _init_rendering_tab(self) -> None:
        """Build the panels top to bottom: Time, Lens, Display; ``render_tab_state`` is the name nerfview reads."""
        self.render_tab_state = new_render_tab_state()
        self.render_tab = RenderTabView(self.render_tab_state)
        reset_up = self.server.gui.add_button(
            "Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            hint="Orbit around the camera's current up, e.g. after the scene loads tilted.",
        )
        reset_up.on_click(_reset_up)
        self._temporal_panel = TemporalPanel(self.server, self._temporal, self.num_timesteps)
        self._lens_panel = LensPanel(self.server, self.render_settings, self.rerender)
        self._rendering_folder = self.server.gui.add_folder("Display")

    def _populate_rendering_tab(self) -> None:
        """Fill Display, then the collapsed Camera Path folder."""
        self._rendering_panel = RenderingPanel(
            self.server,
            self._rendering_folder,
            self.render_settings,
            self.render_tab,
            self.rerender,
        )
        self._camera_path_panel = CameraPathPanel(
            self.server,
            self.server.gui.add_folder("Camera Path", expand_by_default=False),
            self.render_tab,
            self,
            self._output_dir,
        )

    def _apply_lens(self, device: torch.device) -> None:
        """Apply the Lens controls into the render settings."""
        self._lens_panel.apply(device)

    def _render_fn(
        self,
        camera_state: CameraState,
        render_tab_state: RenderTabState,
    ) -> UInt8[np.ndarray, "H W 3"]:
        """Render callback for nerfview — delegates to the render adapter."""
        return self._render_adapter.render_fn(camera_state, render_tab_state)

    def _update_rendered_count(self, meta: dict) -> None:
        """Update the rendered-primitive count every Nth frame; it is the only per-frame device sync."""
        self._frame_counter += 1
        if self._frame_counter % _RENDERED_COUNT_THROTTLE != 0:
            return
        self._info_panel.rendered_primitives.value = self._count_rendered(meta)

    def _count_rendered(self, meta: dict) -> int:
        """Rendered Gaussians from ``gaussian_ids`` (sync-free) or ``radii`` (device reduction)."""
        if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
            return int(meta["gaussian_ids"].numel())
        if "radii" in meta and meta["radii"] is not None:
            return int((meta["radii"] > 0).any(dim=0).sum().item())
        return 0

    def _postprocess(self, out: RenderOutput) -> UInt8[np.ndarray, "H W 3"]:
        """Convert RenderOutput to [H, W, 3] float32 numpy based on render mode."""
        if self.render_settings.render_mode == "alpha":
            alpha = out.alphas[0]
            return apply_float_colormap(alpha, self.render_settings.colormap).cpu().numpy()
        return out.colors[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()

    def render_frame(
        self,
        c2w: Float[np.ndarray, "4 4"],
        fov_rad: float,
        width: int,
        height: int,
        timestamp: float,
    ) -> UInt8[np.ndarray, "H W 3"]:
        """Render one ``[H, W, 3]`` uint8 export frame through the adapter, with the active camera model."""
        return self._render_adapter.render_at(c2w, fov_rad, width, height, timestamp)

    def update_model(self, model: FourDGaussians) -> None:
        """Swap the model; call from the training thread while holding ``self.lock``."""
        self._model = model
        self._render_adapter.on_model_update()
        self._info_panel.total_primitives.value = len(model)


def _reset_up(event: viser.GuiEvent) -> None:
    """Make the requesting client's current camera up the orbit controls' up."""
    if event.client is not None:
        event.client.camera.up_direction = tf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
