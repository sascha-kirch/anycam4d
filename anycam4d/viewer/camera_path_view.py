"""Draws and edits a GUI-free :class:`CameraPathModel` in the viser scene; the panel's only facade to it."""

from __future__ import annotations

import colorsys
import contextlib
import time as time_module
from typing import TYPE_CHECKING

import numpy as np
import viser
import viser.transforms as tf

from anycam4d.viewer.camera_path import CameraPathModel
from anycam4d.viewer.constants import DEFAULT_FPS


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float

    from anycam4d.viewer.camera_path import InterpolatedFrame
    from anycam4d.viewer.camera_path import Keyframe4D


class CameraPathView:
    """Keyframe frustums, spline visualization and in-scene keyframe editor for a :class:`CameraPathModel`."""

    def __init__(self, server: viser.ViserServer, duration_element: viser.GuiInputHandle[float]) -> None:
        """Wrap a fresh model and subscribe to its rebuilds for spline redraws."""
        self._server = server
        self._duration_element = duration_element
        self._model = CameraPathModel()
        self._model.on_rebuilt = self._redraw_spline

        self._frustums: dict[int, viser.CameraFrustumHandle] = {}
        self._spline_nodes: list[viser.SceneNodeHandle] = []
        self._camera_edit_panel: viser.Gui3dContainerHandle | None = None
        self._keyframes_visible: bool = True
        self.show_spline: bool = True
        self.viz_n_frames: int | None = None

    @property
    def default_fov_rad(self) -> float:
        """Default field-of-view (radians) for new keyframes."""
        return self._model.default_fov_rad

    @default_fov_rad.setter
    def default_fov_rad(self, value: float) -> None:
        self._model.default_fov_rad = value

    @property
    def default_transition_sec(self) -> float:
        """Default per-transition time (seconds)."""
        return self._model.default_transition_sec

    @default_transition_sec.setter
    def default_transition_sec(self, value: float) -> None:
        self._model.default_transition_sec = value

    @property
    def default_timestamp(self) -> float:
        """Default τ for new keyframes."""
        return self._model.default_timestamp

    @default_timestamp.setter
    def default_timestamp(self, value: float) -> None:
        self._model.default_timestamp = value

    @property
    def tension(self) -> float:
        """Spline tension."""
        return self._model.tension

    @tension.setter
    def tension(self, value: float) -> None:
        self._model.tension = value

    @property
    def tau_editable(self) -> bool:
        """Whether the in-scene editor may edit per-keyframe τ."""
        return self._model.tau_editable

    @tau_editable.setter
    def tau_editable(self, value: bool) -> None:
        self._model.tau_editable = value

    @property
    def tau_value_fn(self) -> Callable[[int], float] | None:
        """Optional function giving the displayed τ per keyframe index (non-Manual modes)."""
        return self._model.tau_value_fn

    @tau_value_fn.setter
    def tau_value_fn(self, value: Callable[[int], float] | None) -> None:
        self._model.tau_value_fn = value

    def compute_duration(self) -> float:
        """Total path duration in seconds."""
        return self._model.compute_duration()

    def interpolate(self, normalized_t: float) -> InterpolatedFrame | None:
        """Evaluate the path at ``normalized_t`` ∈ [0, 1]."""
        return self._model.interpolate(normalized_t)

    def to_json(self) -> list[dict]:
        """Serialize the keyframes."""
        return self._model.to_json()

    def sorted_keyframe_indices(self) -> list[int]:
        """Keyframe indices in ascending order."""
        return self._model.sorted_keyframe_indices()

    def keyframe_handles(self) -> list[tuple[int, Keyframe4D, viser.CameraFrustumHandle]]:
        """Return ``(index, keyframe, frustum)`` for each keyframe (for transform-control binding)."""
        return [(idx, kf, self._frustums[idx]) for idx, kf in self._model.keyframes() if idx in self._frustums]

    def mark_spline_dirty(self) -> None:
        """Coalesce a deferred interpolator rebuild + spline redraw."""
        self._model.mark_spline_dirty()

    def update_spline(self) -> None:
        """Rebuild interpolators now and redraw the spline."""
        self._model.update_spline()

    def refresh_spline_visualization(self) -> None:
        """Redraw the spline visualization without rebuilding interpolators."""
        self._redraw_spline()

    def add_keyframe(self, keyframe: Keyframe4D, keyframe_index: int | None = None) -> None:
        """Store a keyframe in the model and create its frustum."""
        idx = self._model.add_keyframe(keyframe, keyframe_index)
        self._create_frustum(idx, keyframe)

    def _create_frustum(self, keyframe_index: int, keyframe: Keyframe4D) -> None:
        """Create the frustum + sphere scene nodes and the editor click handler."""
        frustum = self._server.scene.add_camera_frustum(
            f"/render_cameras/{keyframe_index}",
            fov=self._model.default_fov_rad,
            aspect=keyframe.aspect,
            scale=0.1,
            color=(200, 10, 30),
            wxyz=keyframe.wxyz,
            position=keyframe.position,
            visible=self._keyframes_visible,
        )
        self._server.scene.add_icosphere(
            f"/render_cameras/{keyframe_index}/sphere",
            radius=0.03,
            color=(200, 10, 30),
        )
        self._register_frustum_click(frustum, keyframe, keyframe_index)
        self._frustums[keyframe_index] = frustum

    def update_frustum(self, keyframe_index: int, keyframe: Keyframe4D) -> None:
        """Update a keyframe and its persistent frustum node in place, keeping its click handler; create it if unknown."""
        frustum = self._frustums.get(keyframe_index)
        if frustum is None:
            self.add_keyframe(keyframe, keyframe_index)
            return
        frustum.position = keyframe.position
        frustum.wxyz = keyframe.wxyz
        frustum.aspect = keyframe.aspect
        self._model.add_keyframe(keyframe, keyframe_index)

    def set_keyframes_visible(self, visible: bool) -> None:
        """Show or hide every keyframe frustum."""
        self._keyframes_visible = visible
        for frustum in self._frustums.values():
            frustum.visible = visible

    def update_aspect(self, aspect: float) -> None:
        """Set every keyframe's aspect and redraw the frustums."""
        for idx, kf in self._model.keyframes():
            kf.aspect = aspect
            self.update_frustum(idx, kf)

    def clear(self) -> None:
        """Remove all keyframes, frustums, the editor, and spline visualization."""
        self._remove_all_frustums()
        self._model.clear()
        self._close_editor()
        self.update_spline()

    def _remove_all_frustums(self) -> None:
        """Remove every keyframe frustum scene node."""
        for frustum in self._frustums.values():
            frustum.remove()
        self._frustums.clear()

    def apply_lookat(self, target: Float[np.ndarray, "3"]) -> None:
        """Turn every keyframe towards ``target`` and refresh the frustums."""
        for idx in self._model.apply_lookat(target):
            kf = self._model.get(idx)
            if kf is not None:
                self.update_frustum(idx, kf)
        self._duration_element.value = self._model.compute_duration()
        self.update_spline()

    def load_keyframes(self, keyframe_dicts: list[dict]) -> None:
        """Replace the keyframes from :meth:`CameraPathModel.to_json` data and rebuild the scene."""
        self._remove_all_frustums()
        self._close_editor()
        self._model.load_keyframes(keyframe_dicts)
        for idx, kf in self._model.keyframes():
            self._create_frustum(idx, kf)
        self._duration_element.value = self._model.compute_duration()
        self.update_spline()

    def _redraw_spline(self) -> None:
        """Redraw the spline polyline + point cloud from the model's sampled positions."""
        for node in reversed(self._spline_nodes):
            with contextlib.suppress(KeyError):
                node.remove()
        self._spline_nodes.clear()

        if not self.show_spline:
            return
        duration = self._model.compute_duration()
        num_frames = max(2, self.viz_n_frames if self.viz_n_frames is not None else int(duration * DEFAULT_FPS))
        points = self._model.sample_path_positions(num_frames)
        if points is None:
            return

        colors = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0) for h in np.linspace(0.0, 1.0, len(points))])
        self._spline_nodes.append(
            self._server.scene.add_spline_catmull_rom(
                "/render_camera_spline",
                positions=points,
                color=(220, 220, 220),
                closed=False,
                line_width=1.0,
                segments=len(points) + 1,
            )
        )
        self._spline_nodes.append(
            self._server.scene.add_point_cloud(
                "/render_camera_spline/points",
                points=points,
                colors=colors,
                point_size=0.04,
            )
        )

    def _register_frustum_click(
        self,
        frustum: viser.CameraFrustumHandle,
        keyframe: Keyframe4D,
        keyframe_index: int,
    ) -> None:
        """Attach a click handler that opens the keyframe editor panel."""

        @frustum.on_click
        def _(_: viser.SceneNodePointerEvent) -> None:
            self._close_editor()
            self._open_keyframe_editor(keyframe, keyframe_index)

    def _close_editor(self) -> None:
        """Close the currently open keyframe editor panel."""
        if self._camera_edit_panel is not None:
            self._camera_edit_panel.remove()
            self._camera_edit_panel = None

    def _open_keyframe_editor(self, keyframe: Keyframe4D, keyframe_index: int) -> None:
        """Open a 3D panel at the keyframe for editing it in place."""
        server = self._server
        with server.scene.add_3d_gui_container("/camera_edit_panel", position=keyframe.position) as panel:
            self._camera_edit_panel = panel

            display_tau = (
                self._model.tau_value_fn(keyframe_index)
                if (not self._model.tau_editable and self._model.tau_value_fn is not None)
                else keyframe.timestamp
            )
            tau_slider = server.gui.add_slider(
                "τ (timestamp)",
                min=-0.5,
                max=0.5,
                step=0.001,
                initial_value=display_tau,
                disabled=not self._model.tau_editable,
            )

            @tau_slider.on_update
            def _(_: viser.GuiEvent) -> None:
                if self._model.tau_editable:
                    keyframe.timestamp = tau_slider.value
                    self._refresh_keyframe(keyframe, keyframe_index)

            server.gui.add_number(
                "FOV (degrees)",
                initial_value=round(self._model.default_fov_rad * 180.0 / np.pi, 1),
                disabled=True,
            )

            self._create_transition_controls(server, keyframe, keyframe_index)
            self._create_action_buttons(server, keyframe, keyframe_index, panel)

    def _create_transition_controls(
        self,
        server: viser.ViserServer,
        keyframe: Keyframe4D,
        keyframe_index: int,
    ) -> None:
        """Create transition time override controls in the keyframe editor."""
        override_trans = server.gui.add_checkbox(
            "Override Transition",
            initial_value=keyframe.override_transition_enabled,
            disabled=not self._model.tau_editable,
        )
        trans_num = server.gui.add_number(
            "Transition (sec)",
            initial_value=keyframe.override_transition_sec or self._model.default_transition_sec,
            min=0.001,
            max=30.0,
            step=0.01,
            disabled=not keyframe.override_transition_enabled or not self._model.tau_editable,
        )

        @override_trans.on_update
        def _(_: viser.GuiEvent) -> None:
            keyframe.override_transition_enabled = override_trans.value
            trans_num.disabled = not override_trans.value
            self._refresh_keyframe(keyframe, keyframe_index)

        @trans_num.on_update
        def _(_: viser.GuiEvent) -> None:
            keyframe.override_transition_sec = trans_num.value
            self._refresh_keyframe(keyframe, keyframe_index)

    def _create_action_buttons(
        self,
        server: viser.ViserServer,
        keyframe: Keyframe4D,
        keyframe_index: int,
        panel: viser.Gui3dContainerHandle,
    ) -> None:
        """Create go-to, delete, and close buttons in the keyframe editor."""
        goto_btn = server.gui.add_button("Go to")
        delete_btn = server.gui.add_button("Delete", color="red", icon=viser.Icon.TRASH)
        close_btn = server.gui.add_button("Close")

        @goto_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self._animate_camera_to(event.client, keyframe)

        @delete_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                self._confirm_delete(event.client, keyframe_index, panel)

        close_btn.on_click(lambda _event: self._close_editor())

    def _animate_camera_to(self, client: viser.ClientHandle, keyframe: Keyframe4D) -> None:
        """Smoothly animate the client camera to a keyframe pose."""
        t_current = tf.SE3.from_rotation_and_translation(tf.SO3(client.camera.wxyz), client.camera.position)
        t_target = tf.SE3.from_rotation_and_translation(
            tf.SO3(keyframe.wxyz), keyframe.position
        ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))
        t_delta = t_current.inverse() @ t_target
        for j in range(10):
            t_set = t_current @ tf.SE3.exp(t_delta.log() * j / 9.0)
            with client.atomic():
                client.camera.wxyz = t_set.rotation().wxyz
                client.camera.position = t_set.translation()
            time_module.sleep(1.0 / 30.0)

    def _confirm_delete(
        self,
        client: viser.ClientHandle,
        keyframe_index: int,
        panel: viser.Gui3dContainerHandle,
    ) -> None:
        """Show a confirmation modal before deleting a keyframe."""
        with client.gui.add_modal("Confirm") as modal:
            client.gui.add_markdown("Delete keyframe?")
            yes_btn = client.gui.add_button("Yes", color="red", icon=viser.Icon.TRASH)
            cancel_btn = client.gui.add_button("Cancel")

            @yes_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._remove_keyframe(keyframe_index)
                panel.remove()
                self._camera_edit_panel = None
                modal.close()

            cancel_btn.on_click(lambda _event: modal.close())

    def _remove_keyframe(self, keyframe_index: int) -> None:
        """Remove a keyframe + its frustum, then recompute duration and rebuild the spline."""
        frustum = self._frustums.pop(keyframe_index, None)
        if frustum is not None:
            frustum.remove()
        self._model.remove_keyframe(keyframe_index)
        self._duration_element.value = self._model.compute_duration()
        self.update_spline()

    def _refresh_keyframe(self, keyframe: Keyframe4D, keyframe_index: int) -> None:
        """Mutate a keyframe's frustum in place and rebuild the spline."""
        self.update_frustum(keyframe_index, keyframe)
        self._duration_element.value = self._model.compute_duration()
        self.update_spline()
