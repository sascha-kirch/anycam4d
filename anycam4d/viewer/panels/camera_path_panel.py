"""Camera-path authoring, preview, and export panel.

Replaces the procedural ``render_panel.py`` closure soup with a single typed panel class.
The panel builds the Camera Path / Preview / Export sub-folders, owns their handles as typed
dataclasses (no untyped ``handles`` dict), holds the preview runtime state as attributes (no
late-populated ``_PreviewState``), and delegates video rendering to ``VideoExporter``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import viser

from anycam4d.console import notice
from anycam4d.console import warn
from anycam4d.viewer.camera_path import Keyframe4D
from anycam4d.viewer.camera_path_view import CameraPathView
from anycam4d.viewer.constants import DEFAULT_ASPECT
from anycam4d.viewer.constants import DEFAULT_FPS
from anycam4d.viewer.panels.handles import CameraPathHandles
from anycam4d.viewer.panels.handles import ExportHandles
from anycam4d.viewer.panels.handles import PreviewHandles
from anycam4d.viewer.panels.handles import handle_values
from anycam4d.viewer.temporal_schedule import TemporalSchedule
from anycam4d.viewer.video_exporter import VideoExporter


if TYPE_CHECKING:
    from jaxtyping import Float

    from anycam4d.viewer.nerfview_adapter import RenderTabView
    from anycam4d.viewer.viewer import Anycam4DViewer


class CameraPathPanel:
    """Builds and wires the 4D camera-path / preview / export GUI.

    Args:
        server: Viser server.
        folder: The rendering tab folder to populate.
        render_tab: Typed view of the nerfview render-tab state (preview flags, resolution).
        viewer: The 4DGS viewer (pipeline access, lock, render_frame, temporal controller).
        output_dir: Base directory for saved trajectories and rendered videos.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        folder: viser.GuiFolderHandle,
        render_tab: RenderTabView,
        viewer: Anycam4DViewer,
        output_dir: Path,
    ) -> None:
        """Build the full camera-path panel and wire all callbacks."""
        self._server = server
        self._folder = folder
        self._render_tab = render_tab
        self._viewer = viewer
        self._output_dir = output_dir

        # Preview / editor runtime state.
        self._preview_camera_handle: viser.SceneNodeHandle | None = None
        self._preview_folder: viser.GuiFolderHandle | None = None
        self._camera_pose_backup: dict[int, tuple] = {}
        self._transform_controls: list[viser.SceneNodeHandle] = []
        self._lookat_position: Float[np.ndarray, "3"] = np.array([0.0, 0.0, 0.0])
        self._lookat_frame_handle: viser.SceneNodeHandle | None = None
        self._lookat_control: viser.SceneNodeHandle | None = None

        self._build()

    def _build(self) -> None:
        """Create all GUI controls (in folder order) then wire events once all exist."""
        with self._folder:
            self._duration_number = self._server.gui.add_number(
                "Duration (sec)",
                initial_value=0.0,
                disabled=True,
                hint="Total path duration (transitions + holds). Computed automatically.",
            )

        self._camera_path = CameraPathView(self._server, self._duration_number)
        self._video_exporter = VideoExporter(self._server, self._folder, self._viewer)

        self._path = self._build_camera_path_gui()
        self._preview = self._build_preview_gui()
        self._export = self._build_export_gui()

        self._wire_camera_path_events()
        self._wire_preview_events()
        self._wire_export_events()

        # Initial derived state.
        self._update_tau_fn()
        self._update_viz_n_frames()
        self._update_frames_display()

    def _build_camera_path_gui(self) -> CameraPathHandles:
        """Create the Keyframes sub-folder controls."""
        with self._folder, self._server.gui.add_folder("Keyframes"):
            return CameraPathHandles(
                add_kf_btn=self._server.gui.add_button("Add Keyframe", icon=viser.Icon.PLUS),
                clear_kf_btn=self._server.gui.add_button("Clear Keyframes", icon=viser.Icon.TRASH),
                fov_slider=self._server.gui.add_slider("Default FOV", min=5.0, max=175.0, step=0.1, initial_value=50.0),
                default_tau=self._server.gui.add_slider(
                    "Default τ",
                    min=-0.5,
                    max=0.5,
                    step=0.001,
                    initial_value=-0.5,
                    hint="Default timestamp for new keyframes.",
                ),
                trans_num=self._server.gui.add_number(
                    "Default Transition (sec)", initial_value=2.0, min=0.001, max=30.0, step=0.01
                ),
                tension_slider=self._server.gui.add_slider(
                    "Spline Tension", min=0.0, max=1.0, step=0.01, initial_value=0.5
                ),
                move_cb=self._server.gui.add_checkbox("Move Keyframes", initial_value=False),
                show_kf_cb=self._server.gui.add_checkbox("Show Keyframes", initial_value=True),
                show_spline_cb=self._server.gui.add_checkbox("Show Spline", initial_value=True),
                show_lookat_cb=self._server.gui.add_checkbox("Show Look-At Target", initial_value=False),
                apply_lookat_btn=self._server.gui.add_button(
                    "Apply Look-At to All",
                    disabled=True,
                    hint="Reorient all keyframe cameras to face the look-at target.",
                ),
            )

    def _build_preview_gui(self) -> PreviewHandles:
        """Create the Preview sub-folder controls."""
        # Sized from the default schedule: the Export folder (which _schedule reads) is built
        # after this one to keep the GUI order Keyframes → Preview → Export, and with no
        # keyframes yet duration is 0 so this resolves to a single, disabled frame regardless.
        max_frame = TemporalSchedule(
            mode="Manual",
            fps=self._viewer.fps,
            duration=self._duration_number.value,
            num_timesteps=self._viewer.num_timesteps,
        ).max_preview_frame()
        with self._folder:
            self._preview_folder = self._server.gui.add_folder("Preview")
            with self._preview_folder:
                return PreviewHandles(
                    frame_slider=self._server.gui.add_slider(
                        "Preview Frame", min=0, max=max_frame, step=1, initial_value=0, disabled=max_frame <= 1
                    ),
                    play_btn=self._server.gui.add_button(
                        "Play Path", icon=viser.Icon.ROUTE, hint="Move the preview camera along the path."
                    ),
                    pause_btn=self._server.gui.add_button("Pause Path", icon=viser.Icon.PLAYER_PAUSE, visible=False),
                    preview_render_btn=self._server.gui.add_button("Look Through Path", icon=viser.Icon.EYE),
                    exit_preview_btn=self._server.gui.add_button("Exit Path View", color="red", visible=False),
                )

    def _build_export_gui(self) -> ExportHandles:
        """Create the Export sub-folder controls."""
        with self._folder, self._server.gui.add_folder("Export"):
            return ExportHandles(
                render_res=self._server.gui.add_vector2(
                    "Render Resolution", initial_value=(1280, 720), min=(64, 64), max=(4096, 4096), step=1
                ),
                fps_number=self._server.gui.add_number(
                    "FPS", initial_value=self._viewer.fps, min=1.0, max=240.0, step=1.0
                ),
                temporal_mode_dd=self._server.gui.add_dropdown(
                    "Temporal Mode",
                    options=("Manual", "Loop", "Freeze"),
                    initial_value="Manual",
                    hint="Manual: use per-keyframe τ. Loop: auto-sweep τ N times. Freeze: constant τ.",
                ),
                loop_count_num=self._server.gui.add_number(
                    "Loop Count",
                    initial_value=1.0,
                    min=0.25,
                    max=100.0,
                    step=0.25,
                    hint="Number of full τ sweeps (-0.5→0.5) over the video duration.",
                    visible=False,
                ),
                ping_pong_cb=self._server.gui.add_checkbox(
                    "Ping-Pong",
                    initial_value=False,
                    hint="Alternate sweep direction on each loop instead of jumping back to start.",
                    visible=False,
                ),
                global_tau_sl=self._server.gui.add_slider(
                    "Global τ",
                    min=-0.5,
                    max=0.5,
                    step=0.001,
                    initial_value=-0.5,
                    hint="Fixed timestamp applied to every exported frame (Freeze mode).",
                    visible=False,
                ),
                output_dir_text=self._server.gui.add_text(
                    "Output Directory",
                    initial_value=str(self._output_dir),
                    hint="Base directory for saved trajectories and rendered videos.",
                ),
                traj_name=self._server.gui.add_text("Trajectory Name", initial_value="default"),
                video_name=self._server.gui.add_text("Video Filename", initial_value="render"),
                save_btn=self._server.gui.add_button("Save Trajectory", icon=viser.Icon.FILE_EXPORT),
                load_btn=self._server.gui.add_button("Load Trajectory", icon=viser.Icon.FOLDER_OPEN),
                render_btn=self._server.gui.add_button("Render Video", color="green", icon=viser.Icon.VIDEO),
                total_frames_display=self._server.gui.add_number(
                    "Total Frames",
                    initial_value=0.0,
                    disabled=True,
                    hint="Number of frames to render (Loop: loop_count x num_timesteps; else fps x duration).",
                ),
            )

    def _wire_camera_path_events(self) -> None:
        """Bind add/clear/FOV/look-at and spline/visibility/movement callbacks."""
        path = self._path
        path.add_kf_btn.on_click(self._on_add_keyframe)
        path.clear_kf_btn.on_click(self._on_clear_keyframes)
        path.fov_slider.on_update(self._on_fov_changed)
        path.show_lookat_cb.on_update(self._on_show_lookat)
        path.apply_lookat_btn.on_click(lambda _event: self._camera_path.apply_lookat(self._lookat_position))
        path.trans_num.on_update(self._on_transition_changed)
        path.tension_slider.on_update(self._on_tension_changed)
        path.move_cb.on_update(self._on_move_toggled)
        path.show_kf_cb.on_update(lambda _event: self._camera_path.set_keyframes_visible(path.show_kf_cb.value))
        path.show_spline_cb.on_update(self._on_show_spline)

    def _aspect(self) -> float:
        """Current render aspect ratio from the export resolution."""
        res = self._export.render_res.value
        return res[0] / res[1] if res[1] else DEFAULT_ASPECT

    def _on_add_keyframe(self, event: viser.GuiEvent) -> None:
        """Append a keyframe at the requesting client's current camera pose."""
        if event.client_id is None:
            return
        camera = self._server.get_clients()[event.client_id].camera
        self._camera_path.add_keyframe(
            Keyframe4D.from_camera(camera, aspect=self._aspect(), timestamp=self._path.default_tau.value)
        )
        self._duration_number.value = self._camera_path.compute_duration()
        self._camera_path.update_spline()

    def _on_clear_keyframes(self, event: viser.GuiEvent) -> None:
        """Show a confirmation modal, then clear all keyframes and transform gizmos."""
        if event.client_id is None:
            return
        client = self._server.get_clients()[event.client_id]
        with client.atomic(), client.gui.add_modal("Confirm") as modal:
            client.gui.add_markdown("Clear all keyframes?")
            yes_btn = client.gui.add_button("Yes", color="red", icon=viser.Icon.TRASH)
            cancel_btn = client.gui.add_button("Cancel")

            @yes_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._camera_path.clear()
                self._duration_number.value = 0.0
                self._clear_transform_controls()
                modal.close()

            cancel_btn.on_click(lambda _event: modal.close())

    def _on_fov_changed(self, _event: viser.GuiEvent) -> None:
        """Apply the default FOV to all clients and the path, refreshing keyframe aspect."""
        fov_rad = self._path.fov_slider.value / 180.0 * np.pi
        for client in self._server.get_clients().values():
            client.camera.fov = fov_rad
        self._camera_path.default_fov_rad = fov_rad
        self._camera_path.update_aspect(self._aspect())

    def _on_show_lookat(self, event: viser.GuiEvent) -> None:
        """Toggle the look-at target frame and its transform control."""
        if self._path.show_lookat_cb.value:
            self._add_lookat_target(event)
        else:
            self._remove_lookat_target()

    def _add_lookat_target(self, event: viser.GuiEvent) -> None:
        """Create the look-at frame and (client-local) transform control."""
        self._lookat_frame_handle = self._server.scene.add_frame(
            "/lookat_target", axes_length=0.15, axes_radius=0.005, position=self._lookat_position
        )
        self._path.apply_lookat_btn.disabled = False
        if event.client is None:
            return
        ctrl = event.client.scene.add_transform_controls("/lookat_control", scale=0.3, position=self._lookat_position)
        self._lookat_control = ctrl

        @ctrl.on_update
        def _(_: viser.TransformControlsHandle) -> None:
            self._lookat_position = ctrl.position
            if self._lookat_frame_handle is not None:
                self._lookat_frame_handle.position = self._lookat_position

    def _remove_lookat_target(self) -> None:
        """Remove the look-at frame and transform control."""
        if self._lookat_frame_handle is not None:
            self._lookat_frame_handle.remove()
            self._lookat_frame_handle = None
        if self._lookat_control is not None:
            self._lookat_control.remove()
            self._lookat_control = None
        self._path.apply_lookat_btn.disabled = True

    def _on_transition_changed(self, _event: viser.GuiEvent) -> None:
        """Update the default transition time and recompute duration."""
        self._camera_path.default_transition_sec = self._path.trans_num.value
        self._duration_number.value = self._camera_path.compute_duration()

    def _on_tension_changed(self, _event: viser.GuiEvent) -> None:
        """Update spline tension and schedule a coalesced rebuild."""
        self._camera_path.tension = self._path.tension_slider.value
        self._camera_path.mark_spline_dirty()

    def _on_move_toggled(self, event: viser.GuiEvent) -> None:
        """Create or clear per-keyframe transform gizmos."""
        if not self._path.move_cb.value:
            self._clear_transform_controls()
            return
        if event.client is not None:
            self._create_transform_controls(event.client)

    def _on_show_spline(self, _event: viser.GuiEvent) -> None:
        """Toggle spline visibility and redraw."""
        self._camera_path.show_spline = self._path.show_spline_cb.value
        self._camera_path.update_spline()

    def _clear_transform_controls(self) -> None:
        """Remove all keyframe transform gizmos."""
        for c in self._transform_controls:
            c.remove()
        self._transform_controls.clear()

    def _create_transform_controls(self, client: viser.ClientHandle) -> None:
        """Create a transform gizmo per keyframe that drives its pose on drag."""
        for idx, kf, frustum in self._camera_path.keyframe_handles():
            handle = client.scene.add_transform_controls(
                f"/keyframe_move/{idx}", scale=0.4, wxyz=kf.wxyz, position=kf.position
            )
            self._transform_controls.append(handle)
            self._bind_transform_control(handle, kf, frustum)

    def _bind_transform_control(
        self,
        ctrl: viser.TransformControlsHandle,
        kf: Keyframe4D,
        frustum: viser.CameraFrustumHandle,
    ) -> None:
        """Bind a single transform gizmo to mutate its keyframe + frustum on update."""

        @ctrl.on_update
        def _(_: viser.TransformControlsHandle) -> None:
            kf.wxyz = ctrl.wxyz
            kf.position = ctrl.position
            frustum.wxyz = ctrl.wxyz
            frustum.position = ctrl.position
            self._camera_path.mark_spline_dirty()

    def _schedule(self) -> TemporalSchedule:
        """Build the current temporal schedule from the live export handles + GUI duration."""
        return TemporalSchedule(
            mode=self._export.temporal_mode_dd.value,
            fps=self._export.fps_number.value,
            duration=self._duration_number.value,
            num_timesteps=self._viewer.num_timesteps,
            loop_count=self._export.loop_count_num.value,
            ping_pong=self._export.ping_pong_cb.value,
            global_tau=self._export.global_tau_sl.value,
        )

    def _get_max_frame(self) -> int:
        """Maximum preview frame index for the current schedule."""
        return self._schedule().max_preview_frame()

    def _wire_preview_events(self) -> None:
        """Bind preview frame scrub, playback, and preview-render enter/exit."""
        preview = self._preview
        preview.frame_slider.on_update(lambda _event: self._update_preview_at_frame())
        preview.play_btn.on_click(self._on_preview_play)
        preview.pause_btn.on_click(self._on_preview_pause)
        preview.preview_render_btn.on_click(lambda _event: self._enter_preview_render())
        preview.exit_preview_btn.on_click(lambda _event: self._exit_preview_render())

    def _on_preview_play(self, _event: viser.GuiEvent) -> None:
        """Start preview playback on the shared playback timer."""
        self._preview.play_btn.visible = False
        self._preview.pause_btn.visible = True
        self._viewer.temporal.play_preview(self._advance_preview_frame, self._preview_period)

    def _on_preview_pause(self, _event: viser.GuiEvent) -> None:
        """Stop preview playback."""
        self._preview.play_btn.visible = True
        self._preview.pause_btn.visible = False
        self._viewer.temporal.stop_preview()

    def _advance_preview_frame(self) -> None:
        """Advance the preview frame slider by one, wrapping at the schedule's frame count."""
        max_frame = self._get_max_frame()
        if max_frame > 0:
            self._preview.frame_slider.value = (self._preview.frame_slider.value + 1) % max_frame

    def _preview_period(self) -> float:
        """Preview playback period in seconds from the export FPS."""
        return 1.0 / max(self._export.fps_number.value, 1.0)

    def _update_preview_at_frame(self) -> None:
        """Update the preview frustum + render_tab_state for the current preview frame."""
        sched = self._schedule()
        frame_i = self._preview.frame_slider.value
        max_frame = sched.max_preview_frame()
        normalized_t = frame_i / max_frame if max_frame > 0 else 0.0
        frame = self._camera_path.interpolate(normalized_t)
        if frame is None:
            self._remove_preview_camera()
            return
        self._render_tab.preview_time = sched.tau(frame_i, frame.timestamp)
        self._show_preview_camera(frame)

    def _show_preview_camera(self, frame: object) -> None:
        """Display a green frustum at the interpolated camera pose."""
        self._preview_camera_handle = self._server.scene.add_camera_frustum(
            "/preview_camera",
            fov=frame.fov_rad,
            aspect=DEFAULT_ASPECT,
            scale=0.35,
            wxyz=frame.pose.rotation().wxyz,
            position=frame.pose.translation(),
            color=(10, 200, 30),
        )
        if self._render_tab.preview_render:
            for client in self._server.get_clients().values():
                client.camera.wxyz = frame.pose.rotation().wxyz
                client.camera.position = frame.pose.translation()
                client.camera.fov = frame.fov_rad

    def _remove_preview_camera(self) -> None:
        """Remove the preview camera frustum from the scene."""
        if self._preview_camera_handle is not None:
            self._preview_camera_handle.remove()
            self._preview_camera_handle = None

    def _refresh_frame_slider(self) -> None:
        """Recreate the preview frame slider when duration / FPS / mode changes."""
        old_slider = self._preview.frame_slider
        max_frame = self._get_max_frame()
        with self._preview_folder:
            new_slider = self._server.gui.add_slider(
                "Preview Frame",
                min=0,
                max=max_frame,
                step=1,
                initial_value=0,
                disabled=max_frame <= 1,
                order=old_slider.order if old_slider else 0,
            )
        new_slider.on_update(lambda _event: self._update_preview_at_frame())
        if old_slider is not None:
            old_slider.remove()
        self._preview.frame_slider = new_slider
        self._remove_preview_camera()

    def _enter_preview_render(self) -> None:
        """Enter preview-render mode: hide the scene and move cameras to the preview pose."""
        self._render_tab.preview_render = True
        self._preview.preview_render_btn.visible = False
        self._preview.exit_preview_btn.visible = True
        self._server.scene.set_global_visibility(False)
        for client in self._server.get_clients().values():
            self._camera_pose_backup[client.client_id] = (
                client.camera.position,
                client.camera.look_at,
                client.camera.up_direction,
            )
        self._update_preview_at_frame()

    def _exit_preview_render(self) -> None:
        """Exit preview-render mode: restore scene and camera poses."""
        self._render_tab.preview_render = False
        self._preview.preview_render_btn.visible = True
        self._preview.exit_preview_btn.visible = False
        for client in self._server.get_clients().values():
            backup = self._camera_pose_backup.pop(client.client_id, None)
            if backup is not None:
                client.camera.position = backup[0]
                client.camera.look_at = backup[1]
                client.camera.up_direction = backup[2]
                client.flush()
        self._server.scene.set_global_visibility(True)

    def _wire_export_events(self) -> None:
        """Bind the export controls (mode, loop count, fps, duration, save/load/render)."""
        export = self._export
        export.temporal_mode_dd.on_update(self._on_temporal_mode_changed)
        export.loop_count_num.on_update(self._on_export_param_changed)
        export.fps_number.on_update(self._on_export_param_changed)
        self._duration_number.on_update(self._on_duration_changed)
        export.render_res.on_update(lambda _event: self._camera_path.update_aspect(self._aspect()))
        export.save_btn.on_click(lambda _event: self._save_trajectory())
        export.load_btn.on_click(self._on_load_clicked)
        export.render_btn.on_click(lambda _event: self._on_render_clicked())

    def _on_temporal_mode_changed(self, _event: viser.GuiEvent) -> None:
        """Toggle mode-specific control visibility and refresh derived state."""
        export = self._export
        mode = export.temporal_mode_dd.value
        export.loop_count_num.visible = mode == "Loop"
        export.ping_pong_cb.visible = mode == "Loop"
        export.global_tau_sl.visible = mode == "Freeze"
        self._camera_path.tau_editable = mode == "Manual"
        self._path.trans_num.disabled = mode == "Loop"
        self._update_tau_fn()
        self._update_viz_n_frames()
        self._refresh_frame_slider()
        self._update_frames_display()

    def _on_export_param_changed(self, _event: viser.GuiEvent) -> None:
        """React to loop-count / FPS changes (viz frames, slider range, frame count)."""
        self._update_viz_n_frames()
        self._refresh_frame_slider()
        self._update_frames_display()

    def _on_duration_changed(self, _event: viser.GuiEvent) -> None:
        """React to a duration change (viz frames, slider range, frame count)."""
        self._update_viz_n_frames()
        self._refresh_frame_slider()
        self._update_frames_display()

    def _update_viz_n_frames(self) -> None:
        """Set the spline-visualization frame count from the schedule and redraw."""
        self._camera_path.viz_n_frames = self._schedule().n_frames(minimum=2)
        self._camera_path.refresh_spline_visualization()

    def _update_frames_display(self) -> None:
        """Update the Total Frames readout from the schedule."""
        self._export.total_frames_display.value = float(self._schedule().n_frames(minimum=1))

    def _update_tau_fn(self) -> None:
        """Install the per-keyframe τ function for the current temporal mode."""
        mode = self._export.temporal_mode_dd.value
        if mode == "Manual":
            self._camera_path.tau_value_fn = None
        elif mode == "Freeze":
            self._camera_path.tau_value_fn = lambda _kf_idx: self._export.global_tau_sl.value
        else:
            self._camera_path.tau_value_fn = self._loop_tau

    def _loop_tau(self, kf_idx: int) -> float:
        """τ for a keyframe in Loop mode: spread evenly across one sweep by keyframe order."""
        sorted_keys = self._camera_path.sorted_keyframe_indices()
        k = sorted_keys.index(kf_idx) if kf_idx in sorted_keys else 0
        n = len(sorted_keys)
        normalized_t = k / max(1, n - 1) if n > 1 else 0.0
        sched = self._schedule()
        i = round(normalized_t * max(1, sched.n_frames(minimum=1) - 1))
        return sched.tau(i, 0.0)

    def _save_trajectory(self) -> None:
        """Serialize the path + settings to ``<output>/camera_paths/<name>.json``."""
        export = self._export
        data = {
            "version": 1,
            "keyframes": self._camera_path.to_json(),
            "settings": {
                "default_transition_sec": self._camera_path.default_transition_sec,
                "default_fov_deg": self._camera_path.default_fov_rad * 180.0 / np.pi,
                "default_timestamp": self._camera_path.default_timestamp,
                "tension": self._camera_path.tension,
                "render_width": int(export.render_res.value[0]),
                "render_height": int(export.render_res.value[1]),
                "fps": export.fps_number.value,
            },
        }
        out_path = Path(export.output_dir_text.value) / "camera_paths" / f"{export.traj_name.value}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        notice(f"Trajectory saved: {out_path}")

    def _on_load_clicked(self, event: viser.GuiEvent) -> None:
        """Open the trajectory-selection modal for the requesting client."""
        if event.client is not None:
            self._show_load_modal(event.client)

    def _show_load_modal(self, client: viser.ClientHandle) -> None:
        """Show a modal listing saved trajectories and load the chosen one."""
        cam_dir = Path(self._export.output_dir_text.value) / "camera_paths"
        cam_dir.mkdir(parents=True, exist_ok=True)
        filenames = [p.name for p in cam_dir.glob("*.json")]

        with client.gui.add_modal("Load Trajectory") as modal:
            if not filenames:
                client.gui.add_markdown("No saved trajectories found.")
            else:
                client.gui.add_markdown("Select a trajectory:")
                dropdown = client.gui.add_dropdown("File", options=filenames, initial_value=filenames[0])
                load_btn = client.gui.add_button("Load")

                @load_btn.on_click
                def _(_: viser.GuiEvent) -> None:
                    self._apply_loaded_trajectory(json.loads((cam_dir / dropdown.value).read_text()))
                    modal.close()

            cancel_btn = client.gui.add_button("Cancel")
            cancel_btn.on_click(lambda _event: modal.close())

    def _apply_loaded_trajectory(self, data: dict) -> None:
        """Apply loaded JSON to the path geometry and the export controls."""
        settings = data.get("settings", {})
        cp = self._camera_path
        cp.default_transition_sec = settings.get("default_transition_sec", 2.0)
        cp.default_fov_rad = settings.get("default_fov_deg", 50.0) / 180.0 * np.pi
        cp.default_timestamp = settings.get("default_timestamp", -0.5)
        cp.tension = settings.get("tension", 0.5)
        self._export.render_res.value = (settings.get("render_width", 1280), settings.get("render_height", 720))
        self._export.fps_number.value = settings.get("fps", DEFAULT_FPS)
        cp.load_keyframes(data.get("keyframes", []))

    def _all_controls(self) -> list[viser.GuiInputHandle]:
        """Every input handle in the panel, disabled during a video export."""
        return [
            self._duration_number,
            *handle_values(self._path),
            *handle_values(self._preview),
            *handle_values(self._export),
        ]

    def _on_render_clicked(self) -> None:
        """Build the export schedule and hand off to the video exporter."""
        duration = self._camera_path.compute_duration()
        if duration <= 0:
            warn("No camera path to render.")
            return
        export = self._export
        width = int(export.render_res.value[0])
        height = int(export.render_res.value[1])
        fps = int(export.fps_number.value)
        schedule = TemporalSchedule(
            mode=export.temporal_mode_dd.value,
            fps=fps,
            duration=duration,
            num_timesteps=self._viewer.num_timesteps,
            loop_count=export.loop_count_num.value,
            ping_pong=export.ping_pong_cb.value,
            global_tau=export.global_tau_sl.value,
        )
        out_path = Path(export.output_dir_text.value) / "videos" / f"{export.video_name.value}.mp4"
        self._video_exporter.export(self._camera_path, schedule, width, height, fps, out_path, self._all_controls())
