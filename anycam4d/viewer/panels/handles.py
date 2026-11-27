"""Typed GUI-handle containers for the camera-path panel.

Replaces the old untyped ``handles: dict[str, viser.GuiInputHandle]`` service locator.
Each sub-section of the camera-path panel (path authoring, preview, export) owns a typed
dataclass holding its concrete viser handles, so consumers read them as attributes instead
of doing string-keyed ``.get("...")`` lookups with late, implicit population.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import viser


@dataclasses.dataclass
class CameraPathHandles:
    """Handles for the Camera Path authoring sub-folder."""

    add_kf_btn: viser.GuiButtonHandle
    clear_kf_btn: viser.GuiButtonHandle
    fov_slider: viser.GuiSliderHandle
    default_tau: viser.GuiSliderHandle
    trans_num: viser.GuiNumberHandle
    tension_slider: viser.GuiSliderHandle
    move_cb: viser.GuiCheckboxHandle
    show_kf_cb: viser.GuiCheckboxHandle
    show_spline_cb: viser.GuiCheckboxHandle
    show_lookat_cb: viser.GuiCheckboxHandle
    apply_lookat_btn: viser.GuiButtonHandle


@dataclasses.dataclass
class PreviewHandles:
    """Handles for the Preview sub-folder.

    ``frame_slider`` is replaced (not mutated) when duration / FPS / mode change, so it is
    not frozen.
    """

    frame_slider: viser.GuiSliderHandle
    play_btn: viser.GuiButtonHandle
    pause_btn: viser.GuiButtonHandle
    preview_render_btn: viser.GuiButtonHandle
    exit_preview_btn: viser.GuiButtonHandle


@dataclasses.dataclass
class ExportHandles:
    """Handles for the Export sub-folder."""

    render_res: viser.GuiVector2Handle
    fps_number: viser.GuiNumberHandle
    temporal_mode_dd: viser.GuiDropdownHandle
    loop_count_num: viser.GuiNumberHandle
    ping_pong_cb: viser.GuiCheckboxHandle
    global_tau_sl: viser.GuiSliderHandle
    output_dir_text: viser.GuiTextHandle
    traj_name: viser.GuiTextHandle
    video_name: viser.GuiTextHandle
    save_btn: viser.GuiButtonHandle
    load_btn: viser.GuiButtonHandle
    render_btn: viser.GuiButtonHandle
    total_frames_display: viser.GuiNumberHandle


def handle_values(container: object) -> list[viser.GuiInputHandle]:
    """Return the viser handles held by a handles dataclass instance.

    Used to bulk-disable controls during a video export and restore them afterward.

    Args:
        container: A handles dataclass instance (CameraPathHandles / PreviewHandles / ...).

    Returns:
        The concrete handle objects stored on the dataclass fields.
    """
    return [getattr(container, f.name) for f in dataclasses.fields(container)]
