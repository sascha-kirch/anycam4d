"""The only module that touches nerfview's private state (rendering tab, render threads).

``Anycam4DViewer`` overrides nerfview's ``_init_rendering_tab`` / ``_populate_rendering_tab`` hooks; everything
else it needs from the undocumented ``RenderTabState`` goes through :func:`new_render_tab_state` and the typed
:class:`RenderTabView`, so a nerfview upgrade that renames those fields breaks here and nowhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nerfview import RenderTabState


if TYPE_CHECKING:
    from nerfview import Viewer


def install_lock(viewer: Viewer, lock: object) -> None:
    """Replace the viewer's lock, including in render threads of browsers that connected during construction."""
    viewer.lock = lock
    for renderer in viewer._renderers.values():  # noqa: SLF001
        renderer.lock = lock


def new_render_tab_state() -> RenderTabState:
    """Construct nerfview's render-tab state (the one site that does so)."""
    return RenderTabState()


class RenderTabView:
    """Typed, documented view over the nerfview ``RenderTabState`` fields the viewer uses.

    Args:
        state: The nerfview render-tab state to wrap (created by :func:`new_render_tab_state`
            or handed to the render callback by nerfview's render thread).
    """

    def __init__(self, state: RenderTabState) -> None:
        """Wrap a nerfview render-tab state."""
        self._state = state

    @property
    def raw(self) -> RenderTabState:
        """The underlying nerfview state object (for code that must pass it to nerfview)."""
        return self._state

    @property
    def viewer_width(self) -> int:
        """Render width nerfview computed for the current frame."""
        return self._state.viewer_width

    @property
    def viewer_height(self) -> int:
        """Render height nerfview computed for the current frame."""
        return self._state.viewer_height

    @property
    def viewer_res(self) -> int:
        """Maximum interactive render resolution."""
        return self._state.viewer_res

    @viewer_res.setter
    def viewer_res(self, value: int) -> None:
        self._state.viewer_res = value

    @property
    def preview_time(self) -> float:
        """τ used while previewing a camera-path frame."""
        return self._state.preview_time

    @preview_time.setter
    def preview_time(self, value: float) -> None:
        self._state.preview_time = value

    @property
    def preview_render(self) -> bool:
        """Whether the viewer is in preview-render mode."""
        return self._state.preview_render

    @preview_render.setter
    def preview_render(self, value: bool) -> None:
        self._state.preview_render = value
