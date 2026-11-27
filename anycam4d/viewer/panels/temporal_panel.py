"""TemporalPanel: the Time folder (Play, τ, frame, speed, loop), wired to the single :class:`TemporalController`."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import viser

    from anycam4d.viewer.temporal_controller import TemporalController


class TemporalPanel:
    """Builds the temporal-playback controls and wires them to the controller.

    Args:
        server: Viser server.
        controller: The single owner of live τ / playback.
        num_timesteps: Number of discrete timesteps (sets the τ slider step).
    """

    def __init__(
        self,
        server: viser.ViserServer,
        controller: TemporalController,
        num_timesteps: int,
    ) -> None:
        """Build the Time folder and attach its widgets to the controller."""
        self._controller = controller

        with server.gui.add_folder("Time"):
            self.play_button = server.gui.add_button("▶ Play")
            self.time_slider = server.gui.add_slider(
                "τ",
                min=-0.5,
                max=0.5,
                step=1.0 / num_timesteps,
                initial_value=-0.5,
                hint="Normalized timestamp τ ∈ [-0.5, 0.5] (start → end of sequence).",
            )
            self.frame_display = server.gui.add_number(
                "Frame",
                initial_value=0,
                disabled=True,
                hint="Current frame index (0-based), derived from τ.",
            )
            self.speed_slider = server.gui.add_slider("Speed (x)", min=0.1, max=5.0, step=0.1, initial_value=1.0)
            self.loop_toggle = server.gui.add_checkbox("Loop", initial_value=True)

        self.time_slider.on_update(self._on_tau)
        self.play_button.on_click(lambda _event: self._controller.toggle_live())
        self.loop_toggle.on_update(self._on_loop)
        self.speed_slider.on_update(self._on_speed)

        self._controller.attach_widgets(
            slider=self.time_slider,
            frame_display=self.frame_display,
            play_button=self.play_button,
        )

    def _on_tau(self, _event: viser.GuiEvent) -> None:
        """Drive τ from a genuine user drag, ignoring the echo of a programmatic write."""
        if self._controller.suppressing_slider_cb:
            return
        self._controller.set_tau(self.time_slider.value)

    def _on_loop(self, _event: viser.GuiEvent) -> None:
        """Update the controller loop flag."""
        self._controller.loop = self.loop_toggle.value

    def _on_speed(self, _event: viser.GuiEvent) -> None:
        """Update the controller playback speed."""
        self._controller.speed = self.speed_slider.value
