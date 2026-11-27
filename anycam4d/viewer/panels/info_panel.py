"""InfoPanel — Total / Rendered primitive-count readouts.

Replaces ``Anycam4DViewer._setup_info_controls``. The viewer writes ``rendered_primitives`` on
a throttle and ``total_primitives`` on model update; this panel just owns the two displays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import viser


class InfoPanel:
    """Builds the Info folder with total and rendered primitive counts.

    Args:
        server: Viser server.
        total_primitives: Initial total primitive count.
    """

    def __init__(self, server: viser.ViserServer, total_primitives: int) -> None:
        """Build the Info folder."""
        with server.gui.add_folder("Info", expand_by_default=False):
            self.total_primitives = server.gui.add_number(
                "Total Primitives",
                initial_value=total_primitives,
                disabled=True,
            )
            self.rendered_primitives = server.gui.add_number(
                "Rendered Primitives",
                initial_value=0,
                disabled=True,
            )
