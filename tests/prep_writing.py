"""Serial PrepWriter driver for tests; production writes through ``build_variant``."""

from __future__ import annotations

from anycam4d.data.prep.sample import CalibratedView
from anycam4d.data.prep.writer import PrepWriter


def write_views(writer: PrepWriter, views: list[CalibratedView]) -> None:
    for view in views:
        image_path = writer.encode_view(view)
        writer.register_camera(
            view.cam_id,
            view.model,
            view.params,
            view.image.shape[1],
            view.image.shape[0],
            view.reproject_source,
            view.max_incidence_deg,
        )
        writer.add_capture(view.cam_id, view.frame_id, view.timestamp, view.c2w, image_path)
