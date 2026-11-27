"""The full viewer constructs and renders a frame on a real viser server (skipped without CUDA or viser)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anycam4d.config.rendering import RenderingConfig
from anycam4d.primitives.fourdgaussians.model import FourDGaussians
from anycam4d.viewer.camera_path import Keyframe4D
from anycam4d.renderer.factory import BackendFactory
from anycam4d.renderer.pipeline import RenderPipeline
from anycam4d.viewer.viewer import Anycam4DViewer


@pytest.fixture()
def launched_viewer():  # noqa: ANN201
    """A fully constructed Anycam4DViewer on a real server, or skip if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("viewer launch smoke needs CUDA")
    viser = pytest.importorskip("viser")
    try:
        server = viser.ViserServer(host="127.0.0.1", port=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"viser server unavailable: {exc}")

    model = FourDGaussians.from_random(n_points=500, sh_degree=3).to("cuda")
    pipeline = RenderPipeline(backend=BackendFactory.create(RenderingConfig()))
    viewer = Anycam4DViewer(server, model, pipeline, num_timesteps=10)
    yield viewer
    server.stop()


def test_viewer_constructs_with_panels(launched_viewer: Anycam4DViewer) -> None:
    """The viewer builds its panels and render-settings holder without error."""
    assert launched_viewer._rendering_panel is not None  # noqa: SLF001
    assert launched_viewer._camera_path_panel is not None  # noqa: SLF001
    assert launched_viewer._temporal_panel is not None  # noqa: SLF001
    assert launched_viewer._info_panel is not None  # noqa: SLF001
    assert launched_viewer.render_settings.render_mode == "rgb"


def test_viewer_renders_one_frame(launched_viewer: Anycam4DViewer) -> None:
    """The offline render path produces a uint8 RGB image of the requested size."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 3.0  # back the camera off the origin
    rgb = launched_viewer.render_frame(c2w=c2w, fov_rad=1.0, width=32, height=24, timestamp=0.0)
    assert rgb.shape == (24, 32, 3)
    assert rgb.dtype == np.uint8


def test_camera_path_interpolates_after_keyframes(launched_viewer: Anycam4DViewer) -> None:
    """Two keyframes added through the view produce a valid interpolated frame."""
    view = launched_viewer._camera_path_panel._camera_path  # noqa: SLF001
    view.add_keyframe(Keyframe4D(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 1.5, -0.5, False, None))
    view.add_keyframe(Keyframe4D(np.ones(3), np.array([1.0, 0.0, 0.0, 0.0]), 1.5, 0.5, False, None))
    view.update_spline()
    frame = view.interpolate(0.5)
    assert frame is not None
    assert -0.5 <= frame.timestamp <= 0.5
