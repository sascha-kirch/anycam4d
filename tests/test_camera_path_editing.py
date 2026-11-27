"""Camera-path geometry, coalesced spline rebuilds and in-place frustum updates (frustum tests need a viser server)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from anycam4d.viewer.camera_path import CameraPathModel
from anycam4d.viewer.camera_path import Keyframe4D
from anycam4d.viewer.camera_path_view import CameraPathView


def _kf(position: list[float], wxyz: list[float], timestamp: float, aspect: float = 1.5) -> Keyframe4D:
    """Build a keyframe with default override settings."""
    return Keyframe4D(
        position=np.array(position, dtype=float),
        wxyz=np.array(wxyz, dtype=float),
        aspect=aspect,
        timestamp=timestamp,
        override_transition_enabled=False,
        override_transition_sec=None,
    )


def _three_keyframe_path() -> CameraPathModel:
    """Build a fixed 3-keyframe geometry model headlessly."""
    cp = CameraPathModel()
    cp.tension = 0.3
    kfs = [
        _kf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], -0.5),
        Keyframe4D(
            position=np.array([1.0, 2.0, 0.5]),
            wxyz=np.array([0.9238795, 0.0, 0.3826834, 0.0]),
            aspect=1.5,
            timestamp=0.0,
            override_transition_enabled=True,
            override_transition_sec=1.5,
        ),
        _kf([2.0, -1.0, 1.0], [0.7071068, 0.7071068, 0.0, 0.0], 0.5),
    ]
    for i, kf in enumerate(kfs):
        cp._keyframes[i] = kf  # noqa: SLF001
    cp._rebuild_splines()  # noqa: SLF001
    return cp


def test_endpoints_match_keyframes() -> None:
    """t=0 and t=1 return the first and last keyframe pose exactly."""
    cp = _three_keyframe_path()
    first, last = cp.interpolate(0.0), cp.interpolate(1.0)
    assert first is not None
    assert last is not None
    np.testing.assert_allclose(first.pose.translation(), [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(last.pose.translation(), [2.0, -1.0, 1.0], atol=1e-7)


def test_serialization_round_trips() -> None:
    """to_json → load_keyframes reproduces identical interpolation geometry."""
    src = _three_keyframe_path()
    dst = CameraPathModel()
    dst.tension = src.tension
    dst.load_keyframes(src.to_json())
    dst._rebuild_splines()  # noqa: SLF001
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        a, b = src.interpolate(t), dst.interpolate(t)
        assert a is not None
        assert b is not None
        np.testing.assert_allclose(b.pose.translation(), a.pose.translation(), atol=1e-9)
        assert b.timestamp == pytest.approx(a.timestamp, abs=1e-9)


def test_model_has_no_scene_or_gui_calls() -> None:
    """The geometry model module holds no server and makes no viser scene / gui calls."""
    src = (Path(__file__).resolve().parents[1] / "anycam4d" / "viewer" / "camera_path.py").read_text()
    # No server reference and none of the scene/gui construction calls (docstring prose aside).
    assert "_server" not in src
    assert "add_camera_frustum" not in src
    assert "add_3d_gui_container" not in src
    assert "add_spline_catmull_rom" not in src


def _spy_path(quiet_sec: float = 0.02) -> tuple[CameraPathModel, list[int]]:
    """Build a 2-keyframe model with a rebuild spy (no server, no view redraw)."""
    cp = CameraPathModel()
    cp._rebuild_quiet_sec = quiet_sec  # noqa: SLF001
    for i, kf in enumerate(
        (_kf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], -0.5), _kf([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5))
    ):
        cp._keyframes[i] = kf  # noqa: SLF001

    calls: list[int] = []
    real_rebuild = cp._rebuild_splines  # noqa: SLF001

    def _counting_rebuild() -> bool:
        calls.append(1)
        return real_rebuild()

    cp._rebuild_splines = _counting_rebuild  # type: ignore[method-assign]  # noqa: SLF001
    return cp, calls


def test_burst_then_flush_rebuilds_once() -> None:
    """A burst of M mark_spline_dirty calls followed by flush rebuilds exactly once."""
    cp, calls = _spy_path()
    for _ in range(20):
        cp.mark_spline_dirty()
    assert calls == []  # nothing rebuilt yet — all coalesced
    cp.flush_pending_rebuild()
    assert len(calls) == 1


def test_trailing_timer_rebuilds_once() -> None:
    """After a burst, the trailing timer fires a single rebuild on settle."""
    cp, calls = _spy_path(quiet_sec=0.02)
    for _ in range(10):
        cp.mark_spline_dirty()
    time.sleep(0.1)
    assert len(calls) == 1


def test_flush_after_timer_does_not_double_build() -> None:
    """A flush racing a fired timer rebuilds once, not twice (dirty-flag guard)."""
    cp, calls = _spy_path(quiet_sec=0.02)
    cp.mark_spline_dirty()
    time.sleep(0.1)  # let the timer fire
    assert len(calls) == 1
    cp.flush_pending_rebuild()  # nothing pending
    assert len(calls) == 1


def test_flush_with_nothing_pending_is_noop() -> None:
    """flush_pending_rebuild with no pending edit does not rebuild."""
    cp, calls = _spy_path()
    cp.flush_pending_rebuild()
    assert calls == []


def test_interpolate_flushes_pending_edit() -> None:
    """interpolate reflects a just-made edit before any timer could fire (flush-on-read)."""
    cp, _ = _spy_path()
    cp.update_spline()  # build initial splines synchronously
    before = cp.interpolate(1.0)
    assert before is not None
    # Move the last keyframe and only mark dirty (no synchronous rebuild).
    cp._keyframes[1] = _kf([5.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5)  # noqa: SLF001
    cp.mark_spline_dirty()
    after = cp.interpolate(1.0)
    assert after is not None
    assert after.pose.translation()[0] == pytest.approx(5.0, abs=1e-6)


@pytest.fixture()
def viser_view():  # noqa: ANN201
    """A CameraPathView backed by a real viser server, or skip if one cannot start."""
    viser = pytest.importorskip("viser")
    try:
        server = viser.ViserServer(host="127.0.0.1", port=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"viser server unavailable: {exc}")

    class _Dummy:
        value = 0.0

    cp = CameraPathView(server, _Dummy())  # type: ignore[arg-type]
    cp._model.on_rebuilt = lambda: None  # avoid spline scene churn  # noqa: SLF001
    yield cp
    server.stop()


@pytest.fixture()
def viser_server():  # noqa: ANN201
    """A bare viser server with real scene ops, or skip if one cannot start."""
    viser = pytest.importorskip("viser")
    try:
        server = viser.ViserServer(host="127.0.0.1", port=0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"viser server unavailable: {exc}")
    yield server
    server.stop()


def test_viz_refresh_survives_stale_spline(viser_server) -> None:  # noqa: ANN001
    """Refreshing the viz between adding a keyframe and the spline rebuild does not crash."""

    class _Dummy:
        value = 0.0

    cp = CameraPathView(viser_server, _Dummy())  # type: ignore[arg-type]
    cp.add_keyframe(_kf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], -0.5), keyframe_index=0)
    cp.add_keyframe(_kf([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.5), keyframe_index=1)
    cp.update_spline()  # spline built for 2 keyframes (grid 0..1)

    # New keyframe present but spline not yet rebuilt: live count = 3, grid still 0..1.
    cp.add_keyframe(_kf([2.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0.0), keyframe_index=2)
    cp.refresh_spline_visualization()  # must not raise "t too big"


def test_update_frustum_preserves_node_identity(viser_view: CameraPathView) -> None:
    """update_frustum mutates the existing handle in place; no new node, no handler stacking."""
    cp = viser_view
    kf0 = _kf([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], -0.5, aspect=1.5)
    cp.add_keyframe(kf0, keyframe_index=0)
    handle_before = cp._frustums[0]  # noqa: SLF001
    handlers_before = len(handle_before._impl.click_cb)  # noqa: SLF001

    updated = _kf([1.0, 2.0, 3.0], [0.7071068, 0.7071068, 0.0, 0.0], 0.25, aspect=2.0)
    for _ in range(5):
        cp.update_frustum(0, updated)

    handle_after = cp._frustums[0]  # noqa: SLF001
    assert handle_after is handle_before  # same scene-node object, never recreated
    assert len(handle_after._impl.click_cb) == handlers_before  # noqa: SLF001  # no duplicate handlers
    np.testing.assert_allclose(handle_after.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(handle_after.wxyz, [0.7071068, 0.7071068, 0.0, 0.0])
    assert handle_after.aspect == pytest.approx(2.0)
