"""Shared K/c2w derivation from a ``(CaptureSchema, CameraSchema)`` pair.

The single place a canonical dataset's calibration is turned into ``(model, K, params, c2w)``.
Both ``Anycam4DSource._build_view`` (the read-side pivot reconstruction) and the geometry
validator read intrinsics + pose through this one helper, so the two paths can never derive the
pose two different ways — a divergence there would let a convention bug in one path be masked by
the same bug in the other. Torch-free (numpy only), so it stays inside ``prep/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from anycam4d.data.prep.pose import quat_translation_to_c2w


if TYPE_CHECKING:
    from jaxtyping import Float

    from anycam4d.data.schema import CameraSchema
    from anycam4d.data.schema import CaptureSchema
    from anycam4d.enums import CameraModelType


def capture_calibration(
    capture: CaptureSchema, camera: CameraSchema
) -> tuple[CameraModelType, Float[np.ndarray, "3 3"], list[float], Float[np.ndarray, "4 4"]]:
    """Derive ``(model, K, params, c2w)`` for one capture through its camera.

    Args:
        capture: The capture record (pose as ``[qw,qx,qy,qz]`` c2w quaternion + camera-origin
            translation).
        camera: The camera the capture references (model + ``[fx,fy,cx,cy,<distortion...>]``).

    Returns:
        ``(model, K, params, c2w)``: the camera model, the ``[3, 3]`` intrinsic matrix built
        from ``params[:4]``, the params list verbatim (``params[4:]`` is cv2 ``distCoeffs``
        order), and the ``[4, 4]`` OpenCV camera-to-world matrix from the capture pose.
    """
    fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    c2w = quat_translation_to_c2w(capture.rotation, capture.translation)
    return camera.model, k, list(camera.params), c2w
