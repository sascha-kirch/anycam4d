"""Per-model camera facts: distortion count, projection family, supporting backends.

CUDA-free, so the dataset schema can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from anycam4d.enums import BackendType
from anycam4d.enums import CameraModelType


class ProjectionFamily(Enum):
    """Projection family: the compile-time device dispatch tag."""

    PINHOLE_FAMILY = "pinhole_family"
    FISHEYE_FAMILY = "fisheye_family"


# Distortion slots per family, OpenCV order, zero-filled: pinhole [k1,k2,p1,p2,k3,k4,k5,k6,s1..s4], fisheye [k1..k4].
FAMILY_N_SLOTS: dict[ProjectionFamily, int] = {
    ProjectionFamily.PINHOLE_FAMILY: 12,
    ProjectionFamily.FISHEYE_FAMILY: 4,
}

# Must mirror ``enum ProjectionFamily`` in cuda/csrc/common.h.
FAMILY_CUDA_ID: dict[ProjectionFamily, int] = {
    ProjectionFamily.PINHOLE_FAMILY: 0,
    ProjectionFamily.FISHEYE_FAMILY: 1,
}


@dataclass(frozen=True)
class CameraSpec:
    """Per-model camera facts.

    Attributes:
        name: ``CameraModelType`` value.
        family: Projection family.
        n_distortion: Maximum distortion coefficient count.
        backends: Render backends that support the model.
    """

    name: str
    family: ProjectionFamily
    n_distortion: int
    backends: frozenset[BackendType]


CAMERA_REGISTRY: dict[CameraModelType, CameraSpec] = {
    CameraModelType.PINHOLE: CameraSpec(
        name="pinhole",
        family=ProjectionFamily.PINHOLE_FAMILY,
        n_distortion=0,
        backends=frozenset({BackendType.EWA, BackendType.ANYCAM4D}),
    ),
    CameraModelType.OPENCV: CameraSpec(
        name="opencv",
        family=ProjectionFamily.PINHOLE_FAMILY,
        n_distortion=5,  # classic Brown-Conrady [k1,k2,p1,p2,k3]
        backends=frozenset({BackendType.EWA, BackendType.ANYCAM4D}),
    ),
    CameraModelType.OPENCV_FULL: CameraSpec(
        name="opencv_full",
        family=ProjectionFamily.PINHOLE_FAMILY,
        n_distortion=12,  # rational + thin-prism [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4]
        backends=frozenset({BackendType.EWA, BackendType.ANYCAM4D}),
    ),
    CameraModelType.FISHEYE_EQUIDISTANT: CameraSpec(
        name="fisheye_equidistant",
        family=ProjectionFamily.FISHEYE_FAMILY,
        n_distortion=0,  # Equidistant r=f*theta -- Kannala-Brandt with k1=k2=k3=k4=0
        backends=frozenset({BackendType.EWA, BackendType.ANYCAM4D}),
    ),
    CameraModelType.FISHEYE_OPENCV: CameraSpec(
        name="fisheye_opencv",
        family=ProjectionFamily.FISHEYE_FAMILY,
        n_distortion=4,  # Kannala-Brandt [k1,k2,k3,k4]
        backends=frozenset({BackendType.EWA, BackendType.ANYCAM4D}),
    ),
}


def param_count_range(model: CameraModelType) -> tuple[int, int]:
    """Inclusive param count range ``(4, 4 + n_distortion)``; any positional distortion prefix is legal."""
    return 4, 4 + CAMERA_REGISTRY[model].n_distortion


def distortion_width(model: CameraModelType) -> int:
    """Fixed distortion vector width for the model's family (12 pinhole, 4 fisheye)."""
    return FAMILY_N_SLOTS[CAMERA_REGISTRY[model].family]


def family_cuda_id(model: CameraModelType) -> int:
    """Projection-family int the CUDA host switch dispatches on."""
    return FAMILY_CUDA_ID[CAMERA_REGISTRY[model].family]
