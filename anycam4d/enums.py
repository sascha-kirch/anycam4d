"""Centralized enum definitions.

All members are :class:`~enum.StrEnum`: the ``.value`` string is a load-bearing contract — the
on-disk JSON model tag, a Hydra/YAML config value, or a gsplat API string (e.g. ``RenderMode`` is
``"RGB+ED"``). The values are therefore written explicitly and must never be replaced with
``enum.auto()`` (which would derive them from the member name and silently break the contract).
"""

from enum import StrEnum


class BackendType(StrEnum):
    """Render backend type."""

    EWA = "ewa"
    ANYCAM4D = "anycam4d"  # ray casting through the lens, unscented-transform projection


class AppearanceType(StrEnum):
    """Appearance model type."""

    SH4D_ADDITIVE = "sh4d_additive"
    SH4D_MULTIPLICATIVE = "sh4d_multiplicative"


class ModelType(StrEnum):
    """Primitive type stored in ``ModelConfig.model_type`` — dispatch key for ``ModelFactory``."""

    FOURD_GAUSSIANS = "fourdgaussians"


class SamplerType(StrEnum):
    """DataLoader sampler type."""

    UNIFORM = "uniform"
    FRAME_ALIGNED = "frame_aligned"
    TEMPORAL_TILED = "temporal_tiled"


class ParentScore(StrEnum):
    """What MCMC samples relocation and growth parents by.

    ``OPACITY`` is gsplat's; ``MOTION`` is the mean of the opacity and image-gradient percentile ranks times
    ``1 + 3 rank(speed)``.
    """

    OPACITY = "opacity"
    MOTION = "motion"


class StrategyType(StrEnum):
    """Densification strategy type."""

    DEFAULT_SPATIAL = "default_spatial"
    OG_4DGS = "og_4dgs"  # published 4DGS baseline (spatial trigger, full-4D split)
    MCMC_SPATIAL = "mcmc_spatial"
    NULL_4DGS = "null_4dgs"


class SpatialTrigger(StrEnum):
    """Spatial densification trigger source for the gradient (Default/OG) families.

    Screen-space uses the projected ``means2d.grad`` magnitude, which only exists on the EWA
    backend. Parameter-space uses ``|means.grad|`` in world coordinates and therefore works on any
    renderer, including the ray backend (UT) that never form a ``means2d`` gradient.
    """

    SCREENSPACE = "screenspace"  # means2d.grad — EWA backend only
    PARAMSPACE = "paramspace"  # |means.grad| — any renderer (required for ray backends)


class RenderMode(StrEnum):
    """gsplat rasterization render mode."""

    RGB = "RGB"
    RGB_ED = "RGB+ED"


class RasterizeMode(StrEnum):
    """gsplat rasterization mode."""

    CLASSIC = "classic"
    ANTIALIASED = "antialiased"


class CameraModelType(StrEnum):
    """Camera projection model.

    Exactly five members. ``PINHOLE`` and the two OpenCV radial+tangential models
    (``OPENCV`` classic 5-coeff Brown-Conrady, ``OPENCV_FULL`` 12-coeff rational +
    thin-prism) are pinhole-family. ``FISHEYE_EQUIDISTANT`` (the zero-distortion
    ``r=f*theta`` model) and ``FISHEYE_OPENCV`` (Kannala-Brandt) are fisheye-family;
    equidistant is to ``FISHEYE_OPENCV`` what ``PINHOLE`` is to the OpenCV models.
    The ``.value`` strings are the on-disk JSON contract (``"model": "..."``).
    """

    PINHOLE = "pinhole"
    OPENCV = "opencv"
    OPENCV_FULL = "opencv_full"
    FISHEYE_EQUIDISTANT = "fisheye_equidistant"
    FISHEYE_OPENCV = "fisheye_opencv"


class LpipsNet(StrEnum):
    """LPIPS backbone network for the perceptual metric."""

    ALEX = "alex"
    VGG = "vgg"
    SQUEEZE = "squeeze"


class PostProcessorType(StrEnum):
    """Post-processor applied after rasterization."""

    NONE = "none"
    BILATERAL_GRID = "bilateral_grid"
