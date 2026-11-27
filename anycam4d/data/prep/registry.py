"""Dataset registry: one ``conf/datasets`` YAML per source, importer construction and variant stage mapping. No torch."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf
from pydantic import BaseModel
from pydantic import model_validator

from anycam4d.data.prep.importers.deepview import DeepViewImporter
from anycam4d.data.prep.importers.dynerf import DyNeRFImporter
from anycam4d.data.prep.importers.selfcap import SelfCapImporter
from anycam4d.data.prep.stages.resize import Resize
from anycam4d.data.prep.stages.to_pinhole import ToPinhole
from anycam4d.data.prep.variants import VariantSpec


if TYPE_CHECKING:
    from anycam4d.data.prep.colmap import ColmapOptions
    from anycam4d.data.prep.stream import VideoImporterProto


class LicenseInfo(BaseModel):
    """License/source metadata printed before fetching a scene's raw data (EULA reminder)."""

    redistribute: bool
    source: str


class SceneConfig(BaseModel):
    """Per-scene import parameters and variants; ``excluded_cams`` drops cameras before calibration."""

    source_subdir: str
    fps: float
    eval_cams: list[int]
    excluded_cams: list[int] = []  # camera ids dropped before calibration/extraction (e.g. out-of-focus views)
    variants: list[str]
    scales: list[float] = [1.0]  # resolution tiers; every variant is derived at each (1.0 == full native res)
    start_frame: int = 0  # SelfCap setup trim
    end_frame: int | None = None  # SelfCap teardown trim
    calib_dir: str | None = None  # SelfCap separate-calibration release (relative to source_root)
    max_frames: int | None = None
    split_parts: int = 0  # archive ships as <subdir>.z01..zNN + <subdir>.zip (DyNeRF flame_salmon_1)

    @model_validator(mode="after")
    def _eval_cams_not_excluded(self) -> SceneConfig:
        """An excluded camera cannot also be an eval camera — the eval split would come out empty."""
        clash = sorted(set(self.eval_cams) & set(self.excluded_cams))
        if clash:
            raise ValueError(f"camera(s) {clash} are listed in both eval_cams and excluded_cams")
        return self


class DatasetConfig(BaseModel):
    """One dataset registry file: license + source + per-scene configs, keyed by scene name."""

    dataset: str
    license: LicenseInfo
    source_url: str
    scenes: dict[str, SceneConfig]


def load_registry(path: Path) -> DatasetConfig:
    """Parse one dataset YAML into a validated :class:`DatasetConfig`."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return DatasetConfig.model_validate(raw)


def check_scene(cfg: DatasetConfig, scene: str) -> None:
    """Raise a clear error naming the available scenes if ``scene`` is not in the registry."""
    if scene not in cfg.scenes:
        available = ", ".join(sorted(cfg.scenes))
        raise ValueError(f"scene '{scene}' not in {cfg.dataset} registry; available scenes: {available}")


def manifest_source(cfg: DatasetConfig, scene: str) -> str:
    """Human-readable ``dataset/scene`` provenance tag for this scene."""
    return f"{cfg.dataset}/{scene}"


def build_importer(
    cfg: DatasetConfig,
    scene: str,
    source_root: Path,
    work_dir: Path,
    colmap: ColmapOptions,
) -> VideoImporterProto:
    """Construct the importer for ``cfg.dataset``; SelfCap takes no ``work_dir`` or COLMAP options."""
    sc = cfg.scenes[scene]
    scene_dir = source_root / sc.source_subdir

    if cfg.dataset == "dynerf":
        return DyNeRFImporter(
            scene_dir,
            work_dir,
            fps_cap=sc.fps,
            exclude_cams=sc.excluded_cams,
            max_frames=sc.max_frames,
            colmap=colmap,
        )

    if cfg.dataset == "deepview":
        return DeepViewImporter(
            scene_dir,
            work_dir,
            fps_cap=sc.fps,
            exclude_cams=sc.excluded_cams,
            max_frames=sc.max_frames,
            colmap=colmap,
        )

    if cfg.dataset == "selfcap":
        calib_dir = _resolve_calib_dir(sc.calib_dir, source_root)
        return SelfCapImporter(
            scene_dir,
            calib_dir=calib_dir,
            fps_cap=sc.fps,
            start_frame=sc.start_frame,
            end_frame=sc.end_frame,
            exclude_cams=sc.excluded_cams,
            max_frames=sc.max_frames,
        )

    if cfg.dataset == "dynopsis":
        raise ValueError("dynopsis ships prepared: run `download`, there is nothing to build")
    raise ValueError(f"unknown dataset '{cfg.dataset}' (expected dynerf/deepview/selfcap/dynopsis)")


def _resolve_calib_dir(calib_dir: str | None, source_root: Path) -> Path | None:
    """Resolve a SelfCap calibration dir: absolute as-is, else relative to source_root."""
    if calib_dir is None:
        return None
    p = Path(calib_dir)
    return p if p.is_absolute() else source_root / p


def variant_spec(name: str, scale: float = 1.0) -> VariantSpec:
    """Stage pipeline and image format for a variant at a resolution scale.

    ``native`` is an empty pipeline and ``undistorted`` rectifies to pinhole; ``scale != 1`` prepends a
    :class:`Resize`, so the remap runs on the smaller image.
    """
    if name == "native":
        stages: list = []
    elif name == "undistorted":
        stages = [ToPinhole()]
    else:
        raise ValueError(f"unknown variant '{name}' (expected native/undistorted)")

    if scale != 1.0:
        stages = [Resize(factor=scale), *stages]

    return VariantSpec(name=name, stages=stages, image_format="jpg")
