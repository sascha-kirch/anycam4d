"""In-process COLMAP (pycolmap 3.10) SfM, known-pose triangulation and dense MVS for the importers.

Returns anycam4d-native intrinsics, OpenCV camera-to-world poses and point clouds; torch-free.
COLMAP's camera frame is already OpenCV, so no axis flip is needed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap

from anycam4d.console import notice
from anycam4d.console import spinner
from anycam4d.enums import CameraModelType
from anycam4d.point_ops.ply import read_ply


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Float
    from jaxtyping import UInt8


# All cores load one full-resolution image each and can OOM; a small cap only costs time.
_DEFAULT_COLMAP_THREADS = 4


@dataclass(frozen=True)
class ColmapOptions:
    """COLMAP tunables shared by the importers; none of them changes the stored dataset images."""

    gpu: bool = True
    threads: int = _DEFAULT_COLMAP_THREADS
    max_image_size: int = 3200  # SIFT feature-detection cap (sparse)
    dense: bool = True  # densify the init cloud with MVS (falls back to sparse on failure)
    dense_max_image_size: int = 1600  # MVS working-resolution cap (primary GPU-VRAM lever)


@dataclass(frozen=True)
class KnownPoseView:
    """One image with a fixed COLMAP camera (``model``, ``params``) and world-to-camera ``[R | t]``."""

    name: str
    model: str
    params: list[float]
    width: int
    height: int
    cam_from_world: Float[np.ndarray, "3 4"]


@dataclass(frozen=True)
class ColmapCamera:
    """One camera's intrinsics in anycam4d-native form (model + canonical params)."""

    model: CameraModelType
    width: int
    height: int
    params: list[float]  # anycam4d canonical order [fx, fy, cx, cy, <distortion...>]


@dataclass(frozen=True)
class ColmapResult:
    """The outcome of one SfM run: per-image intrinsics + pose, and the sparse cloud."""

    camera_by_image: dict[str, ColmapCamera]  # image filename -> intrinsics
    c2w_by_image: dict[str, Float[np.ndarray, "4 4"]]  # image filename -> OpenCV camera-to-world
    points_xyz: Float[np.ndarray, "N 3"]
    points_rgb: UInt8[np.ndarray, "N 3"]


@dataclass(frozen=True)
class _ColmapModelMap:
    """How a COLMAP camera model maps onto an anycam4d model + canonical param order."""

    model: CameraModelType
    convert: Callable[[list[float]], list[float]]


# COLMAP camera model -> anycam4d model and positional param converter. Leading slots match:
#   PINHOLE        [fx, fy, cx, cy]                 -> pinhole
#   SIMPLE_PINHOLE [f, cx, cy]                      -> pinhole [f, f, cx, cy]
#   OPENCV         [fx, fy, cx, cy, k1, k2, p1, p2] -> opencv
#   OPENCV_FISHEYE [fx, fy, cx, cy, k1, k2, k3, k4] -> fisheye_opencv
_COLMAP_MODEL_MAP: dict[str, _ColmapModelMap] = {
    "SIMPLE_PINHOLE": _ColmapModelMap(CameraModelType.PINHOLE, lambda p: [p[0], p[0], p[1], p[2]]),
    "PINHOLE": _ColmapModelMap(CameraModelType.PINHOLE, lambda p: [p[0], p[1], p[2], p[3]]),
    "OPENCV": _ColmapModelMap(CameraModelType.OPENCV, lambda p: [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]]),
    "OPENCV_FISHEYE": _ColmapModelMap(
        CameraModelType.FISHEYE_OPENCV, lambda p: [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]]
    ),
}


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    *,
    camera_model: str = "OPENCV",
    camera_mode: str = "PER_IMAGE",
    gpu: bool = True,
    skip_if_done: bool = True,
    num_threads: int = _DEFAULT_COLMAP_THREADS,
    max_image_size: int = 3200,
    isolate: bool = True,
) -> ColmapResult:
    """Run COLMAP SfM over ``image_dir`` and return anycam4d-native calibration.

    Args:
        image_dir: One image per physical camera.
        work_dir: Scratch dir for the database and sparse model.
        camera_model: COLMAP camera model name.
        camera_mode: ``"PER_IMAGE"`` (one camera per image) or ``"SINGLE"``.
        gpu: Use GPU SIFT when this pycolmap build supports it.
        skip_if_done: Reuse an existing ``work_dir/sparse/0``.
        num_threads: Worker threads; fewer lowers the RAM peak (``-1`` = all cores).
        max_image_size: SIFT detection downscale cap.
        isolate: Run in a spawned subprocess so an OOM kill becomes a ``RuntimeError``.

    Raises:
        RuntimeError: If reconstruction fails, is killed, or yields no model.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = work_dir / "sparse"

    rec = _load_existing(sparse_dir) if skip_if_done else None
    if rec is None:
        use_gpu = _resolve_gpu(gpu)
        gpu_index = "0" if use_gpu else "-1"
        log_path = work_dir / "colmap_log.txt"
        args = (
            image_dir,
            work_dir,
            sparse_dir,
            camera_model,
            camera_mode,
            gpu_index,
            num_threads,
            max_image_size,
            log_path,
        )
        device = "GPU" if use_gpu else f"CPU, {num_threads} threads"
        notice(f"COLMAP SfM starting — native output streams to {log_path} (tail it for progress).")
        with spinner(f"COLMAP SfM on {image_dir.name} ({device} — can take several minutes)…"):
            if isolate:
                _run_reconstruct_isolated(args, num_threads, use_gpu, log_path)
            else:
                _reconstruct_worker(*args)
        rec = _load_existing(sparse_dir)
        if rec is None:
            raise RuntimeError(f"COLMAP finished but wrote no reconstruction; see {log_path}.")
    return _reconstruction_to_result(rec)


def _resolve_gpu(gpu: bool) -> bool:
    """Return whether to actually use GPU: requested AND supported by this pycolmap build."""
    if gpu and not pycolmap.has_cuda:
        notice("[yellow]pycolmap was built without CUDA — running COLMAP on CPU.[/]")
        return False
    return gpu


def _run_reconstruct_isolated(args: tuple, num_threads: int, use_gpu: bool, log_path: Path) -> None:
    """Run ``_reconstruct_worker`` in a spawned subprocess; raise cleanly if it dies."""
    proc = mp.get_context("spawn").Process(target=_reconstruct_worker, args=args)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(_colmap_failure_message(proc.exitcode, num_threads, use_gpu, log_path))


def _colmap_failure_message(exitcode: int | None, num_threads: int, use_gpu: bool, log_path: Path) -> str:
    """Human-readable guidance for a failed COLMAP subprocess (hard crash vs ordinary error)."""
    if exitcode is not None and exitcode < 0:
        cause = "out of memory" if -exitcode == 9 else "out of memory, or an internal COLMAP error"
        detail = f"COLMAP crashed (killed by signal {-exitcode} — most likely {cause})."
        gpu_hint = " If on GPU, try --no-gpu (CPU) or a smaller --colmap-max-image-size to fit VRAM." if use_gpu else ""
        remedy = (
            f" Retry with fewer threads (currently {num_threads}), e.g. --colmap-threads 2, and/or a smaller "
            f"--colmap-max-image-size.{gpu_hint} For DeepView you can skip COLMAP entirely with --cloud uniform."
        )
        return f"{detail}{remedy} Full log: {log_path}"
    return f"COLMAP failed (exit code {exitcode}); see {log_path} for details."


def _reconstruct_worker(
    image_dir: Path,
    work_dir: Path,
    sparse_dir: Path,
    camera_model: str,
    camera_mode: str,
    gpu_index: str,
    num_threads: int,
    max_image_size: int,
    log_path: Path,
) -> None:
    """Extract, match and map into ``sparse_dir``; top-level for ``spawn``, logging to ``log_path``."""
    _redirect_output(log_path)
    database_path = work_dir / "database.db"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    extract_opts = pycolmap.SiftExtractionOptions(gpu_index=gpu_index)
    extract_opts.num_threads = num_threads
    extract_opts.max_image_size = max_image_size
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=getattr(pycolmap.CameraMode, camera_mode),
        camera_model=camera_model,
        sift_options=extract_opts,
    )

    match_opts = pycolmap.SiftMatchingOptions(gpu_index=gpu_index)
    match_opts.num_threads = num_threads
    pycolmap.match_exhaustive(database_path=database_path, sift_options=match_opts)

    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.num_threads = num_threads
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
        options=mapper_opts,
    )
    if not maps:
        raise RuntimeError("COLMAP mapper reconstructed no model")


def triangulate_with_known_poses(
    image_dir: Path,
    work_dir: Path,
    views: list[KnownPoseView],
    *,
    camera_model: str = "OPENCV_FISHEYE",
    gpu: bool = True,
    num_threads: int = _DEFAULT_COLMAP_THREADS,
    max_image_size: int = 3200,
    isolate: bool = True,
) -> tuple[Path, Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"]]:
    """Triangulate a sparse cloud into provided poses without running the mapper.

    For known calibration (e.g. DeepView), where wide-FoV SfM tends not to register.

    Returns:
        ``(sparse_model_dir, xyz, rgb)``.

    Raises:
        RuntimeError: If triangulation fails or yields no points.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = work_dir / "sparse"
    log_path = work_dir / "colmap_log.txt"
    use_gpu = _resolve_gpu(gpu)
    gpu_index = "0" if use_gpu else "-1"
    args = (image_dir, work_dir, views, camera_model, gpu_index, num_threads, max_image_size, sparse_dir, log_path)

    notice(f"COLMAP known-pose triangulation — native output streams to {log_path} (tail it for progress).")
    device = "GPU" if use_gpu else f"CPU, {num_threads} threads"
    with spinner(f"Triangulating {len(views)} views into the provided poses ({device})…"):
        if isolate:
            proc = mp.get_context("spawn").Process(target=_triangulate_worker, args=args)
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"known-pose triangulation failed (exit code {proc.exitcode}); see {log_path}")
        else:
            _triangulate_worker(*args)

    rec = pycolmap.Reconstruction(str(sparse_dir))
    xyz, rgb = _points(rec)
    return sparse_dir, xyz, rgb


def _triangulate_worker(
    image_dir: Path,
    work_dir: Path,
    views: list[KnownPoseView],
    camera_model: str,
    gpu_index: str,
    num_threads: int,
    max_image_size: int,
    sparse_dir: Path,
    log_path: Path,
) -> None:
    """Extract + match features, build a fixed-pose model, and triangulate it (runs in subprocess)."""
    _redirect_output(log_path)
    database_path = work_dir / "database.db"
    if database_path.exists():
        database_path.unlink()  # a stale DB would collide with re-extraction

    extract_opts = pycolmap.SiftExtractionOptions(gpu_index=gpu_index)
    extract_opts.num_threads = num_threads
    extract_opts.max_image_size = max_image_size
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model=camera_model,
        sift_options=extract_opts,
    )
    match_opts = pycolmap.SiftMatchingOptions(gpu_index=gpu_index)
    match_opts.num_threads = num_threads
    pycolmap.match_exhaustive(database_path=database_path, sift_options=match_opts)

    rec = pycolmap.Reconstruction()
    for i, view in enumerate(views, start=1):
        rec.add_camera(
            pycolmap.Camera(
                model=view.model, width=view.width, height=view.height, params=list(view.params), camera_id=i
            )
        )
        cam_from_world = pycolmap.Rigid3d(np.asarray(view.cam_from_world, dtype=np.float64))
        rec.add_image(pycolmap.Image(name=view.name, camera_id=i, cam_from_world=cam_from_world, image_id=i))
        rec.register_image(i)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    result = pycolmap.triangulate_points(rec, str(database_path), str(image_dir), str(sparse_dir), options=opts)
    if result.num_points3D() == 0:
        raise RuntimeError("known-pose triangulation produced no points (no feature matches?)")


def run_dense_mvs(
    sparse_model_dir: Path,
    image_dir: Path,
    work_dir: Path,
    *,
    gpu: bool = True,
    max_image_size: int = 1600,
    cache_size: float = 16.0,
    num_threads: int = _DEFAULT_COLMAP_THREADS,
    isolate: bool = True,
    skip_if_done: bool = True,
) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None]:
    """Dense cloud from a sparse model: undistort, GPU patch-match, fusion.

    CUDA only. ``max_image_size`` and ``cache_size`` (GB) bound VRAM and RAM; ``skip_if_done`` reuses
    ``work_dir/dense/fused.ply``; ``isolate`` turns a crash into a ``RuntimeError`` so callers can fall back
    to the sparse cloud.

    Returns:
        ``(xyz, rgb)`` of the fused cloud.
    """
    dense_dir = work_dir / "dense"
    fused_ply = dense_dir / "fused.ply"
    log_path = work_dir / "dense_log.txt"

    if skip_if_done and fused_ply.exists():
        notice(f"[yellow]Dense MVS: reusing existing cloud {fused_ply} (skip).[/]")
        return _filter_radial_outliers(*read_ply(fused_ply))

    if not pycolmap.has_cuda:
        raise RuntimeError("dense MVS needs a CUDA-enabled pycolmap (patch_match_stereo is GPU-only)")
    if not gpu:
        raise RuntimeError("dense MVS requires GPU; it cannot run on CPU")

    dense_dir.mkdir(parents=True, exist_ok=True)
    notice(f"Dense MVS starting — native COLMAP output streams to {log_path} (tail it for progress).")

    # Three stages, each an isolated subprocess with its own spinner, so the user sees which step is
    # running and a crash is attributed to the exact stage (with the right remedy).
    _run_dense_stage(
        "Undistorting images",
        "check the log — undistortion rarely fails (usually a bad sparse model or missing images)",
        _undistort_worker,
        (sparse_model_dir, image_dir, dense_dir, max_image_size, log_path, False),
        log_path,
        isolate,
    )
    _run_dense_stage(
        "Patch-match stereo (GPU — the slow step, often many minutes)",
        f"usually out of GPU VRAM — lower --dense-max-image-size (currently {max_image_size}) or --no-gpu",
        _patch_match_worker,
        (dense_dir, max_image_size, cache_size, log_path, True),
        log_path,
        isolate,
    )
    _run_dense_stage(
        "Fusing depth maps into a dense cloud",
        "usually out of host RAM or disk — close other memory-heavy jobs and use a work-dir with space",
        _fusion_worker,
        (dense_dir, fused_ply, max_image_size, cache_size, num_threads, log_path, True),
        log_path,
        isolate,
    )
    if not fused_ply.exists():
        raise RuntimeError(f"dense MVS produced no cloud; see {log_path}")
    xyz, rgb = read_ply(fused_ply)
    return _filter_radial_outliers(xyz, rgb)


def _filter_radial_outliers(
    xyz: Float[np.ndarray, "N 3"],
    rgb: UInt8[np.ndarray, "N 3"] | None,
    *,
    k: float = 3.0,
    base_percentile: float = 95.0,
) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"] | None]:
    """Keep points within ``k`` times the ``base_percentile`` radius, dropping patch-match points near infinity."""
    if xyz.shape[0] == 0:
        return xyz, rgb
    center = np.median(xyz, axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    keep = radius <= k * float(np.percentile(radius, base_percentile))
    n_dropped = int((~keep).sum())
    if n_dropped:
        notice(f"Dense cloud: dropped {n_dropped:,} radial outliers ({int(keep.sum()):,} points kept).")
    return xyz[keep], (rgb[keep] if rgb is not None else None)


def _run_dense_stage(label: str, hint: str, target: Callable, args: tuple, log_path: Path, isolate: bool) -> None:
    """Run one dense stage under its own spinner, isolated; raise a stage-attributed error if it dies."""
    with spinner(f"{label}…"):
        if not isolate:
            target(*args)
            return
        proc = mp.get_context("spawn").Process(target=target, args=args)
        proc.start()
        proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(_dense_failure_message(label, hint, proc.exitcode, log_path))


def _dense_failure_message(label: str, hint: str, exitcode: int | None, log_path: Path) -> str:
    """Stage-attributed guidance for a failed dense-MVS stage."""
    how = f"killed by signal {-exitcode}" if (exitcode is not None and exitcode < 0) else f"exit code {exitcode}"
    return f"Dense MVS failed during '{label}' ({how}) — {hint}. Log: {log_path}"


def _undistort_worker(
    sparse_model_dir: Path, image_dir: Path, dense_dir: Path, max_image_size: int, log_path: Path, append: bool
) -> None:
    """Create the undistorted (pinhole) MVS workspace from the sparse model (runs in subprocess)."""
    _redirect_output(log_path, append=append)
    undistort_opts = pycolmap.UndistortCameraOptions()
    undistort_opts.max_image_size = max_image_size
    pycolmap.undistort_images(
        output_path=str(dense_dir),
        input_path=str(sparse_model_dir),
        image_path=str(image_dir),
        undistort_options=undistort_opts,
    )


def _patch_match_worker(dense_dir: Path, max_image_size: int, cache_size: float, log_path: Path, append: bool) -> None:
    """GPU patch-match stereo depth estimation over the undistorted workspace (runs in subprocess)."""
    _redirect_output(log_path, append=append)
    patch_opts = pycolmap.PatchMatchOptions()
    patch_opts.gpu_index = "0"
    patch_opts.max_image_size = max_image_size
    patch_opts.cache_size = cache_size
    pycolmap.patch_match_stereo(workspace_path=str(dense_dir), options=patch_opts)


def _fusion_worker(
    dense_dir: Path,
    fused_ply: Path,
    max_image_size: int,
    cache_size: float,
    num_threads: int,
    log_path: Path,
    append: bool,
) -> None:
    """Fuse the per-view depth maps into a single dense point cloud at ``fused_ply`` (runs in subprocess)."""
    _redirect_output(log_path, append=append)
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.max_image_size = max_image_size
    fusion_opts.num_threads = num_threads
    fusion_opts.cache_size = cache_size
    fusion_opts.use_cache = True  # stream depth/normal maps through a bounded cache, not all into RAM
    pycolmap.stereo_fusion(output_path=str(fused_ply), workspace_path=str(dense_dir), options=fusion_opts)


def _redirect_output(log_path: Path, *, append: bool = False) -> None:
    """Redirect this process's stdout+stderr (fd level, so native output too) to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(log_path, flags, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def _load_existing(sparse_dir: Path) -> pycolmap.Reconstruction | None:
    """Load a previously written reconstruction from ``sparse/0``, or ``None`` if absent."""
    model_dir = sparse_dir / "0"
    if (model_dir / "cameras.bin").exists() or (model_dir / "cameras.txt").exists():
        return pycolmap.Reconstruction(str(model_dir))
    return None


def _reconstruction_to_result(rec: pycolmap.Reconstruction) -> ColmapResult:
    """Read intrinsics, poses, and points out of a ``pycolmap.Reconstruction``."""
    camera_by_image: dict[str, ColmapCamera] = {}
    c2w_by_image: dict[str, Float[np.ndarray, "4 4"]] = {}
    for image in rec.images.values():
        camera = rec.cameras[image.camera_id]
        camera_by_image[image.name] = _colmap_camera_to_anycam(camera)
        c2w_by_image[image.name] = c2w_from_cam_from_world(image.cam_from_world)

    xyz, rgb = _points(rec)
    return ColmapResult(camera_by_image=camera_by_image, c2w_by_image=c2w_by_image, points_xyz=xyz, points_rgb=rgb)


def _colmap_camera_to_anycam(camera: pycolmap.Camera) -> ColmapCamera:
    """Map a COLMAP camera to an anycam4d ``ColmapCamera`` (model + canonical params)."""
    name = camera.model.name
    mapping = _COLMAP_MODEL_MAP.get(name)
    if mapping is None:
        raise ValueError(f"Unsupported COLMAP camera model '{name}'; add it to _COLMAP_MODEL_MAP")
    params = mapping.convert([float(x) for x in camera.params])
    return ColmapCamera(model=mapping.model, width=int(camera.width), height=int(camera.height), params=params)


def c2w_from_cam_from_world(cam_from_world: pycolmap.Rigid3d) -> Float[np.ndarray, "4 4"]:
    """OpenCV camera-to-world matrix ``[[Rᵀ, -Rᵀt], [0, 1]]`` from COLMAP's world-to-camera ``Rigid3d``."""
    r_w2c = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    t = np.asarray(cam_from_world.translation, dtype=np.float64)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r_w2c.T
    c2w[:3, 3] = -r_w2c.T @ t
    return c2w


def _points(rec: pycolmap.Reconstruction) -> tuple[Float[np.ndarray, "N 3"], UInt8[np.ndarray, "N 3"]]:
    """Extract the sparse cloud: xyz (float32) + rgb (uint8), possibly empty ``(0, 3)``."""
    points = rec.points3D
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    xyz = np.array([p.xyz for p in points.values()], dtype=np.float32)
    rgb = np.array([p.color for p in points.values()], dtype=np.uint8)
    return xyz, rgb
