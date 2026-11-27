"""VariantSpec and build_variant: derive a downscaled or undistorted dataset from a base one.

One thread per camera: decode, remap and encode run inside OpenCV, which releases the GIL, and a camera's resample
plan is built once and reused for its whole sequence. OpenCV's own threading is off (``cv2.setNumThreads(0)``) so
the pool is the only parallelism; peak memory is one frame per worker.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2

from anycam4d.console import progress_bar
from anycam4d.data.prep.build import compose
from anycam4d.data.prep.provenance import build_manifest
from anycam4d.data.prep.source import Anycam4DSource
from anycam4d.data.prep.source import load_view
from anycam4d.data.prep.writer import PrepWriter


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from pathlib import Path

    import numpy as np
    from jaxtyping import Float

    from anycam4d.data.prep.sample import CalibratedView
    from anycam4d.data.prep.source import CaptureRef
    from anycam4d.data.prep.stream import StageProto
    from anycam4d.enums import CameraModelType


@dataclass
class VariantSpec:
    """Names a post-hoc variant: the stage pipeline to apply and how to write it out."""

    name: str
    stages: list[StageProto]
    image_format: str = "webp"
    eval_cams: Iterable[int] | None = None  # None -> inherit the base source's eval split


@dataclass(frozen=True)
class _CaptureRecord:
    """The post-transform metadata for one written capture (pixels already flushed to disk).

    Returned from a worker so the main thread can assemble the dataset JSON without holding any
    image in memory — the calibration is post-transform (undistorting rewrites the model/params).
    """

    cam_id: int
    frame_id: int
    model: CameraModelType
    params: list[float]
    width: int
    height: int
    reproject_source: dict | None
    max_incidence_deg: float | None
    timestamp: float
    c2w: Float[np.ndarray, "4 4"]
    image_path: str


def build_variant(base_dir: Path, out_dir: Path, spec: VariantSpec, jobs: int) -> None:
    """Read ``base_dir``, apply ``spec.stages`` to every capture, and write the variant to ``out_dir``.

    Args:
        base_dir: Root of an already-built base AnyCam4D dataset.
        out_dir: Root for the derived variant (created if absent).
        spec: Stage pipeline, image format, and eval-camera split for the variant.
        jobs: Number of concurrent per-camera worker threads (the renderer's core budget).
    """
    src = Anycam4DSource(base_dir)
    eval_cams = spec.eval_cams if spec.eval_cams is not None else src.eval_camera_ids
    writer = PrepWriter(out_dir, eval_cams=eval_cams, image_format=spec.image_format)

    _render(src, spec.stages, writer, label=spec.name, jobs=jobs)

    points = src.point_cloud()
    xyz, colors = points if points is not None else (None, None)
    manifest = build_manifest(
        source=str(base_dir),
        importer="build_variant",
        cli_args={},
        variant=spec.name,
        stages=[type(stage).__name__ for stage in spec.stages],
    )
    writer.finalize(xyz, colors, manifest)


def _render(source: Anycam4DSource, stages: list[StageProto], writer: PrepWriter, label: str, jobs: int) -> None:
    """Render every capture through ``stages`` in parallel (one worker per camera) into ``writer``.

    Images are encoded concurrently by ``jobs`` worker threads; the dataset metadata is assembled on
    the main thread in a deterministic (frame, camera) order so the output JSON is reproducible run
    to run. OpenCV's own threading is disabled so the pool is the single source of parallelism and
    the two cannot oversubscribe the cores (the cause of a machine-freezing over-commit).
    """
    cv2.setNumThreads(0)  # the thread pool is the only parallelism; keep OpenCV from oversubscribing
    refs_by_camera = source.camera_refs()
    n_captures = sum(len(refs) for refs in refs_by_camera.values())
    max_workers = max(1, min(jobs, len(refs_by_camera)))

    records: list[_CaptureRecord] = []
    with progress_bar(f"{label} · {len(refs_by_camera)} cams", n_captures, unit="captures") as advance:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_render_camera, refs, stages, writer, advance) for refs in refs_by_camera.values()]
            for future in as_completed(futures):
                records.extend(future.result())

    seen: set[int] = set()
    for rec in sorted(records, key=lambda r: (r.frame_id, r.cam_id)):
        if rec.cam_id not in seen:
            writer.register_camera(
                rec.cam_id,
                rec.model,
                rec.params,
                rec.width,
                rec.height,
                rec.reproject_source,
                rec.max_incidence_deg,
            )
            seen.add(rec.cam_id)
        writer.add_capture(rec.cam_id, rec.frame_id, rec.timestamp, rec.c2w, rec.image_path)


def _render_camera(
    refs: list[CaptureRef],
    stages: list[StageProto],
    writer: PrepWriter,
    advance: Callable[[int], None],
) -> list[_CaptureRecord]:
    """Worker: decode, transform, and encode every capture of one camera; return their metadata.

    Runs in a pool thread on its own copy of the stages, so a stage's per-camera resample plan
    (the undistort remap grid) is built once for this camera, reused across its frames, and then freed
    when the worker returns — capping plan memory at the worker count instead of the total camera
    count. Only one decoded frame is held at a time, and the returned records carry no pixels, so a
    camera's whole sequence streams through with bounded memory.
    """
    transform: Callable[[CalibratedView], CalibratedView] = compose(copy.deepcopy(stages))
    records: list[_CaptureRecord] = []
    for ref in refs:
        view = transform(load_view(ref))
        image_path = writer.encode_view(view)
        records.append(
            _CaptureRecord(
                cam_id=view.cam_id,
                frame_id=view.frame_id,
                model=view.model,
                params=view.params,
                width=view.image.shape[1],
                height=view.image.shape[0],
                reproject_source=view.reproject_source,
                max_incidence_deg=view.max_incidence_deg,
                timestamp=view.timestamp,
                c2w=view.c2w,
                image_path=image_path,
            )
        )
        advance(1)
    return records
