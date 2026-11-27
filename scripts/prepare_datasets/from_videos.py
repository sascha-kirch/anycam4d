r"""Build an AnyCam4D dataset from your own synchronized multi-camera videos.

One video per fixed camera, all synchronized (frame ``i`` is the same instant in every video) and at one frame rate;
trim them to a common start first if they are not. Cameras are numbered 0..N-1 in file-name order. COLMAP calibrates
the rig on one frame (intrinsics in the chosen lens model, poses, init point cloud), ffmpeg extracts the frames, and
the result trains directly: ``pixi run train dataset.data_dir=<out>``.

Usage::

    pixi run prepare-videos --videos path/to/videos --out data/my_scene \
        --camera-model opencv --eval-cams 0 [--fps 30] [--start-frame 0] [--calib-frame 120] [--max-frames 300]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anycam4d.console import notice
from anycam4d.data.prep.build_dataset import build_dataset
from anycam4d.data.prep.colmap import ColmapOptions
from anycam4d.data.prep.importers.video_rig import COLMAP_MODEL
from anycam4d.data.prep.importers.video_rig import VideoRigImporter
from anycam4d.data.prep.provenance import build_manifest
from anycam4d.enums import CameraModelType


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True, help="directory with one video per camera")
    parser.add_argument("--out", type=Path, required=True, help="dataset directory to write")
    parser.add_argument(
        "--camera-model",
        choices=[m.value for m in COLMAP_MODEL],
        default=CameraModelType.OPENCV.value,
        help="lens model COLMAP estimates: pinhole, opencv (radial-tangential) or fisheye_opencv (default: opencv)",
    )
    parser.add_argument(
        "--shared-intrinsics", action="store_true", help="identical cameras and settings: one set of intrinsics"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="maximum output frame rate (default: 30)")
    parser.add_argument("--start-frame", type=int, default=0, help="first source frame (default: 0)")
    parser.add_argument("--max-frames", type=int, default=None, help="cap on output frames (default: all)")
    parser.add_argument(
        "--calib-frame", type=int, default=None, help="source frame COLMAP calibrates on (default: --start-frame)"
    )
    parser.add_argument("--eval-cams", type=int, nargs="*", default=[], help="camera ids held out for evaluation")
    parser.add_argument("--exclude-cams", type=int, nargs="*", default=[], help="camera ids to leave out entirely")
    parser.add_argument("--no-dense", action="store_true", help="keep the sparse COLMAP cloud (skip dense MVS)")
    parser.add_argument("--cpu", action="store_true", help="run COLMAP on the CPU")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="scratch dir for COLMAP (default: <out>_work, reused on rerun)"
    )
    return parser.parse_args()


def _check_extraction_settings(args: argparse.Namespace) -> None:
    """Refuse to resume into a dataset extracted with other frame settings: finished cameras are not re-extracted."""
    settings = {
        "videos": str(args.videos.resolve()),
        "start_frame": args.start_frame,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "exclude_cams": sorted(args.exclude_cams),
        "image_format": args.image_format,
    }
    marker = args.out / ".extraction.json"
    if marker.exists() and json.loads(marker.read_text()) != settings:
        raise SystemExit(f"{args.out} was extracted with other settings; pass a new --out or delete it.")
    if not marker.exists() and args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and was not written by this script; pass a new --out.")
    args.out.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(settings, indent=2) + "\n")


def main() -> None:
    """Calibrate, extract and write one dataset."""
    args = _parse_args()
    _check_extraction_settings(args)
    importer = VideoRigImporter(
        args.videos,
        args.work_dir if args.work_dir is not None else args.out.with_name(f"{args.out.name}_work"),
        camera_model=CameraModelType(args.camera_model),
        shared_intrinsics=args.shared_intrinsics,
        calib_frame=args.calib_frame,
        start_frame=args.start_frame,
        fps_cap=args.fps,
        exclude_cams=args.exclude_cams,
        max_frames=args.max_frames,
        colmap=ColmapOptions(gpu=not args.cpu, dense=not args.no_dense),
    )
    notice("cameras: " + ", ".join(f"{cid}={path.name}" for cid, path in importer.video_paths.items()))
    unknown = sorted(set(args.eval_cams) - importer.video_paths.keys())
    if unknown:
        raise SystemExit(f"--eval-cams {unknown} are not among the cameras above (excluded or missing).")
    manifest = build_manifest(
        source=f"videos/{args.videos.resolve().name}",
        importer=VideoRigImporter.__name__,
        cli_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    manifest["videos"] = {cid: path.name for cid, path in importer.video_paths.items()}
    build_dataset(importer, args.out, eval_cams=args.eval_cams, image_format=args.image_format, manifest=manifest)
    notice(f"[green]Dataset written to {args.out}[/]\n  pixi run train dataset.data_dir={args.out}")


if __name__ == "__main__":
    main()
