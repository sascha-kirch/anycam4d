# Dataset preparation

How to turn a multi-view video dataset into an AnyCam4D training dataset. Every command runs through `pixi run`
from the repository root. The on-disk format itself (JSON schema, coordinate conventions, camera models) is
described in the [README](../README.md#dataset-format).

| Dataset | Native camera model | How | Registry |
|---|---|---|---|
| **DyNeRF** (Neural 3D Video) | `opencv` | download + build | [`conf/datasets/dynerf.yaml`](../conf/datasets/dynerf.yaml) |
| **SelfCap** | `opencv` | download (gated) + build | [`conf/datasets/selfcap.yaml`](../conf/datasets/selfcap.yaml) |
| **DeepView** (Immersive Light Field Video) | `fisheye_opencv` | download + build | [`conf/datasets/deepview.yaml`](../conf/datasets/deepview.yaml) |
| **Dynopsis** (ours, synthetic) | `opencv`, `fisheye_opencv` | download only, ships prepared | [`conf/datasets/dynopsis.yaml`](../conf/datasets/dynopsis.yaml) |
| **Your own videos** | `pinhole`, `opencv`, `fisheye_opencv` | `prepare-videos` (§4) | — |

A build has two stages per scene:

```
raw videos ─▶ base (<scene>/)                       ─▶ variants (<scene>_r2/, <scene>_undistorted/, …)
              ffmpeg frames + COLMAP calibration       downscaled and/or undistorted copies of the base
              + MVS point cloud, full resolution
```

---

## 1. Licences

This repository redistributes no third-party data. `download` fetches each dataset from its original home onto your
machine and prints its licence line first; the prepared output stays local. SelfCap is licensed for non-commercial
research only; DyNeRF and DeepView come with their own terms, which you accept at the source. Dynopsis is released
under CC BY 4.0.

---

## 2. Download

```bash
pixi run download --registry conf/datasets/dynerf.yaml --scene flame_steak --dest ~/datasets/DyNeRF
pixi run download --registry conf/datasets/deepview.yaml --dest ~/datasets/DeepView      # every scene
```

Sources: DyNeRF from its GitHub release, SelfCap from Hugging Face, DeepView from per-scene URLs. A build fetches a
missing scene by itself, so `download` is optional before `build-*`.

**SelfCap is a gated Hugging Face dataset.** Before the first download, log in (`hf auth login`, a token with read
access) and accept the terms on the *Files and versions* tab of
[zju3dv/SelfCap-Dataset](https://huggingface.co/datasets/zju3dv/SelfCap-Dataset). Until both are done every request
fails as unauthorized; the fetcher stops and names the missing step.

**Dynopsis ships prepared** ([sascha-kirch/dynopsis](https://huggingface.co/datasets/sascha-kirch/dynopsis)). Each
registry scene is one image set of `scene1` (`opencv`, `opencv_pinhole`, `pinhole`, `fisheye_kb`,
`fisheye_kb_pinhole`, `rotated_gt`), stored as one tar. `download` fetches and unpacks it, and the set trains as is:

```bash
pixi run download --registry conf/datasets/dynopsis.yaml --scene fisheye_kb --dest data/dynopsis
pixi run train --config-name anycam4d dataset.data_dir=data/dynopsis/scene1/fisheye_kb/count_32
```

`rotated_gt` holds ground truth only: the held-out fisheye cameras rotated about their centres, for evaluating
renders of models trained on `fisheye_kb` at camera poses no training camera had.

---

## 3. Build a benchmark dataset

```bash
pixi run build-dynerf   --source-root ~/datasets/DyNeRF   --out-root data/4dgs
pixi run build-selfcap  --source-root ~/datasets/SelfCap  --out-root data/4dgs
pixi run build-deepview --source-root ~/datasets/DeepView --out-root data/4dgs
```

Each `build-*` task is `build-variants --registry conf/datasets/<dataset>.yaml`; append `--scene NAME` for one scene.
Each scene is expected under `<source-root>/<source_subdir>`.

### 3.1 The registry

The registry YAML holds everything scene-specific, so a change needs no code edit:

```yaml
scenes:
  flame_steak:
    source_subdir: flame_steak      # raw directory under --source-root
    fps: 30                          # output frame-rate cap; faster sources are decimated
    eval_cams: [0]                   # camera ids of the eval split
    excluded_cams: []                # cameras dropped before calibration
    variants: [native, undistorted]
    scales: [2]                      # resolution tiers each variant is built at (2 = half)
```

Optional fields: `start_frame` / `end_frame` (trim, used by SelfCap), `max_frames`, `calib_dir` (SelfCap releases
that ship calibration separately), `split_parts` (multi-part zip archives).

### 3.2 Variants

| Variant | Output directory | Content |
|---|---|---|
| `native` | `<scene>` (base), `<scene>_r2` | the base calibration, downscaled at each scale ≠ 1 |
| `undistorted` | `<scene>_undistorted`, `<scene>_undistorted_r2` | reprojected to a pinhole; skipped if the base already is one |

Undistortion cannot fill the whole pinhole frame: a fisheye loses its periphery, and the corners of the target map
outside the source image. Those pixels were never observed, so every reprojected camera records its native lens in
a `reproject_source` block. At load the warp is replayed and the unfilled region is excluded from the loss and from
evaluation instead of being supervised as black.

### 3.3 Which backend trains which variant

| Variant | Camera model | Backends | Preset |
|---|---|---|---|
| `native` | `opencv`, `fisheye_opencv` | `anycam4d` or `ewa` | [`conf/anycam4d.yaml`](../conf/anycam4d.yaml), [`conf/4dgs.yaml`](../conf/4dgs.yaml) |
| `undistorted` | `pinhole` | `ewa` (the published 4DGS setting) or `anycam4d` | [`conf/4dgs.yaml`](../conf/4dgs.yaml) |

Both backends project through the distorted lens: EWA projects each mean exactly and linearizes the covariance
there, AnyCam4D pushes sigma points through the lens (unscented transform).

```bash
pixi run train --config-name anycam4d dataset.data_dir=data/4dgs/flame_steak_r2
pixi run train --config-name 4dgs     dataset.data_dir=data/4dgs/flame_steak_undistorted_r2
```

### 3.4 Resuming

`<out-root>/<scene>.prep_state.json` records each finished stage with a hash of its inputs. A rerun skips a stage
whose marker is set, whose inputs are unchanged and whose output directory exists, so after a crash (network,
COLMAP, out of memory) rerun the same command. A rebuilt base invalidates its variants.

| Flag | Effect |
|---|---|
| *(none)* | resume, skipping finished stages |
| `--force` | rebuild every stage |
| `--from base\|variants` | rebuild from that stage onward |
| `--only <stage>` | run one stage: `base`, `variants`, or a variant name such as `undistorted` |
| `--dry-run` | print the planned work and exit |
| `--jobs N` | worker threads for the variant stage (default: CPU count − 2) |

COLMAP is the memory-hungry step. `--colmap-threads` (fewer threads, less RAM), `--colmap-max-image-size` (feature
extraction resolution), `--dense-max-image-size` (MVS resolution, the main VRAM lever), `--no-gpu` and `--no-dense`
(keep the sparse cloud; also the automatic fallback when MVS fails). COLMAP scratch files live in
`<out-root>/<scene>_work` and are deleted once the base is built.

---

## 4. Your own synchronized videos

A static rig of your own (one video per fixed camera) becomes a dataset in one command. COLMAP calibrates the rig
on one frame (intrinsics in the chosen lens model, poses, the initial point cloud, densified by MVS), ffmpeg
extracts the frames, and the result trains directly:

```bash
pixi run prepare-videos --videos path/to/videos --out data/my_scene --camera-model opencv --eval-cams 0
pixi run train --config-name anycam4d dataset.data_dir=data/my_scene
```

- **Synchronized, one frame rate.** Frame `i` must be the same instant in every video; trim the videos to a common
  start first if they are not (a clap or a light flash is an easy marker). Mixed frame rates are rejected.
- **Cameras** are numbered 0..N−1 in file-name order; the mapping is printed and recorded in `prep_manifest.json`.
  `--eval-cams` holds cameras out for evaluation, `--exclude-cams` drops broken ones before COLMAP sees them.
- **Lens model** (`--camera-model`): `opencv` (radial-tangential, the default) suits most action and machine-vision
  cameras, `fisheye_opencv` wide-angle and fisheye lenses, `pinhole` footage that is already undistorted. Add
  `--shared-intrinsics` when all cameras are the same model with the same lens and settings: one set of intrinsics
  constrains COLMAP far better than one per camera.
- **Calibration frame** (`--calib-frame`, default `--start-frame`): pick a sharp, well-textured frame with little
  motion. If COLMAP leaves a camera unregistered, the run stops and names it; try another frame or exclude it.
- **Time range**: `--start-frame`, `--max-frames`, and `--fps` (a cap: a 60 fps source is decimated to 30).
- `--no-dense` skips MVS (minutes instead of tens of minutes), `--cpu` runs COLMAP without a GPU.

COLMAP's scratch files go to `<out>_work` and are reused on a rerun.

---

## 5. Hand-assembled data

A camera written as a bare `pinhole` with neither a `reproject_source` block nor a `valid/` mask is ambiguous: it
could be a native pinhole (every pixel observed) or a reprojected one with a lost periphery. The loader assumes the
first and logs a warning naming the camera. If that camera was in fact reprojected, either add a `reproject_source`
block describing the native lens (exact and replayable) or provide `valid/<camera_id>.png` (nonzero = observed),
which overrides everything. Datasets built by this guide never take this path.

---

## 6. Output layout

```
data/4dgs/
  flame_steak/                    # base: native calibration, full resolution
    train_dataset.json
    eval_dataset.json
    images/
    points3d.ply                  # COLMAP / MVS initial point cloud
    prep_manifest.json            # provenance
  flame_steak_r2/
  flame_steak_undistorted_r2/
  flame_steak.prep_state.json     # resume markers
```

A DyNeRF or DeepView scene (18–46 cameras × ~300 frames) takes a few GB at full resolution; a half-resolution
variant takes about a quarter of that.

**Provenance.** `prep_manifest.json` records the source (`dataset/scene`), its URL and a hash of the fetched files,
the importer, the anycam4d commit, the arguments, and the tool versions (`pycolmap`, `opencv`, `numpy`).

**Reproducibility.** COLMAP runs in-process through the CUDA build of `pycolmap`. SfM and MVS are not
bit-reproducible, so two builds of one scene differ slightly in calibration and point cloud; the point cloud only
initializes training.

---

## 7. Limitations

**Every frame is seen by the same cameras.** The writer refuses a sequence whose frames are seen by different camera
sets. Consequences:

- No mixed frame rates and no per-camera dropped frames. A frame dropped by every camera is fine.
- The cameras of a frame may carry different timestamps (sub-frame sync offsets): each image is rendered at its own
  time. Only `training.sampler=frame_aligned`, which renders all cameras of a frame in one pass at one time, refuses
  such a dataset. The SelfCap importer does not apply `optimized/sync.json` (up to ±0.04 s), so its builds match the
  paper.
- A broken camera can be excluded for the whole sequence (`excluded_cams`), not for part of it.
