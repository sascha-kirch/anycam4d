#!/usr/bin/env bash
# Train and evaluate every cell of the main comparison, {EWA, AnyCam4D} x {undistorted pinholes, native lens}: arms
# ewa_ud, ewa_nat, ut_ud and ut_nat (ut = AnyCam4D), 21 scenes x 3 seeds, run one after another.
#
#   DATA_ROOT=data OUTPUT_ROOT=runs bash scripts/reproduce/main_tables.sh
#   DRY_RUN=1 SCENES="cs flam" SEEDS="1" bash scripts/reproduce/main_tables.sh     # print the commands only
#
# Expects datasets built with `--out-root $DATA_ROOT/<dataset>` and Dynopsis downloaded to $DATA_ROOT/dynopsis.
# Each run is evaluated on its own split; the undistorted arms are also scored through the native camera, and SelfCap
# additionally against the undistorted test images with LPIPS-VGG. Scores land in <run>/eval*/metrics.json.
#
# mu_t_lr and temporal_spread_init follow the published 4DGS regime of each scene (scripts/og4dgs_regime.py).
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}/main_tables"
SEEDS="${SEEDS:-1 2 3}"
run() { echo "+ $*"; [[ -n "${DRY_RUN:-}" ]] || "$@"; }

BASE=(--config-name 4dgs training.max_steps=30000)
# tag · dataset · undistorted variant · native variant · opacity reset · near plane · accum_renders · sampler
TABLE=(
    "cs    dynerf   cook_spinach_undistorted_r2        cook_spinach_r2            3000  0.01 4 temporal_tiled"
    "fs    dynerf   flame_steak_undistorted_r2         flame_steak_r2             10000 0.01 4 temporal_tiled"
    "ss    dynerf   sear_steak_undistorted_r2          sear_steak_r2              3000  0.01 4 temporal_tiled"
    "flam  deepview flames_undistorted_r2              flames_r2                  3000  0.01 4 uniform"
    "truck deepview truck_undistorted_r2               truck_r2                   3000  0.01 2 uniform"
    "weld  deepview welder_undistorted_r2              welder_r2                  3000  0.01 4 uniform"
    "dykb  dynopsis scene1/fisheye_kb_pinhole/count_32 scene1/fisheye_kb/count_32 3000  0.01 4 uniform"
    "dycv  dynopsis scene1/opencv_pinhole/count_32     scene1/opencv/count_32     3000  0.01 4 uniform"
    "fbk1  selfcap  ftgs_bike1_undistorted             ftgs_bike1                 3000  0.1  4 uniform"
    "fbk2  selfcap  ftgs_bike2_undistorted             ftgs_bike2                 3000  0.1  4 uniform"
    "fco1  selfcap  ftgs_corgi1_undistorted_r2         ftgs_corgi1_r2             3000  0.1  4 uniform"
    "fco2  selfcap  ftgs_corgi2_undistorted_r2         ftgs_corgi2_r2             3000  0.1  4 uniform"
    "fda1  selfcap  ftgs_dance1_undistorted_r2         ftgs_dance1_r2             3000  0.1  4 uniform"
    "fda2  selfcap  ftgs_dance2_undistorted_r2         ftgs_dance2_r2             3000  0.1  4 uniform"
    "cm    dynerf   coffee_martini_undistorted_r2      coffee_martini_r2          3000  0.01 4 temporal_tiled"
    "crb   dynerf   cut_roasted_beef_undistorted_r2    cut_roasted_beef_r2        3000  0.01 4 temporal_tiled"
    "fsal  dynerf   flame_salmon_1_undistorted_r2      flame_salmon_1_r2          3000  0.01 4 temporal_tiled"
    "exh   deepview exhibit_undistorted_r2             exhibit_r2                 3000  0.01 4 uniform"
    "fp1   deepview face_paint_1_undistorted_r2        face_paint_1_r2            3000  0.01 4 uniform"
    "fp2   deepview face_paint_2_undistorted_r2        face_paint_2_r2            3000  0.01 4 uniform"
    "cave  deepview cave_undistorted_r2                cave_r2                    3000  0.01 4 uniform"
)

for entry in "${TABLE[@]}"; do
    read -r tag dataset ud nat reset near accum sampler <<< "${entry}"
    [[ -z "${SCENES:-}" || " ${SCENES} " == *" ${tag} "* ]] || continue
    ud_dir="${DATA_ROOT}/${dataset}/${ud}"
    nat_dir="${DATA_ROOT}/${dataset}/${nat}"
    overrides="$(pixi run -q python scripts/og4dgs_regime.py "${ud_dir}" --overrides)"
    read -r -a regime <<< "${overrides}"
    common=("${regime[@]}" strategy.reset_every="${reset}"
        training.accum_renders="${accum}" rendering.near_plane="${near}" training.sampler="${sampler}")
    near_eval=()
    [[ "${dataset}" == selfcap ]] && near_eval=(--set near_plane=0.3)
    for seed in ${SEEDS}; do
        for arm in ewa_ud ewa_nat ut_ud ut_nat; do
            name="${arm}_${tag}_${seed}"
            data="${ud_dir}" && [[ "${arm}" == *_nat ]] && data="${nat_dir}"
            rendering=() && [[ "${arm}" == ut_* ]] && rendering=(rendering=anycam4d)
            run pixi run train "${BASE[@]}" "${rendering[@]}" dataset.data_dir="${data}" training.seed="${seed}" \
                "${common[@]}" hydra.run.dir="${OUTPUT_ROOT}/${name}"
            ckpt="${OUTPUT_ROOT}/${name}/checkpoints/last.pt"
            run pixi run eval -c "${ckpt}" "${near_eval[@]}" -o "${OUTPUT_ROOT}/${name}/eval/metrics.json"
            if [[ "${arm}" == *_ud ]]; then
                run pixi run eval -c "${ckpt}" "${near_eval[@]}" -d "${nat_dir}" \
                    -o "${OUTPUT_ROOT}/${name}/eval_native/metrics.json"
            fi
            if [[ "${dataset}" == selfcap ]]; then
                run pixi run eval -c "${ckpt}" "${near_eval[@]}" -d "${ud_dir}" --lpips_net vgg \
                    -o "${OUTPUT_ROOT}/${name}/eval_vgg/metrics.json"
            fi
        done
    done
done
