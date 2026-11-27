#!/usr/bin/env bash
# Our EWA against the published 4DGS on DyNeRF: EWA on undistorted pinholes with our two departures from 4DGS
# (additive temporal harmonics, selective Adam), and with each replaced by the published choice. 3 scenes x 3 seeds.
#
#   DATA_ROOT=data OUTPUT_ROOT=runs bash scripts/reproduce/fidelity_table.sh
#   DRY_RUN=1 SCENES="fs" SEEDS="1" bash scripts/reproduce/fidelity_table.sh     # print the commands only
#
# Expects DyNeRF built with `--out-root $DATA_ROOT/dynerf`. PSNR on the held-out camera at the last step lands in
# <run>/eval/metrics.json; the published 4DGS row is quoted from its paper.
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}/fidelity_table"
SEEDS="${SEEDS:-1 2 3}"
run() { echo "+ $*"; [[ -n "${DRY_RUN:-}" ]] || "$@"; }

# tag · undistorted variant · opacity reset
TABLE=(
    "cs cook_spinach_undistorted_r2 3000"
    "fs flame_steak_undistorted_r2  10000"
    "ss sear_steak_undistorted_r2   3000"
)
# row · overrides
declare -A ROWS=(
    [ours]=""
    [multiplicative]="model.appearance_model=sh4d_multiplicative"
    [adam]="training.selective_adam=false"
)

for entry in "${TABLE[@]}"; do
    read -r tag ud reset <<< "${entry}"
    [[ -z "${SCENES:-}" || " ${SCENES} " == *" ${tag} "* ]] || continue
    data="${DATA_ROOT}/dynerf/${ud}"
    read -r -a regime <<< "$(pixi run -q python scripts/og4dgs_regime.py "${data}" --overrides)"
    for seed in ${SEEDS}; do
        for row in ours multiplicative adam; do
            name="${row}_${tag}_${seed}"
            read -r -a extra <<< "${ROWS[${row}]}"
            run pixi run train --config-name 4dgs dataset.data_dir="${data}" training.seed="${seed}" \
                "${regime[@]}" strategy.reset_every="${reset}" "${extra[@]}" hydra.run.dir="${OUTPUT_ROOT}/${name}"
            run pixi run eval -c "${OUTPUT_ROOT}/${name}/checkpoints/last.pt" -o "${OUTPUT_ROOT}/${name}/eval/metrics.json"
        done
    done
done
