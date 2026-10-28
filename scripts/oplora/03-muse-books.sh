#!/usr/bin/env bash
# OPLoRA MUSE Books. Aligned with scripts/curvLoRA/03-muse-books-main.sh.
# NPO beta stays 0.01. GradDiff keeps the lower learning rates.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

oplora_bootstrap
oplora_defaults

OPLORA_BENCH="muse"
OPLORA_SUMMARY="MUSE_SUMMARY.json"
OPLORA_DATA_SPLIT="Books"
OPLORA_MODEL="Llama-2-7b-hf"
OPLORA_MODEL_TAG="${OPLORA_MODEL}"
OPLORA_MODEL_PATH="${ROOT_DIR}/models/muse-bench/MUSE-Books_target"
OPLORA_EXPERIMENT="unlearn/muse/default"
OPLORA_EVAL_EXPERIMENT="eval/muse/default"
OPLORA_FORGET_SPLIT="forget"
OPLORA_RETAIN_SPLIT="retain1"
OPLORA_RETAIN_LOGS="saves/eval/muse_${OPLORA_MODEL}_${OPLORA_DATA_SPLIT}_retrain/MUSE_EVAL.json"
OPLORA_EPOCHS=5

OPLORA_GPUS=2
OPLORA_GPU_MEM=40000
OPLORA_PBS=4
OPLORA_GA=4

OPLORA_NPO_BETA=0.01
OPLORA_SIMNPO_GAMMA=1.0
OPLORA_GRADDIFF_GAMMA=1.0

TRAINERS=(NPO SimNPO GradDiff)
npo_lr_tags=(2e5 5e5 1e4);       npo_lr_vals=(2e-5 5e-5 1e-4)
simnpo_lr_tags=(5e5 1e4 2e4);    simnpo_lr_vals=(5e-5 1e-4 2e-4)
graddiff_lr_tags=(1e5 2e5 5e5);  graddiff_lr_vals=(1e-5 2e-5 5e-5)

oplora_require_file "$OPLORA_RETAIN_LOGS" "${OPLORA_MODEL} ${OPLORA_DATA_SPLIT} retrain reference"

for trainer in "${TRAINERS[@]}"; do
    case "$trainer" in
        NPO)      lr_tags=("${npo_lr_tags[@]}");      lr_vals=("${npo_lr_vals[@]}") ;;
        SimNPO)   lr_tags=("${simnpo_lr_tags[@]}");   lr_vals=("${simnpo_lr_vals[@]}") ;;
        GradDiff) lr_tags=("${graddiff_lr_tags[@]}"); lr_vals=("${graddiff_lr_vals[@]}") ;;
        *) echo "unknown trainer '${trainer}'" >&2; exit 1 ;;
    esac
    case "$trainer" in
        NPO)      variant="beta001" ;;
        SimNPO)   variant="gamma1" ;;
        GradDiff) variant="" ;;
    esac

    for i in "${!lr_vals[@]}"; do
        if [[ -n "$variant" ]]; then
            task="muse_${OPLORA_MODEL_TAG}_${OPLORA_DATA_SPLIT}_${trainer}_${variant}_oplora_k${OPLORA_K}_lr${lr_tags[$i]}"
        else
            task="muse_${OPLORA_MODEL_TAG}_${OPLORA_DATA_SPLIT}_${trainer}_oplora_k${OPLORA_K}_lr${lr_tags[$i]}"
        fi
        oplora_run --task "$task" --trainer "$trainer" --lr "${lr_vals[$i]}"
    done
done
