#!/usr/bin/env bash
# OPLoRA TOFU Qwen3-4B. Aligned with scripts/curvLoRA/05-qwen3-4b-tofu-main.sh.
# Prerequisite: scripts/experiments/26-06-08-qwen3-4b-tofu-finetune.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

oplora_bootstrap
oplora_defaults

OPLORA_BENCH="tofu"
OPLORA_SUMMARY="TOFU_SUMMARY.json"
OPLORA_MODEL="Qwen3-4B-Instruct-2507"
OPLORA_MODEL_TAG="qwen3_4b"
FINETUNE_LR_TAG="5e5"
OPLORA_MODEL_PATH="${ROOT_DIR}/models/open-unlearning/tofu_${OPLORA_MODEL}_full_lr${FINETUNE_LR_TAG}"
OPLORA_EXPERIMENT="unlearn/tofu/default"
OPLORA_EVAL_EXPERIMENT="eval/tofu/default"
OPLORA_EPOCHS=10

OPLORA_GPUS=1
OPLORA_GPU_MEM=40000
OPLORA_PBS=4
OPLORA_GA=8

OPLORA_SIMNPO_GAMMA=1.0
OPLORA_NPO_BETA=0.1

TRAINERS=(SimNPO NPO GradDiff)
SPLITS=(
    "forget10 holdout10 retain90 f10"
    "forget05 holdout05 retain95 f05"
)
LR_TAGS=(1e4 2e4)
LR_VALS=(1e-4 2e-4)

retain_logs_for() {
    case "$1" in
        f10) echo "saves/eval/tofu_${OPLORA_MODEL}_retain90_lr${FINETUNE_LR_TAG}_forget10/TOFU_EVAL.json" ;;
        f05) echo "saves/eval/tofu_${OPLORA_MODEL}_retain95_lr${FINETUNE_LR_TAG}_forget05/TOFU_EVAL.json" ;;
        *) echo "retain_logs_for: unknown split tag '$1'" >&2; return 1 ;;
    esac
}

oplora_require_file "${OPLORA_MODEL_PATH}/config.json" \
    "Qwen3-4B unlearning start checkpoint (run 26-06-08-qwen3-4b-tofu-finetune.sh first)"

for s in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split split_tag <<< "$s"
    OPLORA_FORGET_SPLIT="$forget_split"
    OPLORA_HOLDOUT_SPLIT="$holdout_split"
    OPLORA_RETAIN_SPLIT="$retain_split"
    OPLORA_RETAIN_LOGS="$(retain_logs_for "$split_tag")"
    oplora_require_file "$OPLORA_RETAIN_LOGS" "Qwen3-4B ${retain_split} retain logs"

    for trainer in "${TRAINERS[@]}"; do
        for i in "${!LR_VALS[@]}"; do
            oplora_run \
                --task "tofu_${OPLORA_MODEL_TAG}_${split_tag}_${trainer}_oplora_k${OPLORA_K}_lr${LR_TAGS[$i]}" \
                --trainer "$trainer" \
                --lr "${LR_VALS[$i]}"
        done
    done
done
