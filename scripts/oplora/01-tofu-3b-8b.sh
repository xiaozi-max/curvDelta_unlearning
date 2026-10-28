#!/usr/bin/env bash
# OPLoRA TOFU 3B/8B. Aligned with scripts/curvLoRA/02-tofu-main-3b-8b.sh
# (same splits, trainers, lrs, batch). Projection basis is SVD k=16, not K-FAC.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

oplora_bootstrap
oplora_defaults

OPLORA_BENCH="tofu"
OPLORA_SUMMARY="TOFU_SUMMARY.json"
OPLORA_EXPERIMENT="unlearn/tofu/default"
OPLORA_EVAL_EXPERIMENT="eval/tofu/default"
OPLORA_EPOCHS=10
OPLORA_NPO_BETA=0.1
OPLORA_SIMNPO_GAMMA=1.0

MODELS=(
    "Llama-3.2-3B-Instruct|3B"
    "Llama-3.1-8B-Instruct|8B"
)
TRAINERS=(SimNPO NPO GradDiff)
SPLITS=(
    "forget10 holdout10 retain90 f10"
    "forget05 holdout05 retain95 f05"
)
LR_TAGS=(1e4 2e4)
LR_VALS=(1e-4 2e-4)

set_resources() {
    local mtag="$1" trainer="$2"
    OPLORA_GPU_MEM=40000
    case "$mtag" in
        3B)
            if [[ "$trainer" == "NPO" ]]; then
                OPLORA_GPUS=1; OPLORA_PBS=4; OPLORA_GA=8
            else
                OPLORA_GPUS=1; OPLORA_PBS=16; OPLORA_GA=2
            fi
            ;;
        8B)
            OPLORA_GPUS=2; OPLORA_PBS=4; OPLORA_GA=4
            ;;
        *)
            echo "set_resources: unknown model tag '$mtag'" >&2
            return 1
            ;;
    esac
}

for m in "${MODELS[@]}"; do
    IFS='|' read -r model mtag <<< "$m"
    OPLORA_MODEL="$model"
    OPLORA_MODEL_TAG="$mtag"
    OPLORA_MODEL_PATH="${ROOT_DIR}/models/open-unlearning/tofu_${model}_full"

    for s in "${SPLITS[@]}"; do
        read -r forget_split holdout_split retain_split split_tag <<< "$s"
        OPLORA_FORGET_SPLIT="$forget_split"
        OPLORA_HOLDOUT_SPLIT="$holdout_split"
        OPLORA_RETAIN_SPLIT="$retain_split"
        OPLORA_RETAIN_LOGS="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
        oplora_require_file "$OPLORA_RETAIN_LOGS" "${model} ${retain_split} retain logs"

        for trainer in "${TRAINERS[@]}"; do
            set_resources "$mtag" "$trainer"
            for i in "${!LR_VALS[@]}"; do
                oplora_run \
                    --task "tofu_${mtag}_${split_tag}_${trainer}_oplora_k${OPLORA_K}_adamw_lr${LR_TAGS[$i]}" \
                    --trainer "$trainer" \
                    --lr "${LR_VALS[$i]}"
            done
        done
    done
done
