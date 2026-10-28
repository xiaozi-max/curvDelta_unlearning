#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Loss   : SimNPO (gamma=1.0), NPO (beta=0.1), GradDiff (gamma=1.0, alpha=1.0)
#   lr     : 1e-4, 2e-4
#   GPU    : 1x L40 48GB  (4x8x1 = 32 effective batch)
#
#       bash scripts/experiments/26-06-08-qwen3-4b-tofu-finetune.sh
#
#
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

clora_bootstrap
clora_defaults

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLORA_BENCH="tofu"
CLORA_SUMMARY="TOFU_SUMMARY.json"
CLORA_MODEL="Qwen3-4B-Instruct-2507"
CLORA_MODEL_TAG="qwen3_4b"
FINETUNE_LR_TAG="5e5"
CLORA_MODEL_PATH="${ROOT_DIR}/models/open-unlearning/tofu_${CLORA_MODEL}_full_lr${FINETUNE_LR_TAG}"
CLORA_EXPERIMENT="unlearn/tofu/default"
CLORA_EVAL_EXPERIMENT="eval/tofu/default"

CLORA_ER=0.9
CLORA_EPOCHS=10

CLORA_GPUS=1
CLORA_GPU_MEM=40000
CLORA_PBS=4
CLORA_GA=8          # 4x8x1 = 32

CLORA_CBS=2
CLORA_CHUNKS=8

CLORA_SIMNPO_GAMMA=1.0
CLORA_NPO_BETA=0.1

TRAINERS=(SimNPO NPO GradDiff)

SPLITS=(
    "forget10 holdout10 retain90 f10"
    "forget05 holdout05 retain95 f05"
)

LR_TAGS=(1e4 2e4)
LR_VALS=(1e-4 2e-4)

retain_logs_for() {
    case "$1" in
        f10) echo "saves/eval/tofu_${CLORA_MODEL}_retain90_lr${FINETUNE_LR_TAG}_forget10/TOFU_EVAL.json" ;;
        f05) echo "saves/eval/tofu_${CLORA_MODEL}_retain95_lr${FINETUNE_LR_TAG}_forget05/TOFU_EVAL.json" ;;
        *)   echo "retain_logs_for: unknown split tag '$1'" >&2; return 1 ;;
    esac
}

task_name_for() {
    local trainer="$1" split_tag="$2" lr_tag="$3"
    local variant=""
    if [[ "$trainer" == "SimNPO" ]]; then
        variant="_gamma1"
    fi
    echo "tofu_${CLORA_MODEL_TAG}_${split_tag}_${trainer}_clora_er09${variant}_lr${lr_tag}"
}

echo "==================================================================="
echo "TOFU Qwen3-4B main - CurvLoRA, er=${CLORA_ER}"
echo "   model   : ${CLORA_MODEL}"
echo "   trainers: ${TRAINERS[*]}"
echo "   splits  : f10, f05"
echo "   lr      : ${LR_VALS[*]}"
echo "   runs    : $(( ${#SPLITS[@]} * ${#TRAINERS[@]} * ${#LR_VALS[@]} ))"
echo "==================================================================="

clora_require_file "${CLORA_MODEL_PATH}/config.json" \
    "Qwen3-4B unlearning checkpoint; run the Qwen3-4B fine-tuning script first"

for s in "${SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split split_tag <<< "$s"
    CLORA_FORGET_SPLIT="$forget_split"
    CLORA_HOLDOUT_SPLIT="$holdout_split"
    CLORA_RETAIN_SPLIT="$retain_split"
    CLORA_RETAIN_LOGS="$(retain_logs_for "$split_tag")"

    clora_require_file "$CLORA_RETAIN_LOGS" "Qwen3-4B ${retain_split} retain logs"

    for trainer in "${TRAINERS[@]}"; do
        for i in "${!LR_VALS[@]}"; do
            clora_run \
                --task "$(task_name_for "$trainer" "$split_tag" "${LR_TAGS[$i]}")" \
                --trainer "$trainer" \
                --lr "${LR_VALS[$i]}" \
                --er "$CLORA_ER"
        done
    done
done

echo ""
echo "⑤ complete."
echo "Results: saves/unlearn/tofu_${CLORA_MODEL_TAG}_{f10,f05}_{SimNPO,NPO,GradDiff}_clora_er09*_lr*"
