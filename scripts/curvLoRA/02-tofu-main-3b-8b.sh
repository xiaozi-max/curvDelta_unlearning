#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Loss   : SimNPO (gamma=1.0), NPO (beta=0.1), GradDiff (gamma=1.0, alpha=1.0)
#   lr     : 1e-4, 2e-4
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
CLORA_EXPERIMENT="unlearn/tofu/default"
CLORA_EVAL_EXPERIMENT="eval/tofu/default"

CLORA_ER=0.9
CLORA_EPOCHS=10

CLORA_NPO_BETA=0.1
CLORA_SIMNPO_GAMMA=1.0

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

# ── Per-model / per-trainer resource profile ──────────────────────────────────
set_resources() {
    local mtag="$1" trainer="$2"

    CLORA_GPU_MEM=40000

    case "$mtag" in
        3B)
            CLORA_CBS=4
            CLORA_CHUNKS=1
            if [[ "$trainer" == "NPO" ]]; then
                CLORA_GPUS=2; CLORA_PBS=4; CLORA_GA=4    # 4x4x2 = 32
            else
                CLORA_GPUS=1; CLORA_PBS=16; CLORA_GA=2    # 4x8x1 = 32
            fi
            ;;
        8B)
            CLORA_CBS=1
            CLORA_GPUS=2; CLORA_PBS=4; CLORA_GA=4        # 4x4x2 = 32
            if [[ "$trainer" == "NPO" ]]; then
                CLORA_CHUNKS=8
            else
                CLORA_CHUNKS=4
            fi
            ;;
        *)
            echo "set_resources: unknown model tag '$mtag'" >&2
            return 1
            ;;
    esac
}

echo "==================================================================="
echo "② TOFU main — CurvLoRA, er=${CLORA_ER}"
echo "   models  : ${MODELS[*]}"
echo "   splits  : f10, f05"
echo "   trainers: ${TRAINERS[*]}"
echo "   lr      : ${LR_VALS[*]}"
echo "   runs    : $(( ${#MODELS[@]} * ${#SPLITS[@]} * ${#TRAINERS[@]} * ${#LR_VALS[@]} ))"
echo "==================================================================="

for m in "${MODELS[@]}"; do
    IFS='|' read -r model mtag <<< "$m"
    CLORA_MODEL="$model"
    CLORA_MODEL_TAG="$mtag"
    CLORA_MODEL_PATH="${ROOT_DIR}/models/open-unlearning/tofu_${model}_full"

    for s in "${SPLITS[@]}"; do
        read -r forget_split holdout_split retain_split split_tag <<< "$s"
        CLORA_FORGET_SPLIT="$forget_split"
        CLORA_HOLDOUT_SPLIT="$holdout_split"
        CLORA_RETAIN_SPLIT="$retain_split"
        CLORA_RETAIN_LOGS="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

        clora_require_file "$CLORA_RETAIN_LOGS" "${model} ${retain_split} retain logs"

        for trainer in "${TRAINERS[@]}"; do
            set_resources "$mtag" "$trainer"

            for i in "${!LR_VALS[@]}"; do
                clora_run \
                    --task "tofu_${mtag}_${split_tag}_${trainer}_clora_static_klfisher_er09_adamw_lr${LR_TAGS[$i]}" \
                    --trainer "$trainer" \
                    --lr "${LR_VALS[$i]}" \
                    --er "$CLORA_ER"
            done
        done
    done
done

echo ""
echo "② complete."
echo "Results: saves/unlearn/tofu_{3B,8B}_{f10,f05}_{SimNPO,NPO,GradDiff}_clora_static_klfisher_er09_adamw_lr*"
