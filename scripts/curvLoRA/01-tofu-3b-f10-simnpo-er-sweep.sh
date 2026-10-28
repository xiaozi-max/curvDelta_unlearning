#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Split  : forget10 / retain90 / holdout10
#   Loss   : SimNPO (delta=0, beta=4.5, alpha=1.0, gamma=1.0)
#   γ      : 0.9 → 0.7 → 0.5
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
CLORA_MODEL="Llama-3.2-3B-Instruct"
CLORA_MODEL_TAG="3B"
CLORA_MODEL_PATH="${ROOT_DIR}/models/open-unlearning/tofu_${CLORA_MODEL}_full"
CLORA_EXPERIMENT="unlearn/tofu/default"
CLORA_EVAL_EXPERIMENT="eval/tofu/default"

CLORA_TRAINER="SimNPO"
CLORA_EPOCHS=10

CLORA_GPUS=1
CLORA_GPU_MEM=40000
CLORA_PBS=16
CLORA_GA=2      # 4x8x1 = 32

ER_TAGS=(er09 er08 er07 er06 er05)
ER_VALS=(0.5 0.6 0.7 0.8 0.9)

LR_TAGS=(1e4  2e4  5e4)
LR_VALS=(1e-4 2e-4 5e-4)

# ── Split ─────────────────────────────────────────────────────────────────────
CLORA_FORGET_SPLIT="forget10"
CLORA_HOLDOUT_SPLIT="holdout10"
CLORA_RETAIN_SPLIT="retain90"
CLORA_RETAIN_LOGS="saves/eval/tofu_${CLORA_MODEL}_${CLORA_RETAIN_SPLIT}/TOFU_EVAL.json"

clora_require_file "$CLORA_RETAIN_LOGS" "${CLORA_MODEL} ${CLORA_RETAIN_SPLIT} retain logs"

echo "==================================================================="
echo "① γ (energy_ratio) sensitivity"
echo "   model : ${CLORA_MODEL} / ${CLORA_FORGET_SPLIT} / ${CLORA_TRAINER}"
echo "   γ     : ${ER_VALS[*]}"
echo "   lr    : ${LR_VALS[*]}"
echo "   runs  : $(( ${#ER_VALS[@]} * ${#LR_VALS[@]} ))"
echo "==================================================================="

for e in "${!ER_VALS[@]}"; do
    for i in "${!LR_VALS[@]}"; do
        clora_run \
            --task "tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_${ER_TAGS[$e]}_adamw_lr${LR_TAGS[$i]}" \
            --trainer "$CLORA_TRAINER" \
            --lr "${LR_VALS[$i]}" \
            --er "${ER_VALS[$e]}"
    done
done

echo ""
echo "① complete."
echo "Results: saves/unlearn/tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_er0{5,7,9}_adamw_lr*"
