#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Split  : forget10 / retain90 / holdout10
#   Loss   : SimNPO (delta=0, beta=4.5, alpha=1.0, gamma=1.0)
#   rr     : retain_ratio ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
#   lr     : 1e-4, 2e-4, 5e-4
#
#
#     rr=0.25 -> ..._er09_rratio025_...
#     rr=2.0  -> ..._er09_rratio20_...
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
CLORA_ER=0.9
CLORA_EPOCHS=10

CLORA_GPUS=1
CLORA_GPU_MEM=40000
CLORA_PBS=4
CLORA_GA=8      # 4x8x1 = 32

LR_TAGS=(1e4  2e4  5e4)
LR_VALS=(1e-4 2e-4 5e-4)

RR_VALS=(0.25 0.5 1.0 2.0 4.0)
RR_TAGS=("rratio025" "rratio05" "" "rratio20" "rratio40")

# ── Split ─────────────────────────────────────────────────────────────────────
CLORA_FORGET_SPLIT="forget10"
CLORA_HOLDOUT_SPLIT="holdout10"
CLORA_RETAIN_SPLIT="retain90"
CLORA_RETAIN_LOGS="saves/eval/tofu_${CLORA_MODEL}_${CLORA_RETAIN_SPLIT}/TOFU_EVAL.json"

clora_require_file "$CLORA_RETAIN_LOGS" "${CLORA_MODEL} ${CLORA_RETAIN_SPLIT} retain logs"

echo "==================================================================="
echo "⑥ TOFU 3B f10 SimNPO — retain-data-ratio analysis"
echo "   model : ${CLORA_MODEL}"
echo "   rr    : ${RR_VALS[*]}"
echo "   lr    : ${LR_VALS[*]}"
echo "   runs  : $(( ${#RR_VALS[@]} * ${#LR_VALS[@]} ))"
echo "==================================================================="

for r in "${!RR_VALS[@]}"; do
    rr="${RR_VALS[$r]}"
    rr_tag="${RR_TAGS[$r]}"

    for i in "${!LR_VALS[@]}"; do
        if [[ -n "$rr_tag" ]]; then
            task="tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_er09_${rr_tag}_adamw_lr${LR_TAGS[$i]}"
        else
            task="tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_er09_adamw_lr${LR_TAGS[$i]}"
        fi

        clora_run \
            --task "$task" \
            --trainer "$CLORA_TRAINER" \
            --lr "${LR_VALS[$i]}" \
            --er "$CLORA_ER" \
            --rr "$rr"
    done
done

echo ""
echo "⑥ complete."
echo "Results: saves/unlearn/tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_er09{_rratio025,_rratio05,, _rratio20,_rratio40}_adamw_lr*"
