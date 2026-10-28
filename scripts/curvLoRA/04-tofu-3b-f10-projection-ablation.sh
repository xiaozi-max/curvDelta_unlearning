#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Split  : forget10 / retain90 / holdout10
#   modes  : kfac_top (bilateral), kfac_bottom, random
#   lr     : 1e-4, 2e-4, 5e-4
#
#     tofu_3B_f10_SimNPO_clora_static_klfisher_er09_adamw_lr{1e4,2e4,5e4}
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
CLORA_ER=0.9
CLORA_EPOCHS=10

CLORA_GPUS=1
CLORA_GPU_MEM=40000
CLORA_PBS=16
CLORA_GA=2      # 4x8x1 = 32

RUN_TOP_ANCHOR=false

LR_TAGS=(1e4  2e4  5e4)
LR_VALS=(1e-4 2e-4 5e-4)

# ── Split ─────────────────────────────────────────────────────────────────────
CLORA_FORGET_SPLIT="forget10"
CLORA_HOLDOUT_SPLIT="holdout10"
CLORA_RETAIN_SPLIT="retain90"
CLORA_RETAIN_LOGS="saves/eval/tofu_${CLORA_MODEL}_${CLORA_RETAIN_SPLIT}/TOFU_EVAL.json"

clora_require_file "$CLORA_RETAIN_LOGS" "${CLORA_MODEL} ${CLORA_RETAIN_SPLIT} retain logs"

if [[ "$RUN_TOP_ANCHOR" == "true" ]]; then
    PROJ_MODES=(kfac_top kfac_bottom random)
else
    PROJ_MODES=(kfac_bottom random)
fi

echo "==================================================================="
echo "④ Projection mode ablation — ${CLORA_MODEL} / ${CLORA_FORGET_SPLIT} / ${CLORA_TRAINER}"
echo "   er    : ${CLORA_ER}"
echo "   modes : ${PROJ_MODES[*]}"
echo "   lr    : ${LR_VALS[*]}"
echo "   runs  : $(( ${#PROJ_MODES[@]} * ${#LR_VALS[@]} ))"
echo "==================================================================="

for proj in "${PROJ_MODES[@]}"; do
    for i in "${!LR_VALS[@]}"; do
        clora_run \
            --task "tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_proj${proj}_er09_adamw_lr${LR_TAGS[$i]}" \
            --trainer "$CLORA_TRAINER" \
            --lr "${LR_VALS[$i]}" \
            --er "$CLORA_ER" \
            --proj "$proj"
    done
done

echo ""
if [[ "$RUN_TOP_ANCHOR" == "true" ]]; then
    echo "④ complete (kfac_top re-run included)."
else
    echo "④ complete."
    echo "Existing kfac_top (bilateral) anchor runs:"
    for i in "${!LR_TAGS[@]}"; do
        echo "  saves/unlearn/tofu_${CLORA_MODEL_TAG}_f10_${CLORA_TRAINER}_clora_static_klfisher_er09_adamw_lr${LR_TAGS[$i]}"
    done
    echo "Run script 01 first, or set RUN_TOP_ANCHOR=true."
fi
