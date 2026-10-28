#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#
#   Split  : Books / forget / retain1
#   Loss   : NPO (beta=0.01), SimNPO (gamma=1.0), GradDiff (gamma=1.0)
#
#
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"


clora_bootstrap
clora_defaults

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLORA_BENCH="muse"
CLORA_SUMMARY="MUSE_SUMMARY.json"
CLORA_DATA_SPLIT="Books"
CLORA_MODEL="Llama-2-7b-hf"
CLORA_MODEL_TAG="${CLORA_MODEL}"
CLORA_MODEL_PATH="${ROOT_DIR}/models/muse-bench/MUSE-Books_target"
CLORA_EXPERIMENT="unlearn/muse/default"
CLORA_EVAL_EXPERIMENT="eval/muse/default"

CLORA_FORGET_SPLIT="forget"
CLORA_RETAIN_SPLIT="retain1"
CLORA_RETAIN_LOGS="saves/eval/muse_${CLORA_MODEL}_${CLORA_DATA_SPLIT}_retrain/MUSE_EVAL.json"

CLORA_ER=0.9
CLORA_EPOCHS=5

CLORA_GPUS=4
CLORA_GPU_MEM=40000
CLORA_PBS=2
CLORA_GA=4          # 1x8x4 = 32
CLORA_CBS=1
CLORA_CHUNKS=4

CLORA_NPO_BETA=0.01
CLORA_SIMNPO_GAMMA=1.0
CLORA_GRADDIFF_GAMMA=1.0

# ── Per-trainer LR sweep ──────────────────────────────────────────────────────
TRAINERS=(NPO SimNPO GradDiff)

npo_lr_tags=(2e5 5e5 1e4);       npo_lr_vals=(2e-5 5e-5 1e-4)
simnpo_lr_tags=(5e5 1e4 2e4);    simnpo_lr_vals=(5e-5 1e-4 2e-4)
graddiff_lr_tags=(1e5 2e5 5e5);  graddiff_lr_vals=(1e-5 2e-5 5e-5)

clora_require_file "$CLORA_RETAIN_LOGS" "${CLORA_MODEL} ${CLORA_DATA_SPLIT} retrain reference"

echo "==================================================================="
echo "③ MUSE ${CLORA_DATA_SPLIT} main — CurvLoRA, er=${CLORA_ER}"
echo "   model : ${CLORA_MODEL}  (${CLORA_MODEL_PATH})"
echo "   NPO      beta=0.01  lr=${npo_lr_vals[*]}"
echo "   SimNPO   gamma=1.0  lr=${simnpo_lr_vals[*]}"
echo "   GradDiff gamma=1.0  lr=${graddiff_lr_vals[*]}"
echo "==================================================================="

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
            task="muse_${CLORA_MODEL_TAG}_${CLORA_DATA_SPLIT}_${trainer}_${variant}_clora_klfisher_er09_lr${lr_tags[$i]}"
        else
            task="muse_${CLORA_MODEL_TAG}_${CLORA_DATA_SPLIT}_${trainer}_clora_klfisher_er09_lr${lr_tags[$i]}"
        fi

        clora_run \
            --task "$task" \
            --trainer "$trainer" \
            --lr "${lr_vals[$i]}" \
            --er "$CLORA_ER"
    done
done

echo ""
echo "③ complete."
echo "Results: saves/unlearn/muse_${CLORA_MODEL_TAG}_${CLORA_DATA_SPLIT}_{NPO_beta001,SimNPO_gamma1,GradDiff}_clora_klfisher_er09_lr*"
