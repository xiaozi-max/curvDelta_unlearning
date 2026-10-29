#!/bin/bash
# Experiment: TOFU FFT, LoRA, VILA Baselines (3B/8B, forget10/forget05)
# Date: 2026-05-02
# Doc: docs/experiments/26-05-02-tofu-fft-lora-vila-baselines.md
#
# Baseline matrix for TOFU on Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct:
#   Adaptation: FFT | LoRA | VILA
#   Methods:    GradAscent | SimNPO | NPO
#   Splits:     forget10 | forget05
#   LR sweep:   {5e-6, 1e-5, 2e-5}
#   Total:      2 models × 2 splits × 3 adaptations × 3 methods × 3 LRs = 108 runs
#
# GPU allocation (per model/adaptation):
#   FFT 3B  (any method)       → 2 GPUs, ZeRO3, per_device=4, grad_accum=4  (4×4×2=32)
#   FFT 8B  (any method)       → 4 GPUs, ZeRO3, per_device=2, grad_accum=4  (2×4×4=32)
#   LoRA    (any method)       → 2 GPUs, ZeRO2, per_device=4, grad_accum=4  (4×4×2=32)
#   VILA    GradAscent         → 2 GPUs, ZeRO3, per_device=4, grad_accum=4  (4×4×2=32)
#   VILA    SimNPO/NPO         → 4 GPUs, ZeRO3, per_device=2, grad_accum=4  (2×4×4=32)
#
# Usage:
#   bash scripts/experiments/26-05-02-tofu-fft-lora-vila-baselines.sh

set -euo pipefail
source "common_utils/server_env.sh"

mkdir -p saves/logs

# ── Helpers ───────────────────────────────────────────────────────────────────
count_visible_gpus() {
    awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing ${label}: ${path}" >&2
        echo "Create the required retain reference eval before running this experiment." >&2
        exit 1
    fi
}

# ── Configuration ─────────────────────────────────────────────────────────────
experiment="unlearn/tofu/default"
eval_experiment="eval/tofu/default"

lora_r=32
lora_alpha=64
lora_dropout=0.05
num_train_epochs=10
npo_beta=0.1          # TOFU sequences ~256 tokens; 0.1 is appropriate (vs 0.01 for MUSE 2048-token)
npo_gamma=1.0
simnpo_beta=4.5
simnpo_gamma=1.0
warmup_epochs=1.0
per_device_train_batch_size=4
gradient_accumulation_steps=4

# VILA hyperparameters
vila_sigma=0.05
vila_num_epochs=10
vila_batch_size=2
vila_eps=1e-5

lr_tags=(5e6 1e5 2e5)
lr_vals=(5e-6 1e-5 2e-5)

# ── VILA init ─────────────────────────────────────────────────────────────────
# prepare_vila_init MODEL MODEL_TAG MODEL_PATH FORGET_SPLIT RETAIN_SPLIT SPLIT_TAG
# Prints the vila_dir path to stdout.
prepare_vila_init() {
    local model="$1"
    local model_tag="$2"
    local model_path="$3"
    local forget_split="$4"
    local retain_split="$5"
    local split_tag="$6"

    local vila_dir="saves/vila_init/tofu_${model_tag}_${split_tag}_r${lora_r}_a${lora_alpha}"

    if [[ -f "${vila_dir}/adapter_config.json" && -d "${vila_dir}/base" ]]; then
        echo ">>> [SKIP] VILA init already exists at ${vila_dir}" >&2
        echo "${vila_dir}"
        return
    fi

    echo "" >&2
    echo ">>> Preparing VILA init: ${vila_dir}" >&2
    echo "    sigma=${vila_sigma}  epochs=${vila_num_epochs}  batch_size=${vila_batch_size}  eps=${vila_eps}" >&2

    grab_gpus 1 40000
    python scripts/prepare_vila_init.py \
        experiment="$experiment" \
        model="$model" \
        model.model_args.pretrained_model_name_or_path="$model_path" \
        model.tokenizer_args.pretrained_model_name_or_path="$model_path" \
        forget_split="$forget_split" \
        retain_split="$retain_split" \
        task_name="vila_init_tofu_${model_tag}_${split_tag}" \
        peft=lora \
        peft.r="$lora_r" \
        peft.lora_alpha="$lora_alpha" \
        peft.lora_dropout=0 \
        lora_init=vila \
        lora_init.path="$vila_dir" \
        lora_init.sigma="$vila_sigma" \
        lora_init.num_epochs="$vila_num_epochs" \
        lora_init.batch_size="$vila_batch_size" \
        lora_init.eps="$vila_eps" \
        lora_init.rebalance=false

    echo ">>> VILA init done: ${vila_dir}" >&2
    echo "${vila_dir}"
}

# ── Eval ──────────────────────────────────────────────────────────────────────
run_eval() {
    local task_name="$1"
    local model="$2"
    local forget_split="$3"
    local holdout_split="$4"
    local retain_logs="$5"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping eval for ${task_name} (already done)"
        return
    fi

    grab_gpus 1 40000
    python src/eval.py \
        --config-name=eval.yaml \
        experiment="$eval_experiment" \
        forget_split="$forget_split" \
        holdout_split="$holdout_split" \
        model="$model" \
        task_name="$task_name" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
        paths.output_dir="saves/unlearn/${task_name}/evals" \
        retain_logs_path="$retain_logs" \
        +overwrite=true
}

# ── Per-run train+eval ────────────────────────────────────────────────────────
# run_unlearn_and_eval TASK_NAME TRAINER ADAPTATION_TAG LR
#                      MODEL MODEL_TAG MODEL_PATH
#                      FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT RETAIN_LOGS
#   ADAPTATION_TAG: fft | lora | vila
run_unlearn_and_eval() {
    local task_name="$1"
    local trainer="$2"
    local adapt_tag="$3"
    local lr="$4"
    local model="$5"
    local model_tag="$6"
    local model_path="$7"
    local forget_split="$8"
    local holdout_split="$9"
    local retain_split="${10}"
    local retain_logs="${11}"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping ${task_name} (already done)"
        return
    fi

    # ── Adaptation-specific args ──────────────────────────────────────────────
    local peft_args=()
    local ref_args=()
    local beta_args=()
    local retain_loss_args=()
    local model_load_path="$model_path"
    local accel_config="configs/accelerate/zero2_config.yaml"
    local gpu_count=2
    local per_device="${per_device_train_batch_size}"
    local grad_accum="${gradient_accumulation_steps}"
    local train_done_marker="saves/unlearn/${task_name}/adapter_config.json"

    # 8B FFT needs 4 GPUs to fit in memory
    local is_8b=false
    [[ "$model_tag" == "8B" ]] && is_8b=true

    case "$adapt_tag" in
        fft)
            accel_config="configs/accelerate/default_config.yaml"
            train_done_marker="saves/unlearn/${task_name}/trainer_state.json"
            if $is_8b; then
                gpu_count=4
                per_device=2   # 2×4×4=32
            fi
            if [[ "$trainer" != "GradAscent" ]]; then
                ref_args=(+trainer.method_args.ref_model_path="$model_path")
            fi
            ;;
        lora)
            peft_args=(
                peft=lora
                peft.r="$lora_r"
                peft.lora_alpha="$lora_alpha"
                peft.lora_dropout="$lora_dropout"
            )
            # NPO/SimNPO with LoRA: frozen base model acts as implicit reference
            ;;
        vila)
            accel_config="configs/accelerate/default_config.yaml"
            local split_tag
            case "$forget_split" in
                forget10) split_tag="f10" ;;
                forget05) split_tag="f05" ;;
                *) split_tag="$forget_split" ;;
            esac
            local vila_dir="saves/vila_init/tofu_${model_tag}_${split_tag}_r${lora_r}_a${lora_alpha}"
            model_load_path="${vila_dir}/base"
            peft_args=(
                peft=lora
                peft.r="$lora_r"
                peft.lora_alpha="$lora_alpha"
                peft.lora_dropout=0
                +model.peft_model_path="$vila_dir"
            )
            if [[ "$trainer" != "GradAscent" ]]; then
                ref_args=(+trainer.method_args.ref_model_path="$model_path")
                if $is_8b; then
                    gpu_count=4
                    per_device=2   # 2×4×4=32
                fi
            fi
            ;;
        *)
            echo "ERROR: Unknown adaptation tag: $adapt_tag" >&2
            return 1
            ;;
    esac

    case "$trainer" in
        NPO) beta_args=(trainer.method_args.beta="$npo_beta" trainer.method_args.gamma="$npo_gamma") ;;
        SimNPO) beta_args=(trainer.method_args.beta="$simnpo_beta" trainer.method_args.gamma="$simnpo_gamma") ;;
    esac
    if [[ "$trainer" == "NPO" ]]; then
        retain_loss_args=(trainer.method_args.retain_loss_type=KL)
    fi

    echo ""
    echo ">>> ${task_name}"
    echo "    trainer=${trainer}  adapt=${adapt_tag}  lr=${lr}  epochs=${num_train_epochs}  warmup=${warmup_epochs}"

    if [[ ! -f "$train_done_marker" ]]; then
        grab_gpus "$gpu_count" 40000
        local num_processes
        num_processes=$(count_visible_gpus)
        local port
        port=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

        echo "    world=${num_processes}  bs=${per_device}×${grad_accum}×${num_processes}=$((per_device * grad_accum * num_processes))"

        accelerate launch \
            --config_file "$accel_config" \
            --num_processes "$num_processes" \
            --main_process_port "$port" \
            src/train.py --config-name=unlearn.yaml \
            experiment="$experiment" \
            trainer="$trainer" \
            task_name="$task_name" \
            model="$model" \
            model.model_args.pretrained_model_name_or_path="$model_load_path" \
            forget_split="$forget_split" \
            retain_split="$retain_split" \
            retain_logs_path="$retain_logs" \
            trainer.args.learning_rate="$lr" \
            trainer.args.num_train_epochs="$num_train_epochs" \
            trainer.args.per_device_train_batch_size="$per_device" \
            trainer.args.gradient_accumulation_steps="$grad_accum" \
            +trainer.args.lr_scheduler_type=linear \
            trainer.args.warmup_epochs="$warmup_epochs" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            ${peft_args[@]+"${peft_args[@]}"} \
            ${beta_args[@]+"${beta_args[@]}"} \
            ${retain_loss_args[@]+"${retain_loss_args[@]}"} \
            ${ref_args[@]+"${ref_args[@]}"}
        sleep 5
    else
        echo "    Training done, skipping to eval."
    fi

    run_eval "$task_name" "$model" "$forget_split" "$holdout_split" "$retain_logs"
    sleep 5
}

# ── Matrix runner ─────────────────────────────────────────────────────────────
# run_matrix MODEL MODEL_TAG MODEL_PATH FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT SPLIT_TAG
run_matrix() {
    local model="$1"
    local model_tag="$2"
    local model_path="$3"
    local forget_split="$4"
    local holdout_split="$5"
    local retain_split="$6"
    local split_tag="$7"
    local retain_logs="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

    require_file "$retain_logs" "${model} ${retain_split} retain logs"

    # VILA init — computed once per (model, split), reused across LRs
    local vila_dir
    vila_dir=$(prepare_vila_init "$model" "$model_tag" "$model_path" \
                                 "$forget_split" "$retain_split" "$split_tag")

    local prefix="tofu_${model_tag}_${split_tag}"

    # ── Group A: FFT ──────────────────────────────────────────────────────────
    echo ""; echo "=== ${model_tag} ${split_tag} | Group A: FFT ==="
    for i in "${!lr_tags[@]}"; do
        run_unlearn_and_eval "${prefix}_GA_fft_lr${lr_tags[$i]}"   GradAscent fft "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_SimNPO_fft_lr${lr_tags[$i]}" SimNPO fft "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_NPO_fft_lr${lr_tags[$i]}"  NPO        fft "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
    done

    # ── Group B: LoRA ─────────────────────────────────────────────────────────
    echo ""; echo "=== ${model_tag} ${split_tag} | Group B: LoRA r=${lora_r} ==="
    for i in "${!lr_tags[@]}"; do
        run_unlearn_and_eval "${prefix}_GA_lora_lr${lr_tags[$i]}"   GradAscent lora "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_SimNPO_lora_lr${lr_tags[$i]}" SimNPO lora "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_NPO_lora_lr${lr_tags[$i]}"  NPO        lora "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
    done

    # ── Group C: VILA ─────────────────────────────────────────────────────────
    echo ""; echo "=== ${model_tag} ${split_tag} | Group C: VILA r=${lora_r} ==="
    for i in "${!lr_tags[@]}"; do
        run_unlearn_and_eval "${prefix}_GA_vila_lr${lr_tags[$i]}"   GradAscent vila "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_SimNPO_vila_lr${lr_tags[$i]}" SimNPO vila "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
        run_unlearn_and_eval "${prefix}_NPO_vila_lr${lr_tags[$i]}"  NPO        vila "${lr_vals[$i]}" \
            "$model" "$model_tag" "$model_path" \
            "$forget_split" "$holdout_split" "$retain_split" "$retain_logs"
    done
}

# ##############################################################################
# ── Main ──────────────────────────────────────────────────────────────────────
# ##############################################################################
echo ""
echo "============================================"
echo "TOFU FFT / LoRA / VILA Baseline Matrix"
echo "Date: 2026-05-02"
echo "Models: Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct"
echo "Splits: forget10, forget05"
echo "LR sweep: ${lr_vals[*]}"
echo "npo_beta: ${npo_beta}  warmup_epochs: ${warmup_epochs}"
echo "NPO retain loss: KL"
echo "Total: 108 unlearning runs"
echo "============================================"
echo ""

# ── 3B forget10 ───────────────────────────────────────────────────────────────
run_matrix \
    "Llama-3.2-3B-Instruct" "3B" \
    "${ROOT_DIR}/models/open-unlearning/tofu_Llama-3.2-3B-Instruct_full" \
    "forget10" "holdout10" "retain90" "f10" \
    2>&1 | tee saves/logs/tofu_3B_f10_baselines.log

# ── 3B forget05 ───────────────────────────────────────────────────────────────
run_matrix \
    "Llama-3.2-3B-Instruct" "3B" \
    "${ROOT_DIR}/models/open-unlearning/tofu_Llama-3.2-3B-Instruct_full" \
    "forget05" "holdout05" "retain95" "f05" \
    2>&1 | tee saves/logs/tofu_3B_f05_baselines.log

# ── 8B forget10 ───────────────────────────────────────────────────────────────
run_matrix \
    "Llama-3.1-8B-Instruct" "8B" \
    "${ROOT_DIR}/models/open-unlearning/tofu_Llama-3.1-8B-Instruct_full" \
    "forget10" "holdout10" "retain90" "f10" \
    2>&1 | tee saves/logs/tofu_8B_f10_baselines.log

# ── 8B forget05 ───────────────────────────────────────────────────────────────
run_matrix \
    "Llama-3.1-8B-Instruct" "8B" \
    "${ROOT_DIR}/models/open-unlearning/tofu_Llama-3.1-8B-Instruct_full" \
    "forget05" "holdout05" "retain95" "f05" \
    2>&1 | tee saves/logs/tofu_8B_f05_baselines.log

echo ""
echo "============================================"
echo "All TOFU baseline runs completed."
echo "Results under saves/unlearn/tofu_{3B,8B}_{f10,f05}_*/"
echo "============================================"

