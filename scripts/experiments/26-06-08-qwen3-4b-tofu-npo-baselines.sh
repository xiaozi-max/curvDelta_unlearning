#!/bin/bash
# Experiment: TOFU Qwen3-4B NPO - FFT, LoRA, and VILA
# Date: 2026-06-08
# Doc: docs/experiments/26-06-08-qwen3-4b-tofu-npo.md
#
# Matrix:
#   Model:       Qwen3-4B-Instruct-2507
#   Trainer:     NPO (beta=0.1, alpha=1.0, gamma=1.0)
#   Adaptations: FFT | LoRA | VILA
#   Splits:      forget10, forget05
#   LR sweep:    {5e-5, 1e-4, 2e-4} (unified across all adaptations)
#   Total:       18 runs
#
# GPU allocation (effective batch = 32):
#   FFT      → 2 GPU, ZeRO3, per_device=4, accum=4  (4×4×2=32)
#   LoRA     → 1 GPU, ZeRO2, per_device=4, accum=8  (4×8×1=32)
#   VILA     → 1 GPU, ZeRO3, per_device=4, accum=8  (4×8×1=32)

set -euo pipefail
source "common_utils/server_env.sh"

export HF_HOME="${ROOT_DIR}/.cache/huggingface"

mkdir -p saves/logs

# ── Helpers ───────────────────────────────────────────────────────────────────
count_visible_gpus() {
    awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
}

get_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
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

# ── Shared config ──────────────────────────────────────────────────────────────
model="Qwen3-4B-Instruct-2507"
model_tag="qwen3_4b"

# Finetune LR: must match the finetune run used as the unlearning starting point.
# Set to "2e5" or "5e5" once the best finetune LR is determined.
finetune_lr_tag="5e5"

model_path="${ROOT_DIR}/models/open-unlearning/tofu_${model}_full_lr${finetune_lr_tag}"
experiment="unlearn/tofu/default"
eval_experiment="eval/tofu/default"

lora_r=32
lora_alpha=64
lora_dropout=0.05
target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
num_train_epochs=10
warmup_epochs=1.0
weight_decay=0.01
optim=adamw_8bit
effective_batch_size=32

# NPO hyperparameters
npo_beta=0.1
npo_alpha=1.0
npo_gamma=1.0

# VILA hyperparameters (rebalance=true)
vila_sigma=0.05
vila_num_epochs=10
vila_batch_size=2
vila_eps=1e-5

# Unified LR sweep for all adaptations
lr_tags=(5e5 1e4 2e4)
lr_vals=(5e-5 1e-4 2e-4)

# ── VILA init ─────────────────────────────────────────────────────────────────
prepare_vila_init() {
    local forget_split="$1"
    local retain_split="$2"
    local split_tag="$3"

    local vila_dir="saves/vila_init/tofu_${model_tag}_${split_tag}_lr${finetune_lr_tag}_r${lora_r}_a${lora_alpha}"

    if [[ -f "${vila_dir}/adapter_config.json" && -d "${vila_dir}/base" ]]; then
        echo "${vila_dir}"
        return
    fi

    echo "" >&2
    echo ">>> Preparing VILA init: ${vila_dir}" >&2

    grab_gpus 1 40000 80 4
    python scripts/prepare_vila_init.py \
        model="$model" \
        "model.model_args.pretrained_model_name_or_path=${model_path}" \
        "model.tokenizer_args.pretrained_model_name_or_path=${model_path}" \
        "task_name=vila_init_tofu_${model_tag}_${split_tag}" \
        peft=lora \
        "peft.r=${lora_r}" \
        "peft.lora_alpha=${lora_alpha}" \
        "peft.lora_dropout=${lora_dropout}" \
        "peft.target_modules=[${target_modules}]" \
        "forget_split=${forget_split}" \
        "retain_split=${retain_split}" \
        experiment="$experiment" \
        lora_init=vila \
        "lora_init.sigma=${vila_sigma}" \
        "lora_init.num_epochs=${vila_num_epochs}" \
        "lora_init.batch_size=${vila_batch_size}" \
        "lora_init.eps=${vila_eps}" \
        lora_init.rebalance=true \
        "lora_init.path=${vila_dir}" >&2

    echo "${vila_dir}"
}

# ── Eval ──────────────────────────────────────────────────────────────────────
run_eval() {
    local task_name="$1"
    local forget_split="$2"
    local holdout_split="$3"
    local retain_logs="$4"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping eval for ${task_name} (already done)"
        return
    fi

    grab_gpus 1 40000 80 4
    python src/eval.py \
        --config-name=eval.yaml \
        experiment="$eval_experiment" \
        "forget_split=${forget_split}" \
        "holdout_split=${holdout_split}" \
        model="$model" \
        "task_name=${task_name}" \
        "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}" \
        "paths.output_dir=saves/unlearn/${task_name}/evals" \
        "retain_logs_path=${retain_logs}" \
        +overwrite=true
}

# ── Per-run train+eval ─────────────────────────────────────────────────────────
run_npo() {
    local task_name="$1"
    local adapt_tag="$2"
    local lr="$3"
    local forget_split="$4"
    local holdout_split="$5"
    local retain_split="$6"
    local retain_logs="$7"
    local vila_dir="${8:-}"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping ${task_name} (already done)"
        return
    fi

    local peft_args=()
    local model_load_path="$model_path"
    local accel_config="configs/accelerate/zero2_config.yaml"
    local train_done_marker="saves/unlearn/${task_name}/adapter_config.json"
    local grab_mem=40000
    local gpu_count=1
    local vila_ref_args=()

    case "$adapt_tag" in
        fft)
            accel_config="configs/accelerate/default_config.yaml"
            train_done_marker="saves/unlearn/${task_name}/config.json"
            gpu_count=2
            ;;
        lora)
            peft_args=(
                peft=lora
                "peft.r=${lora_r}"
                "peft.lora_alpha=${lora_alpha}"
                "peft.lora_dropout=${lora_dropout}"
            )
            grab_mem=24000
            ;;
        vila)
            accel_config="configs/accelerate/default_config.yaml"
            model_load_path="${vila_dir}/base"
            peft_args=(
                peft=lora
                "peft.r=${lora_r}"
                "peft.lora_alpha=${lora_alpha}"
                "peft.lora_dropout=${lora_dropout}"
                "peft.target_modules=[${target_modules}]"
                "+model.peft_model_path=${vila_dir}"
            )
            grab_mem=30000
            vila_ref_args=("+trainer.method_args.ref_model_path=$model_path")
            ;;
        *)
            echo "ERROR: Unknown adaptation tag: $adapt_tag" >&2
            return 1
            ;;
    esac

    echo ""
    echo ">>> ${task_name}: NPO adapt=${adapt_tag} lr=${lr}"
    echo "    beta=${npo_beta} alpha=${npo_alpha} gamma=${npo_gamma}"

    if [[ ! -f "$train_done_marker" ]]; then
        grab_gpus "$gpu_count" "$grab_mem" 80 4
        local num_processes
        num_processes=$(count_visible_gpus)
        local gradient_accumulation_steps
        if (( effective_batch_size % (per_device_train_batch_size * num_processes) != 0 )); then
            echo "Effective batch size is not divisible by the selected device batch and process count." >&2
            return 1
        fi
        gradient_accumulation_steps=$((effective_batch_size / (per_device_train_batch_size * num_processes)))
        local port
        port=$(get_free_port)

        echo "    world=${num_processes}  bs=${per_device_train_batch_size}×${gradient_accumulation_steps}×${num_processes}=$((per_device_train_batch_size * gradient_accumulation_steps * num_processes))"

        accelerate launch \
            --config_file "$accel_config" \
            --num_processes "$num_processes" \
            --main_process_port "$port" \
            src/train.py --config-name=unlearn.yaml \
            experiment="$experiment" \
            trainer=NPO \
            "task_name=${task_name}" \
            model="$model" \
            "model.model_args.pretrained_model_name_or_path=${model_load_path}" \
            "forget_split=${forget_split}" \
            "retain_split=${retain_split}" \
            "retain_logs_path=${retain_logs}" \
            "trainer.method_args.beta=${npo_beta}" \
            "trainer.method_args.alpha=${npo_alpha}" \
            "trainer.method_args.gamma=${npo_gamma}" \
            "trainer.args.learning_rate=${lr}" \
            "trainer.args.num_train_epochs=${num_train_epochs}" \
            "trainer.args.per_device_train_batch_size=${per_device_train_batch_size}" \
            "trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps}" \
            "trainer.args.optim=${optim}" \
            "++trainer.args.weight_decay=${weight_decay}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no \
            +trainer.args.lr_scheduler_type=linear \
            "trainer.args.warmup_epochs=${warmup_epochs}" \
            ${peft_args[@]+"${peft_args[@]}"} \
            ${vila_ref_args[@]+"${vila_ref_args[@]}"}

        sleep 5
    else
        echo "    Training done, skipping to eval."
    fi

    run_eval "$task_name" "$forget_split" "$holdout_split" "$retain_logs"
    sleep 5
}

# ── Matrix runner ──────────────────────────────────────────────────────────────
run_matrix() {
    local forget_split="$1"
    local holdout_split="$2"
    local retain_split="$3"
    local split_tag="$4"
    local retain_logs="$5"

    require_file "$retain_logs" "${model} ${retain_split} retain logs"

    local vila_dir
    vila_dir=$(prepare_vila_init "$forget_split" "$retain_split" "$split_tag")

    # FFT / LoRA / VILA
    for adapt_tag in fft lora vila; do
        for i in "${!lr_tags[@]}"; do
            run_npo \
                "tofu_${model_tag}_${split_tag}_NPO_${adapt_tag}_lr${lr_tags[$i]}" \
                "$adapt_tag" "${lr_vals[$i]}" \
                "$forget_split" "$holdout_split" "$retain_split" "$retain_logs" \
                "$vila_dir"
        done
    done

}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "TOFU Qwen3-4B NPO - FFT / LoRA / VILA"
echo "Date: 2026-06-08"
echo "Model: ${model}"
echo "NPO: beta=${npo_beta} alpha=${npo_alpha} gamma=${npo_gamma}"
echo "LR sweep: ${lr_vals[*]}"
echo "Total: 18 unlearning runs"
echo "============================================"
echo ""

per_device_train_batch_size=4

run_matrix "forget10" "holdout10" "retain90" "f10" \
    "saves/eval/tofu_${model}_retain90_lr${finetune_lr_tag}_forget10/TOFU_EVAL.json" \
    2>&1 | tee "saves/logs/tofu_${model_tag}_npo_f10.log"

run_matrix "forget05" "holdout05" "retain95" "f05" \
    "saves/eval/tofu_${model}_retain95_lr${finetune_lr_tag}_forget05/TOFU_EVAL.json" \
    2>&1 | tee "saves/logs/tofu_${model_tag}_npo_f05.log"

echo ""
echo "TOFU Qwen3-4B NPO runs complete."
echo "Results under saves/unlearn/tofu_qwen3_4b_{f10,f05}_NPO_*/"
