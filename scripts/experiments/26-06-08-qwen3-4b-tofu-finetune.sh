#!/bin/bash
# Experiment: TOFU Qwen3-4B Finetuning (full / retain90 / retain95)
# Date: 2026-06-08
# Doc: docs/experiments/26-06-08-qwen3-4b-tofu-finetune.md
#
# Finetune Qwen3-4B-Instruct-2507 on TOFU full, retain90, retain95.
# Each model is then evaluated on forget10 and forget05.
# Outputs the retain reference logs required by unlearning experiments.

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

# ── Shared config ──────────────────────────────────────────────────────────────
model="Qwen3-4B-Instruct-2507"
model_tag="qwen3_4b"
base_model_path="${ROOT_DIR}/models/Qwen/Qwen3-4B-Instruct-2507"
finetune_output_dir="${ROOT_DIR}/models/open-unlearning"

finetune_experiment="finetune/tofu/default"
eval_experiment="eval/tofu/default"

gpu_count=2
per_device_train_batch_size=4
gradient_accumulation_steps=8   # 4×8×2 = 32 on 2 GPUs
num_train_epochs=5
learning_rate=5e-5
warmup_epochs=1.0
weight_decay=0.01
optim="adamw_8bit"
accel_config="configs/accelerate/default_config.yaml"

splits=(
    "forget10 holdout10 retain90"
    "forget05 holdout05 retain95"
)

# ── Finetune ──────────────────────────────────────────────────────────────────
run_finetune() {
    local task_name="$1"
    local dataset_name="$2"

    local train_done_marker="${finetune_output_dir}/${task_name}/config.json"

    if [[ -f "$train_done_marker" ]]; then
        echo ">>> Skipping finetune ${task_name} (already done)"
        return
    fi

    echo ""
    echo ">>> Finetuning: ${task_name} on ${dataset_name}"
    echo "    lr=${learning_rate} epochs=${num_train_epochs}"

    grab_gpus "$gpu_count" 40000 80 4
    local num_processes
    num_processes=$(count_visible_gpus)
    local port
    port=$(get_free_port)

    echo "    world=${num_processes}  bs=${per_device_train_batch_size}×${gradient_accumulation_steps}×${num_processes}=$((per_device_train_batch_size * gradient_accumulation_steps * num_processes))"

    accelerate launch \
        --config_file "$accel_config" \
        --num_processes "$num_processes" \
        --main_process_port "$port" \
        src/train.py --config-name=train.yaml \
        experiment="$finetune_experiment" \
        task_name="$task_name" \
        model="$model" \
        "model.model_args.pretrained_model_name_or_path=${base_model_path}" \
        "data/datasets@data.train=TOFU_QA_retain" \
        "data.train.TOFU_QA_retain.args.hf_args.name=${dataset_name}" \
        "paths.output_dir=${finetune_output_dir}/${task_name}" \
        trainer.args.eval_strategy=no \
        "trainer.args.learning_rate=${learning_rate}" \
        trainer.args.num_train_epochs="$num_train_epochs" \
        trainer.args.per_device_train_batch_size="$per_device_train_batch_size" \
        trainer.args.gradient_accumulation_steps="$gradient_accumulation_steps" \
        trainer.args.optim="$optim" \
        ++trainer.args.weight_decay="$weight_decay" \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.warmup_epochs="$warmup_epochs"

    sleep 5
}

# ── Eval ──────────────────────────────────────────────────────────────────────
run_eval() {
    local task_name="$1"
    local forget_split="$2"
    local holdout_split="$3"
    local model_path="$4"
    local retain_csv_path="${5:-}"   # CSV with ideal model's forget_Truth_Ratio

    local summary_file="saves/eval/${task_name}/TOFU_SUMMARY.json"

    if [[ -f "$summary_file" ]]; then
        echo ">>> Skipping eval ${task_name} (already done)"
        return
    fi

    echo ""
    echo ">>> Evaluating: ${task_name} on ${forget_split}"

    local retain_args=()
    if [[ -n "$retain_csv_path" ]]; then
        retain_args=("retain_logs_path=${retain_csv_path}")
        echo "    retain_logs: ${retain_csv_path}"
    fi

    grab_gpus 1 40000 80 4
    python src/eval.py --config-name=eval.yaml \
        experiment="$eval_experiment" \
        forget_split="$forget_split" \
        holdout_split="$holdout_split" \
        task_name="$task_name" \
        model="$model" \
        "model.model_args.pretrained_model_name_or_path=${model_path}" \
        "paths.output_dir=saves/eval/${task_name}" \
        "${retain_args[@]+"${retain_args[@]}"}" \
        +overwrite=true

    sleep 5
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "TOFU Qwen3-4B Finetuning (lr=5e-5)"
echo "Date: 2026-06-08"
echo "Model: ${model}"
echo "Datasets: full, retain90, retain95"
echo "============================================"
echo ""

# ── 1. Finetune on full, retain90, retain95 ───────────────────────────────
run_finetune "tofu_${model}_full_lr5e5" "full"
run_finetune "tofu_${model}_retain90_lr5e5" "retain90"
run_finetune "tofu_${model}_retain95_lr5e5" "retain95"

# ── 2. Eval retain models (produce TOFU_EVAL for KS-test reference) ────────
run_eval \
    "tofu_${model}_retain90_lr5e5_forget10" \
    "forget10" "holdout10" \
    "${finetune_output_dir}/tofu_${model}_retain90_lr5e5"

run_eval \
    "tofu_${model}_retain95_lr5e5_forget05" \
    "forget05" "holdout05" \
    "${finetune_output_dir}/tofu_${model}_retain95_lr5e5"

# ── 3. Eval full model (with retain reference for forget_quality) ──────────
run_eval \
    "tofu_${model}_full_lr5e5_forget10" \
    "forget10" "holdout10" \
    "${finetune_output_dir}/tofu_${model}_full_lr5e5" \
    "saves/eval/tofu_${model}_retain90_lr5e5_forget10/TOFU_EVAL.json"

run_eval \
    "tofu_${model}_full_lr5e5_forget05" \
    "forget05" "holdout05" \
    "${finetune_output_dir}/tofu_${model}_full_lr5e5" \
    "saves/eval/tofu_${model}_retain95_lr5e5_forget05/TOFU_EVAL.json"

echo ""
echo "TOFU Qwen3-4B finetuning (lr=5e-5) complete."
echo ""
echo "Models:"
echo "  ${finetune_output_dir}/tofu_${model}_{full,retain90,retain95}_lr{2e5,5e5}/"
echo ""
echo "Evals:"
echo "  saves/eval/tofu_${model}_full_lr{2e5,5e5}_{forget10,forget05}/TOFU_SUMMARY.json"
echo "  saves/eval/tofu_${model}_retain90_lr{2e5,5e5}_forget10/TOFU_EVAL.json"
echo "  saves/eval/tofu_${model}_retain95_lr{2e5,5e5}_forget05/TOFU_EVAL.json"
