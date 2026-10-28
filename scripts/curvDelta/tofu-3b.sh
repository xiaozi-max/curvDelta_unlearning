#!/bin/bash
# Experiment: TOFU 3B CurvDelta — GradDiff / NPO / SimNPO LR sweep
# Date: 2026-08-10
# Doc: docs/experiments/26-08-10-tofu-3b-curvature-dense-trainers-lr-sweep.md
#
# Matrix:
#   splits: forget10/retain90/holdout10 (f10), forget05/retain95/holdout05 (f05)
#   trainers: GradDiff | NPO | SimNPO   (NPO currently commented out in the run loop)
#   lr in {1e-5, 2e-5, 5e-5, 1e-4}
#   static CurvDelta: cache_mode=static, energy_ratio=0.9
#   Total: 16 unlearning runs = 2 splits x 2 trainers x 4 LR values (24 with NPO).
#
# GPU allocation (full-rank like FFT; 1x L40 48GB, ZeRO-3):
#   per_device=2, grad_accum=16, 1 GPU -> effective batch 32.

set -euo pipefail

source "common_utils/server_env.sh"

mkdir -p saves/logs

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

model="Llama-3.2-3B-Instruct"
model_tag="3B"
model_path="${ROOT_DIR}/models/open-unlearning/tofu_${model}_full"
experiment="unlearn/tofu/default"
eval_experiment="eval/tofu/default"
accel_config="configs/accelerate/default_config.yaml"

splits=(
    "forget10 holdout10 retain90 f10"
    "forget05 holdout05 retain95 f05"
)

# CurvDelta settings (full-rank adapter; memory profile like FFT).
# 3B fits on a single L40 48GB under ZeRO-3; increase num_gpus if OOM.
num_gpus=1
per_device_train_batch_size=2
gradient_accumulation_steps=16
energy_ratio=0.9
collection_batch_size=2
num_chunks=1

num_train_epochs=10
optim="adamw_8bit"

# Trainer hyperparameters.
npo_beta=0.1
npo_alpha=1.0
npo_gamma=1.0
simnpo_delta=0.0
simnpo_beta=4.5      # default
simnpo_alpha=1.0
simnpo_gamma=1.0
graddiff_gamma=1.0
graddiff_alpha=1.0

lr_tags=(1e5 2e5 5e5 1e4)
lr_vals=(1e-5 2e-5 5e-5 1e-4)

run_eval() {
    local task_name="$1"
    local forget_split="$2"
    local holdout_split="$3"
    local retain_logs="$4"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping ${task_name} eval (already complete)"
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

run_curvdelta() {
    local trainer="$1"
    local lr="$2"
    local lr_tag="$3"
    local forget_split="$4"
    local holdout_split="$5"
    local retain_split="$6"
    local split_tag="$7"
    local retain_logs="$8"
    local task_name="tofu_${model_tag}_${split_tag}_${trainer}_curvDelta_er09_adamw_lr${lr_tag}"

    if [[ -f "saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json" ]]; then
        echo ">>> Skipping ${task_name} (already done)"
        return
    fi

    local method_args=()
    case "$trainer" in
        NPO)
            method_args=(
                trainer.method_args.beta="$npo_beta"
                trainer.method_args.alpha="$npo_alpha"
                trainer.method_args.gamma="$npo_gamma"
            )
            ;;
        SimNPO)
            method_args=(
                trainer.method_args.delta="$simnpo_delta"
                trainer.method_args.beta="$simnpo_beta"
                trainer.method_args.alpha="$simnpo_alpha"
                trainer.method_args.gamma="$simnpo_gamma"
            )
            ;;
        GradDiff)
            method_args=(
                trainer.method_args.gamma="$graddiff_gamma"
                trainer.method_args.alpha="$graddiff_alpha"
            )
            ;;
        *)
            echo "ERROR: Unknown trainer: $trainer" >&2
            return 1
            ;;
    esac

    echo ""
    echo ">>> ${task_name}: ${trainer} + static CurvDelta er=${energy_ratio}, lr=${lr}"

    if [[ ! -f "saves/unlearn/${task_name}/config.json" ]]; then
        local port
        port=$(get_free_port)

        grab_gpus "$num_gpus" 40000
        accelerate launch \
            --config_file "$accel_config" \
            --num_processes "$num_gpus" \
            --main_process_port "$port" \
            src/train.py --config-name=unlearn.yaml \
            experiment="$experiment" \
            trainer="$trainer" \
            task_name="$task_name" \
            model="$model" \
            model.model_args.pretrained_model_name_or_path="$model_path" \
            forget_split="$forget_split" \
            retain_split="$retain_split" \
            retain_logs_path="$retain_logs" \
            peft=curvDelta \
            curvDelta=default \
            "${method_args[@]}" \
            trainer.method_args.curvDelta_configs.cache_mode=static \
            trainer.method_args.curvDelta_configs.energy_ratio="$energy_ratio" \
            trainer.method_args.curvDelta_configs.collection_batch_size="$collection_batch_size" \
            trainer.method_args.curvDelta_configs.num_chunks="$num_chunks" \
            trainer.args.learning_rate="$lr" \
            trainer.args.num_train_epochs="$num_train_epochs" \
            trainer.args.per_device_train_batch_size="$per_device_train_batch_size" \
            trainer.args.gradient_accumulation_steps="$gradient_accumulation_steps" \
            trainer.args.optim="$optim" \
            ++trainer.args.weight_decay=0.01 \
            trainer.args.gradient_checkpointing=true \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.eval_strategy=no

        sleep 5
    else
        echo "    Training done, skipping to eval."
    fi

    run_eval "$task_name" "$forget_split" "$holdout_split" "$retain_logs"
    sleep 5
}

for split_entry in "${splits[@]}"; do
    read -r forget_split holdout_split retain_split split_tag <<< "$split_entry"
    retain_logs="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    require_file "$retain_logs" "${model} ${retain_split} retain logs"

    for trainer in NPO SimNPO GradDiff; do
        for i in "${!lr_tags[@]}"; do
            run_curvdelta "$trainer" "${lr_vals[$i]}" "${lr_tags[$i]}" \
                "$forget_split" "$holdout_split" "$retain_split" "$split_tag" "$retain_logs"
        done
    done
done

echo ""
echo ">>> TOFU 3B CurvDelta trainers LR sweep completed."
echo "Results under saves/unlearn/tofu_${model_tag}_{f10,f05}_{NPO,SimNPO,GradDiff}_curvDelta_er09_adamw_lr*/evals/"
