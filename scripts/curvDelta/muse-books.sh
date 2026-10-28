#!/bin/bash
# Experiment: MUSE Books CurvDelta — GradDiff / NPO / SimNPO LR sweep
# Date: 2026-08-10
# Doc: docs/experiments/26-08-10-muse-books-curvature-dense-trainers-lr-sweep.md
#
# Matrix:
#   trainers: GradDiff | NPO | SimNPO
#   lr in {1e-5, 2e-5, 5e-5, 1e-4}
#   static CurvDelta: cache_mode=static, energy_ratio=0.7
#   Total: 12 unlearning runs.
#
# GPU allocation (full-rank like FFT; 4x L40 48GB, ZeRO-3):
#   MUSE sequences are 2048 tokens -> per_device=1, grad_accum=8, 4 GPUs -> effective batch 32.

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

model="Llama-2-7b-hf"
model_path="${ROOT_DIR}/models/muse-bench/MUSE-Books_target"
experiment="unlearn/muse/default"
eval_experiment="eval/muse/default"
data_split="Books"
forget_split="forget"
retain_split="retain1"
retain_logs="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
# MUSE uses ZeRO-3 + CPU param offload (offload_param only; optimizer stays HF-owned).
# TOFU-8B keeps the default config (no offload).
# accel_config="configs/accelerate/param_offload_config.yaml"
accel_config="configs/accelerate/default_config.yaml"
# CurvDelta settings (full-rank adapter; memory profile like FFT).
num_gpus=2
per_device_train_batch_size=1
gradient_accumulation_steps=16
energy_ratio=0.7
collection_batch_size=1
num_chunks=2

num_train_epochs=5
warmup_epochs=1.0
optim="adamw_8bit"

# Trainer hyperparameters.  MUSE NPO uses beta=0.01 (suited to 2048-token sequences).
npo_beta=0.01
npo_alpha=1.0
npo_gamma=1.0
simnpo_delta=0.0
simnpo_beta=4.5      # default
simnpo_alpha=1.0
simnpo_gamma=1.0
graddiff_gamma=1.0
graddiff_alpha=1.0

graddiff_lr_tags=(1e5 2e5 5e5)
graddiff_lrs=(1e-5 2e-5 5e-5)
npo_lr_tags=(2e5 5e5 1e4)
npo_lrs=(2e-5 5e-5 1e-4)
simnpo_lr_tags=(5e5 1e4 2e4)
simnpo_lrs=(5e-5 1e-4 2e-4)

run_eval() {
    local task_name="$1"

    if [[ -f "saves/unlearn/${task_name}/evals/MUSE_SUMMARY.json" ]]; then
        echo ">>> Skipping ${task_name} eval (already complete)"
        return
    fi

    grab_gpus 1 40000 100 4
    python src/eval.py \
        --config-name=eval.yaml \
        experiment="$eval_experiment" \
        data_split="$data_split" \
        model="$model" \
        task_name="$task_name" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
        paths.output_dir="saves/unlearn/${task_name}/evals" \
        retain_logs_path="$retain_logs" \
        eval.muse.overwrite=true
}

run_curvdelta() {
    local trainer="$1"
    local lr="$2"
    local lr_tag="$3"
    local task_name="muse_${model}_${data_split}_${trainer}_curvDelta_er07_adamw_lr${lr_tag}"

    if [[ -f "saves/unlearn/${task_name}/evals/MUSE_SUMMARY.json" ]]; then
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

        grab_gpus "$num_gpus" 40000 100 4
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
            data_split="$data_split" \
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
            +trainer.args.warmup_epochs="$warmup_epochs" \
            trainer.args.lr_scheduler_type=linear \
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

    run_eval "$task_name"
    sleep 5
}

require_file "$retain_logs" "${model} ${data_split} retain logs"

for trainer in GradDiff NPO SimNPO; do
    case "$trainer" in
        GradDiff) lr_tags=("${graddiff_lr_tags[@]}"); lr_vals=("${graddiff_lrs[@]}") ;;
        NPO) lr_tags=("${npo_lr_tags[@]}"); lr_vals=("${npo_lrs[@]}") ;;
        SimNPO) lr_tags=("${simnpo_lr_tags[@]}"); lr_vals=("${simnpo_lrs[@]}") ;;
    esac
    for i in "${!lr_vals[@]}"; do
        run_curvdelta "$trainer" "${lr_vals[$i]}" "${lr_tags[$i]}"
    done
done

echo ""
echo ">>> MUSE Books CurvDelta trainers LR sweep completed."
echo "Results under saves/unlearn/muse_${model}_${data_split}_{NPO,SimNPO,GradDiff}_curvDelta_er07_adamw_lr*/evals/"
