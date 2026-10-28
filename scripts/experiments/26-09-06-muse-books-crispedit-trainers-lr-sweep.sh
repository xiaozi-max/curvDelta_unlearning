#!/bin/bash
# Experiment: MUSE Books CrispEdit — GradDiff / NPO / SimNPO LR sweep
# Date: 2026-09-06
# Doc: docs/experiments/26-09-06-muse-books-crispedit-trainers-lr-sweep.md
#
# MUSE Books with GradDiff, NPO, and SimNPO using effective batch size 32.

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
data_split=Books
forget_split="forget"
retain_split="retain1"
retain_logs="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
accel_config="configs/accelerate/ddp_config.yaml"

# CrispEdit settings (direct base-weight, optimizer-space projection).
energy_threshold=0.9
retain_n_samples=256
collection_batch_size=1
num_chunks=8
# Layers override for a reduced-scope smoke run (empty = all layers). Edit here if OOM.
crispedit_layers="[]"

num_gpus=4
per_device_train_batch_size=1
gradient_accumulation_steps=8

num_train_epochs=5
warmup_epochs=1.0
optim="adamw_8bit"   # ignored by CrispEditAdamW (always fp32); kept for parity in logs

# Trainer hyperparameters. MUSE NPO uses beta=0.01 (2048-token sequences).
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

run_crispedit() {
    local trainer="$1"
    local lr="$2"
    local lr_tag="$3"
    local task_name="muse_${model}_${data_split}_${trainer}_crispedit_er09_adamw_lr${lr_tag}"

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
    echo ">>> ${task_name}: ${trainer} + static CrispEdit er=${energy_threshold}, lr=${lr}"

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
            peft=crispedit \
            crispedit=default \
            trainer.method_args.crispedit_configs.energy_threshold="$energy_threshold" \
            trainer.method_args.crispedit_configs.layers="$crispedit_layers" \
            trainer.method_args.crispedit_configs.retain_n_samples="$retain_n_samples" \
            trainer.method_args.crispedit_configs.collection_batch_size="$collection_batch_size" \
            trainer.method_args.crispedit_configs.num_chunks="$num_chunks" \
            "${method_args[@]}" \
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
        run_crispedit "$trainer" "${lr_vals[$i]}" "${lr_tags[$i]}"
    done
done

echo ""
echo ">>> MUSE Books CrispEdit trainers LR sweep completed."
echo "Results under saves/unlearn/muse_${model}_${data_split}_{NPO,SimNPO,GradDiff}_crispedit_er09_adamw_lr*/evals/"
