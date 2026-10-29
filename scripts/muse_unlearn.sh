#!/usr/bin/env bash
# Optional MUSE Books baseline reproduction. The target and retrain checkpoints
# can instead be downloaded with: bash common_utils/download_cli.sh muse
set -euo pipefail
source "common_utils/server_env.sh"

data_split=Books
model=Llama-2-7b-hf
model_path="${ROOT_DIR}/models/muse-bench/MUSE-Books_target"
retain_logs="saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json"
trainers=(GradAscent GradDiff NPO SimNPO)

grab_gpus 2 40000
for trainer in "${trainers[@]}"; do
    task_name="muse_${model}_${data_split}_${trainer}"
    accelerate launch --config_file configs/accelerate/default_config.yaml \
        src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/muse/default.yaml \
        model="$model" \
        model.model_args.pretrained_model_name_or_path="$model_path" \
        data_split="$data_split" \
        trainer="$trainer" \
        task_name="$task_name" \
        retain_logs_path="$retain_logs" \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.gradient_accumulation_steps=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

    python src/eval.py --config-name=eval.yaml \
        experiment=eval/muse/default.yaml \
        data_split="$data_split" \
        task_name="$task_name" \
        model="$model" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
        paths.output_dir="saves/unlearn/${task_name}/evals" \
        retain_logs_path="$retain_logs"
done
