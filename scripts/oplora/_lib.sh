#!/usr/bin/env bash
# Shared driver for OPLoRA runs. Same launch/eval/skip pattern as
# scripts/curvLoRA/_lib.sh, without K-FAC collection.
#
# ZeRO-2 only: configs/accelerate/zero2_config.yaml. OPLoRA folds the
# projection into lora_A/lora_B, which requires unsharded adapter weights.

OPLORA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

oplora_bootstrap() {
    cd "$OPLORA_REPO_ROOT"
    # shellcheck disable=SC1090
    source "${OPLORA_REPO_ROOT}/common_utils/server_env.sh"
    export HF_HOME="${ROOT_DIR}/.cache/huggingface"
    mkdir -p saves/logs
}

oplora_defaults() {
    OPLORA_BENCH="tofu"
    OPLORA_DATA_SPLIT="Books"
    OPLORA_SUMMARY="TOFU_SUMMARY.json"

    OPLORA_MODEL=""
    OPLORA_MODEL_TAG=""
    OPLORA_MODEL_PATH=""
    OPLORA_EXPERIMENT="unlearn/tofu/default"
    OPLORA_EVAL_EXPERIMENT="eval/tofu/default"

    OPLORA_FORGET_SPLIT="forget10"
    OPLORA_HOLDOUT_SPLIT="holdout10"
    OPLORA_RETAIN_SPLIT="retain90"
    OPLORA_RETAIN_LOGS=""

    OPLORA_PEFT_R=32
    OPLORA_PEFT_ALPHA=64
    OPLORA_PEFT_DROPOUT=0.05
    OPLORA_K=16

    OPLORA_EPOCHS=10
    OPLORA_PBS=4
    OPLORA_GA=4
    OPLORA_OPTIM="adamw_8bit"
    OPLORA_WARMUP=1.0
    OPLORA_LR_SCHED="linear"
    OPLORA_ACCEL="configs/accelerate/zero2_config.yaml"

    OPLORA_GPUS=2
    OPLORA_GPU_MEM=40000
    OPLORA_UTIL=100
    OPLORA_MAX_PROC=4
    OPLORA_GPU_IDS="${OPLORA_GPU_IDS:-$DEFAULT_GPU_IDS}"

    OPLORA_SIMNPO_DELTA=0.0
    OPLORA_SIMNPO_BETA=4.5
    OPLORA_SIMNPO_ALPHA=1.0
    OPLORA_SIMNPO_GAMMA=1.0
    OPLORA_NPO_BETA=0.1
    OPLORA_NPO_ALPHA=1.0
    OPLORA_NPO_GAMMA=1.0
    OPLORA_GRADDIFF_GAMMA=1.0
    OPLORA_GRADDIFF_ALPHA=1.0
}

oplora_require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing $2: $1" >&2
        echo "Create the required retain reference eval before running." >&2
        exit 1
    fi
}

oplora_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

oplora_gpu_count() {
    awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES not set}"
}

oplora_set_trainer_args() {
    OPLORA_METHOD_ARGS=()
    case "$1" in
        SimNPO)
            OPLORA_METHOD_ARGS=(
                "trainer.method_args.delta=${OPLORA_SIMNPO_DELTA}"
                "trainer.method_args.beta=${OPLORA_SIMNPO_BETA}"
                "trainer.method_args.alpha=${OPLORA_SIMNPO_ALPHA}"
                "trainer.method_args.gamma=${OPLORA_SIMNPO_GAMMA}"
            )
            ;;
        NPO)
            OPLORA_METHOD_ARGS=(
                "trainer.method_args.beta=${OPLORA_NPO_BETA}"
                "trainer.method_args.alpha=${OPLORA_NPO_ALPHA}"
                "trainer.method_args.gamma=${OPLORA_NPO_GAMMA}"
            )
            ;;
        GradDiff)
            OPLORA_METHOD_ARGS=(
                "trainer.method_args.gamma=${OPLORA_GRADDIFF_GAMMA}"
                "trainer.method_args.alpha=${OPLORA_GRADDIFF_ALPHA}"
            )
            ;;
        *)
            echo "oplora_set_trainer_args: unsupported trainer '$1'" >&2
            return 1
            ;;
    esac
}

oplora_eval() {
    local task="$1"

    if [[ -f "saves/unlearn/${task}/evals/${OPLORA_SUMMARY}" ]]; then
        echo ">>> [skip eval] ${task} (already done)"
        return
    fi

    grab_gpus 1 "$OPLORA_GPU_MEM" "$OPLORA_UTIL" "$OPLORA_MAX_PROC" "$OPLORA_GPU_IDS"

    if [[ "$OPLORA_BENCH" == "muse" ]]; then
        python src/eval.py --config-name=eval.yaml \
            experiment="$OPLORA_EVAL_EXPERIMENT" \
            data_split="$OPLORA_DATA_SPLIT" \
            model="$OPLORA_MODEL" \
            task_name="$task" \
            "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task}" \
            "paths.output_dir=saves/unlearn/${task}/evals" \
            "retain_logs_path=${OPLORA_RETAIN_LOGS}" \
            eval.muse.overwrite=true
    else
        python src/eval.py --config-name=eval.yaml \
            experiment="$OPLORA_EVAL_EXPERIMENT" \
            forget_split="$OPLORA_FORGET_SPLIT" \
            holdout_split="$OPLORA_HOLDOUT_SPLIT" \
            model="$OPLORA_MODEL" \
            task_name="$task" \
            "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task}" \
            "paths.output_dir=saves/unlearn/${task}/evals" \
            "retain_logs_path=${OPLORA_RETAIN_LOGS}" \
            +overwrite=true
    fi
    sleep 5
}

# oplora_run --task NAME --trainer SimNPO --lr 1e-4
oplora_run() {
    local task="" trainer="" lr=""
    local epochs="$OPLORA_EPOCHS"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task)    task="$2"; shift 2 ;;
            --trainer) trainer="$2"; shift 2 ;;
            --lr)      lr="$2"; shift 2 ;;
            --epochs)  epochs="$2"; shift 2 ;;
            --gpus)    OPLORA_GPUS="$2"; shift 2 ;;
            --gpu-mem) OPLORA_GPU_MEM="$2"; shift 2 ;;
            *) echo "oplora_run: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    if [[ -z "$task" || -z "$trainer" || -z "$lr" ]]; then
        echo "oplora_run: --task, --trainer and --lr are required" >&2
        return 1
    fi

    if [[ -f "saves/unlearn/${task}/evals/${OPLORA_SUMMARY}" ]]; then
        echo ">>> [skip] ${task} (already done)"
        return
    fi

    oplora_set_trainer_args "$trainer"

    echo ""
    echo ">>> ${task}"
    echo "    trainer=${trainer} lr=${lr} k=${OPLORA_K} r=${OPLORA_PEFT_R} epochs=${epochs}"

    if [[ ! -f "saves/unlearn/${task}/adapter_config.json" ]]; then
        local port nproc
        port=$(oplora_free_port)
        grab_gpus "$OPLORA_GPUS" "$OPLORA_GPU_MEM" "$OPLORA_UTIL" "$OPLORA_MAX_PROC" "$OPLORA_GPU_IDS"
        nproc=$(oplora_gpu_count)

        echo "    world=${nproc} bs=${OPLORA_PBS}x${OPLORA_GA}x${nproc}=$((OPLORA_PBS * OPLORA_GA * nproc))"

        local data_args=()
        if [[ "$OPLORA_BENCH" == "muse" ]]; then
            data_args=("data_split=${OPLORA_DATA_SPLIT}")
        fi

        # TOFU configs already have warmup_epochs and lack lr_scheduler_type.
        # MUSE configs are the other way around. Do not swap the '+' prefix.
        local sched_args warmup_args
        if [[ "$OPLORA_BENCH" == "muse" ]]; then
            sched_args=("trainer.args.lr_scheduler_type=${OPLORA_LR_SCHED}")
            warmup_args=("+trainer.args.warmup_epochs=${OPLORA_WARMUP}")
        else
            sched_args=("+trainer.args.lr_scheduler_type=${OPLORA_LR_SCHED}")
            warmup_args=("trainer.args.warmup_epochs=${OPLORA_WARMUP}")
        fi

        accelerate launch \
            --config_file "$OPLORA_ACCEL" \
            --num_processes "$nproc" \
            --main_process_port "$port" \
            src/train.py --config-name=unlearn.yaml \
            experiment="$OPLORA_EXPERIMENT" \
            trainer="$trainer" \
            task_name="$task" \
            model="$OPLORA_MODEL" \
            "model.model_args.pretrained_model_name_or_path=${OPLORA_MODEL_PATH}" \
            forget_split="$OPLORA_FORGET_SPLIT" \
            retain_split="$OPLORA_RETAIN_SPLIT" \
            retain_logs_path="$OPLORA_RETAIN_LOGS" \
            "${data_args[@]+"${data_args[@]}"}" \
            peft=oplora \
            "peft.r=${OPLORA_PEFT_R}" \
            "peft.lora_alpha=${OPLORA_PEFT_ALPHA}" \
            "peft.lora_dropout=${OPLORA_PEFT_DROPOUT}" \
            "peft.oplora_k=${OPLORA_K}" \
            "${OPLORA_METHOD_ARGS[@]}" \
            "trainer.args.learning_rate=${lr}" \
            "trainer.args.num_train_epochs=${epochs}" \
            "trainer.args.per_device_train_batch_size=${OPLORA_PBS}" \
            "trainer.args.gradient_accumulation_steps=${OPLORA_GA}" \
            "trainer.args.optim=${OPLORA_OPTIM}" \
            ++trainer.args.weight_decay=0.01 \
            "${sched_args[@]}" \
            "${warmup_args[@]}" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.eval_strategy=no

        sleep 5
    else
        echo "    training done, skipping to eval."
    fi

    oplora_eval "$task"
}
