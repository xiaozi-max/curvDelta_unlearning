#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Shared driver for the CurvLoRA re-run scripts (01..04).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#   clora_bootstrap      # cd to repo root, source server_env.sh, set HF_HOME
#   clora_defaults       # populate every CLORA_* knob with its default
#   ... override knobs, then call clora_run ...
#
#   CLORA_GPU_IDS="1 2" bash scripts/curvLoRA/01-tofu-3b-f10-simnpo-er-sweep.sh
#
# Conventions inherited from the newest script generation (26-08-10 / 26-09-*):
#   static CurvLoRA, KL-Fisher curvature collection, linear scheduler,
#   warmup_epochs=1.0, resume-safe (skips a run whose eval summary exists).
#
# Trainer method_args are ALWAYS passed explicitly on the CLI. This is
# deliberate: the 26-06-04 MUSE scripts silently fell back to the config
# defaults (NPO beta=0.1 instead of 0.01, SimNPO gamma=0.125 instead of 1.0).
# ══════════════════════════════════════════════════════════════════════════════

# ── Repo root ─────────────────────────────────────────────────────────────────
CLORA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Bootstrap ─────────────────────────────────────────────────────────────────
clora_bootstrap() {
    # grab_gpus() in server_env.sh resolves common_utils/get_gpu_ids.py
    # relative to cwd, so pin cwd to the repo root.
    cd "$CLORA_REPO_ROOT"
    # shellcheck disable=SC1090
    source "${CLORA_REPO_ROOT}/common_utils/server_env.sh"
    export HF_HOME="${ROOT_DIR}/.cache/huggingface"
    mkdir -p saves/logs
}

# ── Defaults ──────────────────────────────────────────────────────────────────
clora_defaults() {
    # Benchmark switch: "tofu" | "muse"
    CLORA_BENCH="tofu"
    CLORA_DATA_SPLIT="Books"            # muse only
    CLORA_SUMMARY="TOFU_SUMMARY.json"   # muse: MUSE_SUMMARY.json

    # Model — set in each script
    CLORA_MODEL=""
    CLORA_MODEL_TAG=""
    CLORA_MODEL_PATH=""
    CLORA_EXPERIMENT="unlearn/tofu/default"
    CLORA_EVAL_EXPERIMENT="eval/tofu/default"

    # Split — reassigned per loop iteration
    CLORA_FORGET_SPLIT="forget10"
    CLORA_HOLDOUT_SPLIT="holdout10"
    CLORA_RETAIN_SPLIT="retain90"
    CLORA_RETAIN_LOGS=""

    # CurvLoRA
    CLORA_PEFT_R=32
    CLORA_PEFT_ALPHA=64
    CLORA_PEFT_DROPOUT=0.05
    CLORA_ER=0.9
    CLORA_PROJ="kfac_top"
    CLORA_RR=1.0
    CLORA_CHUNKS=1
    CLORA_CBS=4

    # Training
    CLORA_EPOCHS=10
    CLORA_PBS=4
    CLORA_GA=4
    CLORA_OPTIM="adamw_8bit"
    CLORA_WARMUP=1.0
    CLORA_LR_SCHED="linear"
    CLORA_ACCEL="configs/accelerate/zero2_config.yaml"

    # GPU grabbing
    CLORA_GPUS=2
    CLORA_GPU_MEM=40000
    CLORA_UTIL=100
    CLORA_MAX_PROC=4
    CLORA_GPU_IDS="${CLORA_GPU_IDS:-$DEFAULT_GPU_IDS}"

    # Trainer hyperparameters (explicit -- see header note)
    CLORA_SIMNPO_DELTA=0.0
    CLORA_SIMNPO_BETA=4.5
    CLORA_SIMNPO_ALPHA=1.0
    CLORA_SIMNPO_GAMMA=1.0
    CLORA_NPO_BETA=0.1
    CLORA_NPO_ALPHA=1.0
    CLORA_NPO_GAMMA=1.0
    CLORA_GRADDIFF_GAMMA=1.0
    CLORA_GRADDIFF_ALPHA=1.0
}

# ── Small helpers ─────────────────────────────────────────────────────────────
clora_require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Missing $2: $1" >&2
        echo "Create the required retain reference eval before running." >&2
        exit 1
    fi
}

clora_free_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

clora_gpu_count() {
    awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES not set}"
}

# ── Trainer method args ───────────────────────────────────────────────────────
# Sets the global array CLORA_METHOD_ARGS for the given trainer.
clora_set_trainer_args() {
    CLORA_METHOD_ARGS=()
    case "$1" in
        SimNPO)
            CLORA_METHOD_ARGS=(
                "trainer.method_args.delta=${CLORA_SIMNPO_DELTA}"
                "trainer.method_args.beta=${CLORA_SIMNPO_BETA}"
                "trainer.method_args.alpha=${CLORA_SIMNPO_ALPHA}"
                "trainer.method_args.gamma=${CLORA_SIMNPO_GAMMA}"
            )
            ;;
        NPO)
            CLORA_METHOD_ARGS=(
                "trainer.method_args.beta=${CLORA_NPO_BETA}"
                "trainer.method_args.alpha=${CLORA_NPO_ALPHA}"
                "trainer.method_args.gamma=${CLORA_NPO_GAMMA}"
            )
            ;;
        GradDiff)
            CLORA_METHOD_ARGS=(
                "trainer.method_args.gamma=${CLORA_GRADDIFF_GAMMA}"
                "trainer.method_args.alpha=${CLORA_GRADDIFF_ALPHA}"
            )
            ;;
        *)
            echo "clora_set_trainer_args: unsupported trainer '$1'" >&2
            return 1
            ;;
    esac
}

# ── Eval ──────────────────────────────────────────────────────────────────────
clora_eval() {
    local task="$1"

    if [[ -f "saves/unlearn/${task}/evals/${CLORA_SUMMARY}" ]]; then
        echo ">>> [skip eval] ${task} (already done)"
        return
    fi

    grab_gpus 1 "$CLORA_GPU_MEM" "$CLORA_UTIL" "$CLORA_MAX_PROC" "$CLORA_GPU_IDS"

    if [[ "$CLORA_BENCH" == "muse" ]]; then
        python src/eval.py --config-name=eval.yaml \
            experiment="$CLORA_EVAL_EXPERIMENT" \
            data_split="$CLORA_DATA_SPLIT" \
            model="$CLORA_MODEL" \
            task_name="$task" \
            "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task}" \
            "paths.output_dir=saves/unlearn/${task}/evals" \
            "retain_logs_path=${CLORA_RETAIN_LOGS}" \
            eval.muse.overwrite=true
    else
        python src/eval.py --config-name=eval.yaml \
            experiment="$CLORA_EVAL_EXPERIMENT" \
            forget_split="$CLORA_FORGET_SPLIT" \
            holdout_split="$CLORA_HOLDOUT_SPLIT" \
            model="$CLORA_MODEL" \
            task_name="$task" \
            "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task}" \
            "paths.output_dir=saves/unlearn/${task}/evals" \
            "retain_logs_path=${CLORA_RETAIN_LOGS}" \
            +overwrite=true
    fi
    sleep 5
}

# ── Train + eval one run ──────────────────────────────────────────────────────
#   clora_run --task NAME --trainer SIMNPO --lr 1e-4 [--er 0.9] [--proj kfac_top]
#             [--rr 1.0] [--chunks 1] [--cbs 4] [--epochs 10] [--gpus 2] [--gpu-mem 40000]
clora_run() {
    local task="" trainer="" lr=""
    local er="$CLORA_ER" proj="$CLORA_PROJ"
    local chunks="$CLORA_CHUNKS" cbs="$CLORA_CBS" epochs="$CLORA_EPOCHS"
    local rr="$CLORA_RR"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task)    task="$2";      shift 2 ;;
            --trainer) trainer="$2";   shift 2 ;;
            --lr)      lr="$2";        shift 2 ;;
            --er)      er="$2";        shift 2 ;;
            --proj)    proj="$2";      shift 2 ;;
            --rr)      rr="$2";        shift 2 ;;
            --chunks)  chunks="$2";    shift 2 ;;
            --cbs)     cbs="$2";       shift 2 ;;
            --epochs)  epochs="$2";    shift 2 ;;
            --gpus)    CLORA_GPUS="$2";    shift 2 ;;
            --gpu-mem) CLORA_GPU_MEM="$2"; shift 2 ;;
            *) echo "clora_run: unknown option '$1'" >&2; return 1 ;;
        esac
    done

    if [[ -z "$task" || -z "$trainer" || -z "$lr" ]]; then
        echo "clora_run: --task, --trainer and --lr are required" >&2
        return 1
    fi

    if [[ -f "saves/unlearn/${task}/evals/${CLORA_SUMMARY}" ]]; then
        echo ">>> [skip] ${task} (already done)"
        return
    fi

    clora_set_trainer_args "$trainer"

    echo ""
    echo ">>> ${task}"
    echo "    trainer=${trainer} lr=${lr} er=${er} proj=${proj} rr=${rr} chunks=${chunks} cbs=${cbs} epochs=${epochs}"

    if [[ ! -f "saves/unlearn/${task}/adapter_config.json" ]]; then
        local port nproc
        port=$(clora_free_port)
        grab_gpus "$CLORA_GPUS" "$CLORA_GPU_MEM" "$CLORA_UTIL" "$CLORA_MAX_PROC" "$CLORA_GPU_IDS"
        nproc=$(clora_gpu_count)

        echo "    world=${nproc} bs=${CLORA_PBS}x${CLORA_GA}x${nproc}=$((CLORA_PBS * CLORA_GA * nproc))"

        local data_args=()
        if [[ "$CLORA_BENCH" == "muse" ]]; then
            data_args=("data_split=${CLORA_DATA_SPLIT}")
        fi

        local sched_args warmup_args
        if [[ "$CLORA_BENCH" == "muse" ]]; then
            sched_args=("trainer.args.lr_scheduler_type=${CLORA_LR_SCHED}")
            warmup_args=("+trainer.args.warmup_epochs=${CLORA_WARMUP}")
        else
            sched_args=("+trainer.args.lr_scheduler_type=${CLORA_LR_SCHED}")
            warmup_args=("trainer.args.warmup_epochs=${CLORA_WARMUP}")
        fi

        accelerate launch \
            --config_file "$CLORA_ACCEL" \
            --num_processes "$nproc" \
            --main_process_port "$port" \
            src/train.py --config-name=unlearn.yaml \
            experiment="$CLORA_EXPERIMENT" \
            trainer="$trainer" \
            task_name="$task" \
            model="$CLORA_MODEL" \
            "model.model_args.pretrained_model_name_or_path=${CLORA_MODEL_PATH}" \
            forget_split="$CLORA_FORGET_SPLIT" \
            retain_split="$CLORA_RETAIN_SPLIT" \
            retain_logs_path="$CLORA_RETAIN_LOGS" \
            "${data_args[@]+"${data_args[@]}"}" \
            peft=curvLoRA \
            "peft.r=${CLORA_PEFT_R}" \
            "peft.lora_alpha=${CLORA_PEFT_ALPHA}" \
            "peft.lora_dropout=${CLORA_PEFT_DROPOUT}" \
            curvLoRA=default \
            "${CLORA_METHOD_ARGS[@]}" \
            trainer.method_args.curvLoRA_configs.cache_mode=static \
            "trainer.method_args.curvLoRA_configs.energy_ratio=${er}" \
            "trainer.method_args.curvLoRA_configs.projection_mode=${proj}" \
            trainer.method_args.curvLoRA_configs.retain_loss_type=KL \
            trainer.method_args.curvLoRA_configs.fisher_sample_seed=0 \
            trainer.method_args.curvLoRA_configs.fisher_temperature=1.0 \
            "trainer.method_args.curvLoRA_configs.collection_batch_size=${cbs}" \
            "trainer.method_args.curvLoRA_configs.num_chunks=${chunks}" \
            "trainer.method_args.curvLoRA_configs.retain_ratio=${rr}" \
            "trainer.args.learning_rate=${lr}" \
            "trainer.args.num_train_epochs=${epochs}" \
            "trainer.args.per_device_train_batch_size=${CLORA_PBS}" \
            "trainer.args.gradient_accumulation_steps=${CLORA_GA}" \
            "trainer.args.optim=${CLORA_OPTIM}" \
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

    clora_eval "$task"
}
