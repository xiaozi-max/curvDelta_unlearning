#!/usr/bin/env bash
# Configure data/cache storage and an explicit GPU pool without host detection.
_ENV_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-${_ENV_REPO_ROOT}/data}"
export ROOT_DIR="$PROJECT_DATA_ROOT"
export HF_HOME="${HF_HOME:-${PROJECT_DATA_ROOT}/.cache/huggingface}"
export DEFAULT_GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Compatibility interface used by the public launchers. IDs may use commas/spaces.
# Memory/utilization thresholds are advisory inputs retained from source launchers;
# allocation is explicit and never polls or races for another job's GPUs.
grab_gpus() {
    local count="$1" pool="${5:-$DEFAULT_GPU_IDS}"
    export CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES="$(python "${_ENV_REPO_ROOT}/common_utils/get_gpu_ids.py" --count "$count" --ids "$pool")" || return
}
