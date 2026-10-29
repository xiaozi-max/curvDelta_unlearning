#!/usr/bin/env bash
# Download the checkpoints and datasets used by the representative experiments.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"

group="${1:-all}"
case "$group" in
    all|tofu|muse) ;;
    *) echo "Usage: bash common_utils/download_cli.sh [all|tofu|muse]" >&2; exit 2 ;;
esac

download_model() {
    local repo_id="$1"
    local local_repo_id="${2:-$repo_id}"
    hf download "$repo_id" \
        --local-dir "${PROJECT_DATA_ROOT}/models/${local_repo_id}" \
        --exclude "*.bin" "*.pth"
}

download_dataset() {
    local repo_id="$1"
    hf download "$repo_id" --repo-type dataset \
        --local-dir "${PROJECT_DATA_ROOT}/datasets/${repo_id}"
}

if [[ "$group" == "all" || "$group" == "tofu" ]]; then
    download_dataset "locuslab/TOFU"
    download_model "open-unlearning/tofu_Llama-3.2-3B-Instruct_full"
    download_model "open-unlearning/tofu_Llama-3.1-8B-Instruct_full"
    download_model "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90"
    download_model "open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95"
    download_model "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90"
    download_model "open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95"
fi

if [[ "$group" == "all" || "$group" == "muse" ]]; then
    download_dataset "muse-bench/MUSE-Books"
    download_model "muse-bench/MUSE-books_target" "muse-bench/MUSE-Books_target"
    download_model "muse-bench/MUSE-books_retrain" "muse-bench/MUSE-Books_retrain"
fi
