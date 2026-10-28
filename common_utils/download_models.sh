#!/usr/bin/env bash
# Download public or access-authorized checkpoints into PROJECT_DATA_ROOT/models.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/server_env.sh"
if (( $# == 0 )); then
    echo "Usage: bash common_utils/download_models.sh ORGANIZATION/MODEL [...]" >&2
    exit 2
fi
for model_id in "$@"; do
    if [[ ! "$model_id" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ || "$model_id" == *..* ]]; then
        echo "Expected a Hugging Face organization/model ID." >&2
        exit 2
    fi
    huggingface-cli download "$model_id" --local-dir "${PROJECT_DATA_ROOT}/models/${model_id}"
done
