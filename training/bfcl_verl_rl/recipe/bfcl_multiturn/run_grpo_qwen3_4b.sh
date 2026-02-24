#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

timestamp=$(date "+%Y%m%d_%H%M%S")
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-4b-bfcl-grpo-${timestamp}}

MODEL_PATH="$MODEL_PATH" EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  bash "$SCRIPT_DIR/run_grpo_qwen3_common.sh" "$@"

