#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
export AGENT_API_BASE="http://127.0.0.1:8000/v1"
export AGENT_API_KEY="EMPTY"
export USER_SIM_MAX_RETRIES="${USER_SIM_MAX_RETRIES:-10}"
export USER_SIM_RETRY_BACKOFF="${USER_SIM_RETRY_BACKOFF:-2}"
unset OPENAI_API_KEY
unset OPENAI_API_BASE
unset AGENT_COMPLETION_KWARGS
unset SUMMARY_COMPLETION_KWARGS

LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_ROOT/tau-bench-qwen3-8b-retail-airline-${timestamp}.log"

echo "[tau-bench-qwen3-8b] log_file=$log_file"

{
  echo "[tau-bench-qwen3-8b] started_at=$(date)"
  echo "[tau-bench-qwen3-8b] cwd=$PWD"
  echo "[tau-bench-qwen3-8b] AGENT_API_BASE=${AGENT_API_BASE:-}"
  echo "[tau-bench-qwen3-8b] AGENT_API_KEY=${AGENT_API_KEY:+<set>}"
  echo "[tau-bench-qwen3-8b] OPENROUTER_API_KEY=${OPENROUTER_API_KEY:+<set>}"

  for env in retail airline; do
    echo "[tau-bench-qwen3-8b] running env=$env"
    MODEL="openai/qwen3-8b" \
      MODEL_PROVIDER=openai \
      SUMMARY_MODEL="openai/qwen3-8b" \
      SUMMARY_MODEL_PROVIDER=openai \
      ENV_NAME="$env" \
      TASK_SPLIT=test \
      NUM_TRIALS=1 \
      MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}" \
      ADAPTATION_COUNTS="${ADAPTATION_COUNTS:-0 1 2}" \
      ./run_tau_bench_env_adaptation.sh
    echo "[tau-bench-qwen3-8b] finished env=$env"
  done

  echo "[tau-bench-qwen3-8b] finished_at=$(date)"
} 2>&1 | tee "$log_file"
