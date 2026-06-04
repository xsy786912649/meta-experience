#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"
export AGENT_API_BASE="http://127.0.0.1:8001/v1"
export AGENT_API_KEY="EMPTY"
unset OPENAI_API_KEY
unset OPENAI_API_BASE
unset AGENT_LLM_ARGS
unset SUMMARY_LLM_ARGS

DOMAIN_LIST="${DOMAIN_LIST:-retail airline telecom}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_ROOT"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$LOG_ROOT/tau2-bench-qwen3-4b-${timestamp}.log"

echo "[tau2-bench-qwen3-4b] log_file=$log_file"

{
  echo "[tau2-bench-qwen3-4b] started_at=$(date)"
  echo "[tau2-bench-qwen3-4b] cwd=$PWD"
  echo "[tau2-bench-qwen3-4b] domains=$DOMAIN_LIST"
  echo "[tau2-bench-qwen3-4b] AGENT_API_BASE=${AGENT_API_BASE:-}"
  echo "[tau2-bench-qwen3-4b] AGENT_API_KEY=${AGENT_API_KEY:+<set>}"
  echo "[tau2-bench-qwen3-4b] OPENROUTER_API_KEY=${OPENROUTER_API_KEY:+<set>}"

  for domain in $DOMAIN_LIST; do
    echo "[tau2-bench-qwen3-4b] running domain=$domain"
    AGENT_LLM="openai/qwen3-4b" \
      SUMMARY_LLM="openai/qwen3-4b" \
      DOMAIN="$domain" \
      TASK_SPLIT_NAME="${TASK_SPLIT_NAME:-base}" \
      NUM_TRIALS=1 \
      MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}" \
      ADAPTATION_COUNTS="${ADAPTATION_COUNTS:-0 1 2}" \
      ./run_tau2_bench_env_adaptation.sh
    echo "[tau2-bench-qwen3-4b] finished domain=$domain"
  done

  echo "[tau2-bench-qwen3-4b] finished_at=$(date)"
} 2>&1 | tee "$log_file"
