#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/tau-bench"
cd "$BENCH_DIR"

ENV_NAME="${ENV_NAME:-retail}"
TASK_SPLIT="${TASK_SPLIT:-test}"
MODEL="${MODEL:?MODEL is required, e.g. MODEL=gpt-4o}"
MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"
USER_MODEL="${USER_MODEL:-$MODEL}"
USER_MODEL_PROVIDER="${USER_MODEL_PROVIDER:-$MODEL_PROVIDER}"
SUMMARY_MODEL="${SUMMARY_MODEL:-$MODEL}"
SUMMARY_MODEL_PROVIDER="${SUMMARY_MODEL_PROVIDER:-$MODEL_PROVIDER}"
AGENT_STRATEGY="${AGENT_STRATEGY:-tool-calling}"
USER_STRATEGY="${USER_STRATEGY:-llm}"
ADAPTATION_COUNTS="${ADAPTATION_COUNTS:-0 1 2}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
MAX_SUPPORT_STEPS="${MAX_SUPPORT_STEPS:-30}"
MAX_QUERY_STEPS="${MAX_QUERY_STEPS:-30}"
SEED="${SEED:-10}"
LOG_DIR="${LOG_DIR:-results_env_adaptation}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MEMO_TEMPERATURE="${MEMO_TEMPERATURE:-0.0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:--1}"
SHUFFLE="${SHUFFLE:-0}"

cmd=(
  python run_env_adaptation.py
  --env "$ENV_NAME"
  --task-split "$TASK_SPLIT"
  --model "$MODEL"
  --model-provider "$MODEL_PROVIDER"
  --user-model "$USER_MODEL"
  --user-model-provider "$USER_MODEL_PROVIDER"
  --summary-model "$SUMMARY_MODEL"
  --summary-model-provider "$SUMMARY_MODEL_PROVIDER"
  --agent-strategy "$AGENT_STRATEGY"
  --user-strategy "$USER_STRATEGY"
  --num-trials "$NUM_TRIALS"
  --max-concurrency "$MAX_CONCURRENCY"
  --max-support-steps "$MAX_SUPPORT_STEPS"
  --max-query-steps "$MAX_QUERY_STEPS"
  --seed "$SEED"
  --log-dir "$LOG_DIR"
  --temperature "$TEMPERATURE"
  --memo-temperature "$MEMO_TEMPERATURE"
  --start-index "$START_INDEX"
  --end-index "$END_INDEX"
  --shuffle "$SHUFFLE"
  --adaptation-counts
)

for count in $ADAPTATION_COUNTS; do
  cmd+=("$count")
done

if [ -n "${TASK_IDS:-}" ]; then
  cmd+=(--task-ids)
  for task_id in $TASK_IDS; do
    cmd+=("$task_id")
  done
fi

if [ -n "${FEW_SHOT_DISPLAYS_PATH:-}" ]; then
  cmd+=(--few-shot-displays-path "$FEW_SHOT_DISPLAYS_PATH")
fi

cmd+=("$@")

echo "[tau-bench] ${cmd[*]}"
"${cmd[@]}"
