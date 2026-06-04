#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/tau-bench"
cd "$BENCH_DIR"

ENV_NAME="${ENV_NAME:-retail}"
TASK_SPLIT="${TASK_SPLIT:-test}"
MODEL="${MODEL:?MODEL is required, e.g. MODEL=gpt-4o}"
MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"
USER_MODEL="${USER_MODEL:-deepseek/deepseek-v4-flash}"
USER_MODEL_PROVIDER="${USER_MODEL_PROVIDER:-openrouter}"
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
AGENT_COMPLETION_KWARGS="${AGENT_COMPLETION_KWARGS:-}"
SUMMARY_COMPLETION_KWARGS="${SUMMARY_COMPLETION_KWARGS:-}"

if [ "$USER_MODEL_PROVIDER" = "openrouter" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is required when USER_MODEL_PROVIDER=openrouter" >&2
  exit 1
fi

if [ -z "$AGENT_COMPLETION_KWARGS" ] && { [ -n "${AGENT_API_BASE:-}" ] || [ -n "${AGENT_API_KEY:-}" ]; }; then
  AGENT_COMPLETION_KWARGS="$(python3 -c 'import json, os; d={}; 
api_base=os.environ.get("AGENT_API_BASE"); api_key=os.environ.get("AGENT_API_KEY");
if api_base: d["api_base"]=api_base
if api_key: d["api_key"]=api_key
print(json.dumps(d))')"
fi

if [ -z "$SUMMARY_COMPLETION_KWARGS" ]; then
  if [ -n "${SUMMARY_API_BASE:-}" ] || [ -n "${SUMMARY_API_KEY:-}" ]; then
    SUMMARY_COMPLETION_KWARGS="$(python3 -c 'import json, os; d={};
api_base=os.environ.get("SUMMARY_API_BASE"); api_key=os.environ.get("SUMMARY_API_KEY");
if api_base: d["api_base"]=api_base
if api_key: d["api_key"]=api_key
print(json.dumps(d))')"
  else
    SUMMARY_COMPLETION_KWARGS="$AGENT_COMPLETION_KWARGS"
  fi
fi

python3 -c 'import json, os, sys
for name in ("AGENT_COMPLETION_KWARGS", "SUMMARY_COMPLETION_KWARGS"):
    value = os.environ.get(name) or "{}"
    try:
        json.loads(value)
    except Exception as exc:
        print(f"{name} must be valid JSON: {value}\n{exc}", file=sys.stderr)
        sys.exit(1)
'

cmd=(
  python3 run_env_adaptation.py
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
  --agent-completion-kwargs "${AGENT_COMPLETION_KWARGS:-{}}"
  --summary-completion-kwargs "${SUMMARY_COMPLETION_KWARGS:-{}}"
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
