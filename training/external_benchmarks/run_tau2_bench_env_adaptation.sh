#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/tau2-bench-official"
cd "$BENCH_DIR"

DOMAIN="${DOMAIN:-airline}"
AGENT_LLM="${AGENT_LLM:?AGENT_LLM is required, e.g. AGENT_LLM=gpt-4.1}"
USER_LLM="${USER_LLM:-openrouter/deepseek/deepseek-v4-flash}"
SUMMARY_LLM="${SUMMARY_LLM:-$AGENT_LLM}"
AGENT="${AGENT:-llm_agent}"
USER_SIMULATOR="${USER_SIMULATOR:-user_simulator}"
TASK_SPLIT_NAME="${TASK_SPLIT_NAME:-base}"
ADAPTATION_COUNTS="${ADAPTATION_COUNTS:-0 1 2}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_ERRORS="${MAX_ERRORS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
SEED="${SEED:-300}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_DIR="${LOG_DIR:-data/env_adaptation}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-1024}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-1024}"
export AGENT_MAX_TOKENS
export SUMMARY_MAX_TOKENS
AGENT_LLM_ARGS_PROVIDED="${AGENT_LLM_ARGS+x}"
SUMMARY_LLM_ARGS_PROVIDED="${SUMMARY_LLM_ARGS+x}"
AGENT_LLM_ARGS="${AGENT_LLM_ARGS:-{\"temperature\":0}}"
USER_LLM_ARGS="${USER_LLM_ARGS:-{\"temperature\":0}}"
SUMMARY_LLM_ARGS="${SUMMARY_LLM_ARGS:-{\"temperature\":0}}"

if [[ "$USER_LLM" == openrouter/* ]] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is required when USER_LLM uses OpenRouter" >&2
  exit 1
fi

if [ -n "${AGENT_API_BASE:-}" ] || [ -n "${AGENT_API_KEY:-}" ]; then
  AGENT_LLM_ARGS="$(python3 -c 'import json, os; d={"temperature":0};
api_base=os.environ.get("AGENT_API_BASE"); api_key=os.environ.get("AGENT_API_KEY");
max_tokens=os.environ.get("AGENT_MAX_TOKENS");
if api_base: d["api_base"]=api_base
if api_key: d["api_key"]=api_key
if max_tokens: d["max_tokens"]=int(max_tokens)
print(json.dumps(d))')"
fi

if [ -n "${SUMMARY_API_BASE:-}" ] || [ -n "${SUMMARY_API_KEY:-}" ] || [ -n "${SUMMARY_MAX_TOKENS:-}" ]; then
    SUMMARY_LLM_ARGS="$(python3 -c 'import json, os; d={"temperature":0};
api_base=os.environ.get("SUMMARY_API_BASE") or os.environ.get("AGENT_API_BASE");
api_key=os.environ.get("SUMMARY_API_KEY") or os.environ.get("AGENT_API_KEY");
max_tokens=os.environ.get("SUMMARY_MAX_TOKENS");
if api_base: d["api_base"]=api_base
if api_key: d["api_key"]=api_key
if max_tokens: d["max_tokens"]=int(max_tokens)
print(json.dumps(d))')"
elif [ -z "$SUMMARY_LLM_ARGS_PROVIDED" ] && { [ -n "${AGENT_API_BASE:-}" ] || [ -n "${AGENT_API_KEY:-}" ]; }; then
  SUMMARY_LLM_ARGS="$AGENT_LLM_ARGS"
fi

python3 -c 'import json, sys
for name, value in {
    "AGENT_LLM_ARGS": sys.argv[1],
    "USER_LLM_ARGS": sys.argv[2],
    "SUMMARY_LLM_ARGS": sys.argv[3],
}.items():
    value = value or "{}"
    try:
        json.loads(value)
    except Exception as exc:
        print(f"{name} must be valid JSON: {value}\n{exc}", file=sys.stderr)
        sys.exit(1)
' "$AGENT_LLM_ARGS" "$USER_LLM_ARGS" "$SUMMARY_LLM_ARGS"

cmd=(
  python3 run_env_adaptation.py
  --domain "$DOMAIN"
  --agent "$AGENT"
  --agent-llm "$AGENT_LLM"
  --agent-llm-args "$AGENT_LLM_ARGS"
  --user "$USER_SIMULATOR"
  --user-llm "$USER_LLM"
  --user-llm-args "$USER_LLM_ARGS"
  --summary-llm "$SUMMARY_LLM"
  --summary-llm-args "$SUMMARY_LLM_ARGS"
  --task-split-name "$TASK_SPLIT_NAME"
  --num-trials "$NUM_TRIALS"
  --max-steps "$MAX_STEPS"
  --max-errors "$MAX_ERRORS"
  --max-concurrency "$MAX_CONCURRENCY"
  --seed "$SEED"
  --log-level "$LOG_LEVEL"
  --log-dir "$LOG_DIR"
  --adaptation-counts
)

for count in $ADAPTATION_COUNTS; do
  cmd+=("$count")
done

if [ -n "$NUM_TASKS" ]; then
  cmd+=(--num-tasks "$NUM_TASKS")
fi

if [ -n "${TASK_SET_NAME:-}" ]; then
  cmd+=(--task-set-name "$TASK_SET_NAME")
fi

if [ -n "${TASK_IDS:-}" ]; then
  cmd+=(--task-ids)
  for task_id in $TASK_IDS; do
    cmd+=("$task_id")
  done
fi

if [ -n "${TIMEOUT:-}" ]; then
  cmd+=(--timeout "$TIMEOUT")
fi

if [ "${ENFORCE_COMMUNICATION_PROTOCOL:-0}" = "1" ]; then
  cmd+=(--enforce-communication-protocol)
fi

if [ -n "${RETRIEVAL_CONFIG:-}" ]; then
  cmd+=(--retrieval-config "$RETRIEVAL_CONFIG")
fi

if [ -n "${RETRIEVAL_CONFIG_KWARGS:-}" ]; then
  cmd+=(--retrieval-config-kwargs "$RETRIEVAL_CONFIG_KWARGS")
fi

cmd+=("$@")

echo "[tau2-bench] ${cmd[*]}"
"${cmd[@]}"
