#!/bin/bash
set -x

PROJECT_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
CONFIG_PATH="$REPO_ROOT/recipe/bfcl_meta/config"
TOOL_CONFIG_PATH="$REPO_ROOT/recipe/bfcl_multiturn/config/tool_config/bfcl_tool_config.yaml"

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/bfcl_meta_rl}
VAL_SPLIT=${VAL_SPLIT:-seen}  # seen | unseen
NGPU=${NGPU:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bfcl-meta-train-no-memo}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}_${TIMESTAMP}.log}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[bfcl_meta_no_memo] logging to $LOG_FILE"

TRAIN_FILE="$DATA_DIR/train.parquet"
TRAIN_FILES="['$TRAIN_FILE']"
if [ "$VAL_SPLIT" = "unseen" ]; then
  VAL_FILE="$DATA_DIR/test_unseen.parquet"
else
  VAL_FILE="$DATA_DIR/test_seen.parquet"
fi
VAL_FILES="['$VAL_FILE']"

ulimit -n 65535
export VLLM_USE_V1=1
ray stop --force

python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=bfcl_meta_grpo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.multi_turn.disable_query_memo=true \
  trainer.rollout_only=true \
  trainer.val_before_train=false \
  trainer.logger=['console'] \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$REPO_ROOT/checkpoints/bfcl_meta_rl/${EXPERIMENT_NAME}" \
  trainer.save_freq=1000000 \
  trainer.test_freq=0 \
  trainer.total_epochs="$TOTAL_EPOCHS" "$@"
