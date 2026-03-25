#!/bin/bash
set -x

PROJECT_DIR="$(pwd)"
export WANDB_API_KEY="5806b898ce5b350fa77a2975885a5d187bc2bf9e"
wandb login $WANDB_API_KEY
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$REPO_ROOT/recipe/bfcl_multiturn/config"
TOOL_CONFIG_PATH="$CONFIG_PATH/tool_config/bfcl_tool_config.yaml"

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/bfcl_multiturn_rl}
SPLIT=${SPLIT:-seen}  # seen | unseen
CKPT_PATH=${CKPT_PATH:?CKPT_PATH is required}
NGPU=${NGPU:-4}
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/eval_${SPLIT}_${TIMESTAMP}.log}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[bfcl_multiturn] logging to $LOG_FILE"

if [ "$SPLIT" = "unseen" ]; then
  VAL_FILE="$DATA_DIR/test_unseen.parquet"
else
  VAL_FILE="$DATA_DIR/test_seen.parquet"
fi

TRAIN_FILES="['$DATA_DIR/train.parquet']"
VAL_FILES="['$VAL_FILE']"

python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name=bfcl_multiturn_grpo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=disable \
  trainer.resume_from_path="$CKPT_PATH" \
  trainer.logger=['console'] "$@"
