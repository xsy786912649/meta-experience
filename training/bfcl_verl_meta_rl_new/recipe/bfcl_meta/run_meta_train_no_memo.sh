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
PROJECT_NAME=${PROJECT_NAME:-bfcl_meta_rl}
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
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.05 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.multi_turn.disable_query_memo=true \
  trainer.rollout_only=true \
  trainer.val_before_train=false \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$REPO_ROOT/checkpoints/$PROJECT_NAME/${EXPERIMENT_NAME}" \
  trainer.save_freq=1000000 \
  trainer.test_freq=0 \
  trainer.total_epochs="$TOTAL_EPOCHS" "$@"
