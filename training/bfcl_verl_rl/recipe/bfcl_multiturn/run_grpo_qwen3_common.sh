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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bfcl-multiturn-grpo}
PROJECT_NAME=${PROJECT_NAME:-bfcl_multiturn_rl}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/bfcl_multiturn_rl}
VAL_SPLIT=${VAL_SPLIT:-seen}  # seen | unseen
NGPU=${NGPU:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-40}
LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.2}
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}_${TIMESTAMP}.log}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[bfcl_multiturn] logging to $LOG_FILE"

TRAIN_FILES="['$DATA_DIR/train.parquet']"
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
  --config-name=bfcl_multiturn_grpo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.05 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$HOME/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME" \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs="$TOTAL_EPOCHS" "$@"
