# run on 8xH20
# make sure your current working directory is the root of the project

# [TODO] we might do `dump_rollout_generations` for debugging

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
export WANDB_API_KEY="5806b898ce5b350fa77a2975885a5d187bc2bf9e"
wandb login $WANDB_API_KEY

export RAY_TMPDIR=/ebs-basemodeling/siyuanxu/ray_tmp
mkdir -p "$RAY_TMPDIR"

train_dir=/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/verl_toolmock/data
train_files="['$train_dir/train.parquet']"
val_files="['$train_dir/test.parquet']"
train_batch_size=256
ppo_mini_batch_size=$((train_batch_size / 2))
timestamp=$(date "+%Y%m%d_%H%M%S")

project_name="toolmock_async_rl"
# hyperparamters:
max_assistant_turns=20
experiment_name="qwen3-8b-toolmock-grpo_rollout8_${timestamp}"

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#NGPU=8

export CUDA_VISIBLE_DEVICES=4,5,6,7
NGPU=4
#export RAY_TMPDIR="/ebs-basemodeling/siyuanxu/ray_tmp"

export VLLM_USE_V1=1
ray stop --force

echo "📦 Merge and save local cache"
CACHE_FILE="/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/cache/tool_cache.json"
S3_DIR="s3://shopqa-users/siyuanxu/tool_mock"
S3_FILE="${S3_DIR}/cache/tool_cache.json"
CHECKPOINT_DIR="/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/verl_output/checkpoints/${project_name}/${experiment_name}"

#python verl/utils/s3_utils/merge_and_save_local_cache.py \
#    --cache_file "$CACHE_FILE" \
#    --s3_uri "$S3_FILE"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=128 \
    data.max_prompt_length=16384 \
    data.max_response_length=4096 \
    data.truncation='error' \
    data.custom_cls.path=pkg://verl/utils/dataset/tool_research_dataset \
    data.custom_cls.name=ToolResearchRLDataset \
    actor_rollout_ref.model.path=/ebs-basemodeling/siyuanxu/model_inference/model/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.prompt_length=32768 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.completion_callback=tests.workers.rollout.my_test_tool_completion_callback.Qwen3CustomToolCompletionCallback \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files="$train_files" \
    data.val_files="$val_files"  \
    trainer.total_epochs=1 $@

# upload the local to s3 command 
#echo "📦 Uploading checkpoints and cache to S3..."
#aws s3 cp --recursive "$CHECKPOINT_DIR" "${S3_DIR}/verl_output/checkpoints/${project_name}/${experiment_name}"

#python verl/utils/s3_utils/merge_and_upload_cache.py \
#    --cache_file "$CACHE_FILE" \
#    --s3_uri "$S3_FILE"

