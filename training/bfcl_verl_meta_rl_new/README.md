# BFCL Multi-turn RL Quick Start

路径约定：
- `<REPO_ROOT>` = `/Users/xusiyuan/Desktop/meta-experience/training/bfcl_verl_rl`

## 1) 预处理数据

```bash
cd <REPO_ROOT>
python3 -m recipe.bfcl_multiturn.preprocess_bfcl_multiturn_rl
```

默认参数：
- `local_dir=data/bfcl_multiturn_rl`
- `seed=42`
- `unseen_env_ratio=0.15`
- `seen_test_ratio=0.19`
- `train_size=1000`

输出：
- `data/bfcl_multiturn_rl/train.parquet`
- `data/bfcl_multiturn_rl/test_seen.parquet`
- `data/bfcl_multiturn_rl/test_unseen.parquet`
- `data/bfcl_multiturn_rl/split_summary.json`

## 2) 启动训练

### 直接跑预设模型脚本

```bash
cd <REPO_ROOT>
bash recipe/bfcl_multiturn/run_grpo_qwen3_0.6b.sh
# 或
bash recipe/bfcl_multiturn/run_grpo_qwen3_1.7b.sh
# 或
bash recipe/bfcl_multiturn/run_grpo_qwen3_4b.sh
# 或
bash recipe/bfcl_multiturn/run_grpo_qwen3_8b.sh
```

### 用通用脚本覆盖参数

```bash
cd <REPO_ROOT>

MODEL_PATH=Qwen/Qwen3-8B \
EXPERIMENT_NAME=qwen3-8b-bfcl-grpo \
DATA_DIR=<REPO_ROOT>/data/bfcl_multiturn_rl \
VAL_SPLIT=seen \
NGPU=8 \
TOTAL_EPOCHS=60 \
LR_WARMUP_STEPS_RATIO=0.2 \
bash recipe/bfcl_multiturn/run_grpo_qwen3_common.sh
```

当前通用脚本默认值：
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `trainer.test_freq=2`
- `NGPU=8`
- `TOTAL_EPOCHS=60`

## 3) 评测（seen / unseen）

```bash
cd <REPO_ROOT>

bash recipe/bfcl_multiturn/run_eval_split.sh \
  MODEL_PATH=Qwen/Qwen3-8B \
  SPLIT=seen \
  CKPT_PATH=/path/to/checkpoint

bash recipe/bfcl_multiturn/run_eval_split.sh \
  MODEL_PATH=Qwen/Qwen3-8B \
  SPLIT=unseen \
  CKPT_PATH=/path/to/checkpoint
```

## 4) 日志与保存路径

- 训练日志：`<REPO_ROOT>/logs/`
- checkpoint：`<REPO_ROOT>/checkpoints//`
