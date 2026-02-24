# BFCL Multi-turn RL

## 1) Prepare data split (seen-env + unseen-env)

```bash
cd /Users/xusiyuan/Desktop/meta-experience/training/bfcl_verl_rl
python3 -m recipe.bfcl_multiturn.preprocess_bfcl_multiturn_rl \
  --local_dir data/bfcl_multiturn_rl \
  --unseen_env_ratio 0.25 \
  --seen_test_ratio 0.2
```

Outputs:
- `data/bfcl_multiturn_rl/train.parquet`
- `data/bfcl_multiturn_rl/test_seen.parquet`
- `data/bfcl_multiturn_rl/test_unseen.parquet`
- `data/bfcl_multiturn_rl/split_summary.json`

## 2) Train (Qwen3 series)

```bash
bash recipe/bfcl_multiturn/run_grpo_qwen3_8b.sh
bash recipe/bfcl_multiturn/run_grpo_qwen3_4b.sh
bash recipe/bfcl_multiturn/run_grpo_qwen3_1.7b.sh
bash recipe/bfcl_multiturn/run_grpo_qwen3_0.6b.sh
```

Common overrides:
- `DATA_DIR=data/bfcl_multiturn_rl`
- `VAL_SPLIT=seen` or `VAL_SPLIT=unseen`
- `NGPU=4`

## 3) Evaluate seen/unseen split

```bash
bash recipe/bfcl_multiturn/run_eval_split.sh \
  MODEL_PATH=Qwen/Qwen3-8B \
  SPLIT=seen \
  CKPT_PATH=/path/to/checkpoint

bash recipe/bfcl_multiturn/run_eval_split.sh \
  MODEL_PATH=Qwen/Qwen3-8B \
  SPLIT=unseen \
  CKPT_PATH=/path/to/checkpoint
```
