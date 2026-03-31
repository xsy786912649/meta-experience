# BFCL Meta RL

This recipe trains a meta-summarizer with two alternating pipelines.

Each training sample is a same-environment pair `(support_task, query_task)`:

1. Run the current policy on the support task and keep the support conversation.
2. Append a summarize request to that support conversation.
3. Generate an environment experience memo.
4. Inject the memo into the query task system prompt.
5. Run the current policy on the query task and use query success as reward.

The training alternates by batch between:

- shared-support-summary: one support trajectory is shared across `rollout.n` summaries, and loss only covers summary tokens
- full-support-summary: `rollout.n` independent support trajectories are sampled, each followed by one summary, and loss covers both support assistant actions and summary tokens

For `rollout.n = 8`, the query task is shared within a GRPO group.

## Data Split Semantics

Source BFCL multiturn data is split by `env_key`, not by task:

- `train.parquet`: seen environments used for meta-train
- `test_seen.parquet`: held-out tasks from seen environments
- `test_unseen.parquet`: tasks from environments absent during meta-train

Meta pair construction:

- train pairs: support from `train`, query from `train`, same `env_key`, different task ids
- seen eval pairs: support from `train`, query from `test_seen`, same `env_key`
- unseen eval pairs: support from `test_unseen`, query from `test_unseen`, same `env_key`, different task ids

Support trajectories from `test_unseen` are allowed at evaluation time because they are test-time adaptation context, not gradient updates.

## Entry Points

### 1. Prepare Meta Pair Data

Build pair-level parquet files from the original BFCL multiturn splits:

```bash
cd /path/to/bfcl_verl_meta_rl
python3 recipe/bfcl_meta/preprocess_bfcl_meta_pairs.py \
  --source_dir data/bfcl_multiturn_rl \
  --output_dir data/bfcl_meta_rl
```

### 2. Single-Sample Debug

Runs one support-summary-query chain and prints:

- support trajectory
- summary full output
- summary context text
- query success

```bash
cd /path/to/bfcl_verl_meta_rl
python3 recipe/bfcl_meta/debug_meta_single.py \
  --repo-root . \
  --data-file data/bfcl_meta_rl/test_seen.parquet \
  --index 0 \
  --model-path /path/to/model \
  --served-model your-served-model-name
```

### 3. Train

Model-specific wrappers are provided for:

- `recipe/bfcl_meta/run_grpo_qwen3_0.6b.sh`
- `recipe/bfcl_meta/run_grpo_qwen3_1.7b.sh`
- `recipe/bfcl_meta/run_grpo_qwen3_4b.sh`
- `recipe/bfcl_meta/run_grpo_qwen3_8b.sh`

```bash
cd /path/to/bfcl_verl_meta_rl
bash recipe/bfcl_meta/run_grpo_qwen3_8b.sh
```

Optional:

- `MODEL_PATH=/your/local/or/hf/model/path`
- `DATA_DIR=/.../data/bfcl_meta_rl`
- `VAL_SPLIT=seen` or `VAL_SPLIT=unseen`
- `NGPU=8`

### 4. Test

```bash
cd /path/to/bfcl_verl_meta_rl
MODEL_PATH=/path/to/model \
CKPT_PATH=/path/to/checkpoint \
SPLIT=seen \
bash recipe/bfcl_meta/run_meta_eval.sh
```

Use `SPLIT=unseen` to evaluate unseen environments.

### 5. Merge Checkpoint To HF Model

After training, the saved actor checkpoint is still in FSDP shard format. To convert it into a Hugging Face model directory:

```bash
cd /path/to/bfcl_verl_meta_rl
PYTHONPATH=$(pwd) python scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir checkpoints/bfcl_meta_rl/<experiment_name>/global_step_<step>/actor \
  --target_dir checkpoints/bfcl_meta_rl/<experiment_name>/global_step_<step>/model \
  --hf_model_path Qwen/Qwen3-8B
```

Notes:

- `--local_dir` should point to the saved actor checkpoint directory.
- `--target_dir` is the output Hugging Face model directory after merge.
- `--hf_model_path` should match the base model used for training, for example `Qwen/Qwen3-8B`.
