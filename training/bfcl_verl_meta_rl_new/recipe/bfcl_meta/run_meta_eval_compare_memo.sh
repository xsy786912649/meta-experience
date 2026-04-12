#!/bin/bash
set -euo pipefail
set -x

PROJECT_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="$REPO_ROOT/recipe/bfcl_meta/config"
TOOL_CONFIG_PATH="$REPO_ROOT/recipe/bfcl_multiturn/config/tool_config/bfcl_tool_config.yaml"

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/bfcl_meta_rl}
NGPU=${NGPU:-4}
DATASETS=${DATASETS:-"train seen unseen"}  # train | seen | unseen
EXTRA_ARGS=("$@")
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR=${LOG_DIR:-$REPO_ROOT/logs}
RUN_ROOT=${RUN_ROOT:-$LOG_DIR/meta_eval_compare_${TIMESTAMP}}
mkdir -p "$LOG_DIR" "$RUN_ROOT"
LOG_FILE=${LOG_FILE:-$LOG_DIR/meta_eval_compare_${TIMESTAMP}.log}
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[bfcl_meta_compare] logging to $LOG_FILE"
echo "[bfcl_meta_compare] run_root=$RUN_ROOT"

TRAIN_FILE="$DATA_DIR/train.parquet"
TRAIN_FILES="['$TRAIN_FILE']"

resolve_val_file() {
  local dataset="$1"
  case "$dataset" in
    train)
      echo "$DATA_DIR/train.parquet"
      ;;
    seen)
      echo "$DATA_DIR/test_seen.parquet"
      ;;
    unseen)
      echo "$DATA_DIR/test_unseen.parquet"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac
}

run_eval_mode() {
  local dataset="$1"
  local mode="$2"
  local disable_query_memo="$3"
  local val_file
  val_file="$(resolve_val_file "$dataset")"
  local val_files="['$val_file']"
  local dump_dir="$RUN_ROOT/${dataset}_${mode}"

  mkdir -p "$dump_dir"

  python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name=bfcl_meta_grpo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$val_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.multi_turn.disable_query_memo="$disable_query_memo" \
    actor_rollout_ref.rollout.multi_turn.support_temperature=1.0 \
    actor_rollout_ref.rollout.multi_turn.query_temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    trainer.n_gpus_per_node="$NGPU" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="$dump_dir" \
    trainer.log_val_generations=0 \
    trainer.logger=['console'] "${EXTRA_ARGS[@]}"
}

for dataset in $DATASETS; do
  run_eval_mode "$dataset" "with_memo" "false"
  run_eval_mode "$dataset" "no_memo" "true"
done

python3 - <<'PY' "$RUN_ROOT" $DATASETS
import glob
import json
import os
import sys

run_root = sys.argv[1]
datasets = sys.argv[2:]


def summarize_dump(path: str) -> dict:
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        return {"total": 0, "query_ran": 0, "success": 0, "acc": 0.0, "neg_half": 0, "mean_score": 0.0}

    rows = []
    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    scores = [float(row.get("score", 0.0)) for row in rows]
    total = len(scores)
    if any("query_success" in row for row in rows):
        success = sum(bool(row.get("query_success")) for row in rows)
        query_ran = sum(bool(row.get("query_ran", True)) for row in rows)
    else:
        success = sum(abs(score - 1.0) < 1e-9 for score in scores)
        query_ran = total
    neg_half = sum(abs(score + 0.5) < 1e-9 for score in scores)
    mean_score = sum(scores) / total if total else 0.0
    acc = success / total if total else 0.0
    return {
        "total": total,
        "query_ran": query_ran,
        "success": success,
        "acc": acc,
        "neg_half": neg_half,
        "mean_score": mean_score,
    }


print("\n=== Meta Eval Compare Memo ===")
print("dataset\tmode\ttotal\tquery_ran\tsuccess\tacc\tneg_half\tmean_score")
for dataset in datasets:
    for mode in ("with_memo", "no_memo"):
        metrics = summarize_dump(os.path.join(run_root, f"{dataset}_{mode}"))
        print(
            f"{dataset}\t{mode}\t{metrics['total']}\t{metrics['query_ran']}\t{metrics['success']}\t"
            f"{metrics['acc']:.6f}\t{metrics['neg_half']}\t{metrics['mean_score']:.6f}"
        )

print("\n=== Accuracy Delta (with_memo - no_memo) ===")
for dataset in datasets:
    with_metrics = summarize_dump(os.path.join(run_root, f"{dataset}_with_memo"))
    no_metrics = summarize_dump(os.path.join(run_root, f"{dataset}_no_memo"))
    delta = with_metrics["acc"] - no_metrics["acc"]
    print(
        f"{dataset}\twith_memo={with_metrics['acc']:.6f}\t"
        f"no_memo={no_metrics['acc']:.6f}\tdelta={delta:+.6f}"
    )
PY
