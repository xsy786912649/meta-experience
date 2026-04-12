import argparse
import json
import os
import random
from collections import defaultdict

import datasets


def _load_rows(path: str) -> list[dict]:
    return datasets.load_dataset("parquet", data_files=path)["train"].to_list()


def _decode_payload(row: dict) -> dict:
    payload = row["total_messages"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _experience_key(row: dict) -> str:
    if row.get("experience_key"):
        return row["experience_key"]
    payload = _decode_payload(row)
    category = row.get("category") or payload.get("category", "")
    return f"{row['env_key']}::{category}"


def _build_pair_row(split_name: str, support_row: dict, query_row: dict) -> dict:
    support_payload = _decode_payload(support_row)
    query_payload = _decode_payload(query_row)
    experience_key = _experience_key(query_row)
    pair_payload = {
        "pair_split": split_name,
        "env_key": query_row["env_key"],
        "category": query_row.get("category") or query_payload.get("category"),
        "experience_key": experience_key,
        "support": support_payload,
        "query": query_payload,
    }
    pair_id = f"{split_name}:{support_row['id']}->{query_row['id']}"
    return {
        "data_source": "bfcl_meta_summary",
        "pair_id": pair_id,
        "env_key": query_row["env_key"],
        "category": query_row.get("category") or query_payload.get("category"),
        "experience_key": experience_key,
        "support_id": support_row["id"],
        "query_id": query_row["id"],
        "flag": pair_id,
        "reward_model": None,
        "tools_kwargs": "",
        "messages": "",
        "total_messages": json.dumps(pair_payload, ensure_ascii=False),
    }


def _group_by_experience(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_experience_key(row)].append(row)
    return grouped


def _build_train_pairs(train_rows: list[dict]) -> list[dict]:
    grouped = _group_by_experience(train_rows)
    pairs = []
    for experience_key, rows in grouped.items():
        for support_row in rows:
            for query_row in rows:
                pairs.append(_build_pair_row("train", support_row, query_row))
    return pairs


def _split_seen_pairs(all_seen_pairs: list[dict], test_seen_pair_count: int, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = all_seen_pairs[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_test_seen = min(max(0, test_seen_pair_count), len(shuffled))
    test_seen_pairs = shuffled[:n_test_seen]
    train_pairs = shuffled[n_test_seen:]
    for pair in test_seen_pairs:
        pair["pair_id"] = pair["pair_id"].replace("train:", "test_seen:", 1)
        pair["flag"] = pair["pair_id"]
        payload = json.loads(pair["total_messages"])
        payload["pair_split"] = "test_seen"
        pair["total_messages"] = json.dumps(payload, ensure_ascii=False)
    return train_pairs, test_seen_pairs


def _build_unseen_eval_pairs(test_unseen_rows: list[dict]) -> list[dict]:
    grouped = _group_by_experience(test_unseen_rows)
    pairs = []
    for experience_key, rows in grouped.items():
        for support_row in rows:
            for query_row in rows:
                pairs.append(_build_pair_row("test_unseen", support_row, query_row))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/bfcl_multiturn_rl")
    parser.add_argument("--output_dir", default="data/bfcl_meta_rl")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test_seen_pair_count", type=int, default=256)
    args = parser.parse_args()

    source_dir = os.path.abspath(os.path.expanduser(args.source_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    train_rows = _load_rows(os.path.join(source_dir, "train.parquet"))
    test_unseen_rows = _load_rows(os.path.join(source_dir, "test_unseen.parquet"))

    all_seen_pairs = _build_train_pairs(train_rows)
    train_pairs, test_seen_pairs = _split_seen_pairs(
        all_seen_pairs,
        test_seen_pair_count=args.test_seen_pair_count,
        seed=args.seed,
    )
    test_unseen_pairs = _build_unseen_eval_pairs(test_unseen_rows)

    datasets.Dataset.from_list(train_pairs).to_parquet(os.path.join(output_dir, "train.parquet"))
    datasets.Dataset.from_list(test_seen_pairs).to_parquet(os.path.join(output_dir, "test_seen.parquet"))
    datasets.Dataset.from_list(test_unseen_pairs).to_parquet(os.path.join(output_dir, "test_unseen.parquet"))

    summary = {
        "source_dir": source_dir,
        "num_source_train_tasks": len(train_rows),
        "num_source_test_seen_tasks": 0,
        "num_source_test_unseen_tasks": len(test_unseen_rows),
        "num_source_train_experience_keys": len({_experience_key(row) for row in train_rows}),
        "num_source_test_seen_experience_keys": 0,
        "num_source_test_unseen_experience_keys": len({_experience_key(row) for row in test_unseen_rows}),
        "test_seen_pair_count_target": args.test_seen_pair_count,
        "num_train_pairs": len(train_pairs),
        "num_test_seen_pairs": len(test_seen_pairs),
        "num_test_unseen_pairs": len(test_unseen_pairs),
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
