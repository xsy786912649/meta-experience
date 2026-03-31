import argparse
import json
import os
from collections import defaultdict

import datasets


def _load_rows(path: str) -> list[dict]:
    return datasets.load_dataset("parquet", data_files=path)["train"].to_list()


def _decode_payload(row: dict) -> dict:
    payload = row["total_messages"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _build_pair_row(split_name: str, support_row: dict, query_row: dict) -> dict:
    support_payload = _decode_payload(support_row)
    query_payload = _decode_payload(query_row)
    pair_payload = {
        "pair_split": split_name,
        "env_key": query_row["env_key"],
        "support": support_payload,
        "query": query_payload,
    }
    pair_id = f"{split_name}:{support_row['id']}->{query_row['id']}"
    return {
        "data_source": "bfcl_meta_summary",
        "pair_id": pair_id,
        "env_key": query_row["env_key"],
        "support_id": support_row["id"],
        "query_id": query_row["id"],
        "flag": pair_id,
        "reward_model": None,
        "tools_kwargs": "",
        "messages": "",
        "total_messages": json.dumps(pair_payload, ensure_ascii=False),
    }


def _group_by_env(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["env_key"]].append(row)
    return grouped


def _build_train_pairs(train_rows: list[dict]) -> list[dict]:
    grouped = _group_by_env(train_rows)
    pairs = []
    for env_key, rows in grouped.items():
        for support_row in rows:
            for query_row in rows:
                if support_row["id"] == query_row["id"]:
                    continue
                pairs.append(_build_pair_row("train", support_row, query_row))
    return pairs


def _build_seen_eval_pairs(train_rows: list[dict], test_seen_rows: list[dict]) -> list[dict]:
    train_grouped = _group_by_env(train_rows)
    query_grouped = _group_by_env(test_seen_rows)
    pairs = []
    for env_key, query_rows in query_grouped.items():
        support_rows = train_grouped.get(env_key, [])
        for query_row in query_rows:
            for support_row in support_rows:
                if support_row["id"] == query_row["id"]:
                    continue
                pairs.append(_build_pair_row("test_seen", support_row, query_row))
    return pairs


def _build_unseen_eval_pairs(test_unseen_rows: list[dict]) -> list[dict]:
    grouped = _group_by_env(test_unseen_rows)
    pairs = []
    for env_key, rows in grouped.items():
        for support_row in rows:
            for query_row in rows:
                if support_row["id"] == query_row["id"]:
                    continue
                pairs.append(_build_pair_row("test_unseen", support_row, query_row))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", default="data/bfcl_multiturn_rl")
    parser.add_argument("--output_dir", default="data/bfcl_meta_rl")
    args = parser.parse_args()

    source_dir = os.path.abspath(os.path.expanduser(args.source_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    train_rows = _load_rows(os.path.join(source_dir, "train.parquet"))
    test_seen_rows = _load_rows(os.path.join(source_dir, "test_seen.parquet"))
    test_unseen_rows = _load_rows(os.path.join(source_dir, "test_unseen.parquet"))

    train_pairs = _build_train_pairs(train_rows)
    test_seen_pairs = _build_seen_eval_pairs(train_rows, test_seen_rows)
    test_unseen_pairs = _build_unseen_eval_pairs(test_unseen_rows)

    datasets.Dataset.from_list(train_pairs).to_parquet(os.path.join(output_dir, "train.parquet"))
    datasets.Dataset.from_list(test_seen_pairs).to_parquet(os.path.join(output_dir, "test_seen.parquet"))
    datasets.Dataset.from_list(test_unseen_pairs).to_parquet(os.path.join(output_dir, "test_unseen.parquet"))

    summary = {
        "source_dir": source_dir,
        "num_source_train_tasks": len(train_rows),
        "num_source_test_seen_tasks": len(test_seen_rows),
        "num_source_test_unseen_tasks": len(test_unseen_rows),
        "num_train_pairs": len(train_pairs),
        "num_test_seen_pairs": len(test_seen_pairs),
        "num_test_unseen_pairs": len(test_unseen_pairs),
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
