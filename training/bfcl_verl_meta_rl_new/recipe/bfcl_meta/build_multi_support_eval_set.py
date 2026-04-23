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
    payload = _decode_payload(row)
    return row.get("experience_key") or payload.get("experience_key", "")


def _support_dedup_key(row: dict) -> str:
    payload = _decode_payload(row)
    support_payload = payload["support"]
    support_id = support_payload.get("id")
    if support_id:
        return f"id:{support_id}"
    return "json:" + json.dumps(support_payload, ensure_ascii=False, sort_keys=True)


def build_multi_support_rows(rows: list[dict], split_name: str, support_count: int, seed: int) -> list[dict]:
    support_pool_by_env = defaultdict(dict)
    for row in rows:
        support_pool_by_env[row["env_key"]].setdefault(_support_dedup_key(row), row)

    rng = random.Random(seed)
    output_rows = []
    num_rows_total = 0
    num_rows_kept = 0
    num_rows_skipped_insufficient_supports = 0
    num_unique_supports_total = sum(len(pool) for pool in support_pool_by_env.values())
    for row in rows:
        num_rows_total += 1
        env_key = row["env_key"]
        payload = _decode_payload(row)
        original_support_key = _support_dedup_key(row)
        support_pool_rows = list(support_pool_by_env[env_key].values())
        extra_support_rows = [
            support_row for support_row in support_pool_rows if _support_dedup_key(support_row) != original_support_key
        ]
        rng.shuffle(extra_support_rows)
        selected_support_rows = [row] + extra_support_rows[: max(0, support_count - 1)]
        if len(selected_support_rows) < support_count:
            num_rows_skipped_insufficient_supports += 1
            continue

        num_rows_kept += 1
        supports = [_decode_payload(support_row)["support"] for support_row in selected_support_rows]
        support_id_suffix = ",".join(support_row["support_id"] for support_row in selected_support_rows)
        pair_payload = {
            "pair_split": split_name,
            "env_key": env_key,
            "category": row.get("category") or payload.get("category"),
            "experience_key": _experience_key(row),
            "supports": supports,
            "queries": [payload["query"]],
        }
        pair_id = f"{split_name}:{support_count}supports:{support_id_suffix}->{row['query_id']}"
        output_rows.append(
            {
                "data_source": "bfcl_meta_summary",
                "pair_id": pair_id,
                "env_key": env_key,
                "category": row.get("category") or payload.get("category"),
                "experience_key": _experience_key(row),
                "support_id": support_id_suffix,
                "query_id": row["query_id"],
                "flag": pair_id,
                "reward_model": None,
                "tools_kwargs": "",
                "messages": "",
                "total_messages": json.dumps(pair_payload, ensure_ascii=False),
            }
        )
    return output_rows, {
        "num_rows_total": num_rows_total,
        "num_rows_kept": num_rows_kept,
        "num_rows_skipped_insufficient_supports": num_rows_skipped_insufficient_supports,
        "num_envs": len(support_pool_by_env),
        "num_unique_supports_total": num_unique_supports_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--support-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = _load_rows(os.path.abspath(os.path.expanduser(args.input)))
    output_rows, stats = build_multi_support_rows(rows, args.split, args.support_count, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(args.output))), exist_ok=True)
    datasets.Dataset.from_list(output_rows).to_parquet(os.path.abspath(os.path.expanduser(args.output)))
    print(
        json.dumps(
            {
                "input": os.path.abspath(os.path.expanduser(args.input)),
                "output": os.path.abspath(os.path.expanduser(args.output)),
                "split": args.split,
                "support_count": args.support_count,
                "seed": args.seed,
                "num_rows": len(output_rows),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
