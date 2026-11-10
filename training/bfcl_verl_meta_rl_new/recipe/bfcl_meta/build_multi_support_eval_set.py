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


def _experience_group(row: dict, payload: dict) -> tuple[str, str]:
    env_key = row["env_key"]
    category = payload["query"].get("category") or row.get("category") or payload.get("category")
    support_category = payload["support"].get("category")
    if not category or support_category != category:
        raise ValueError(
            f"Pair {row['pair_id']} must have matching support and query categories "
            f"(support={support_category!r}, query={category!r})."
        )
    return env_key, category


def _support_dedup_key(row: dict) -> str:
    payload = _decode_payload(row)
    support_payload = payload["support"]
    support_id = support_payload.get("id")
    if support_id:
        return f"id:{support_id}"
    return "json:" + json.dumps(support_payload, ensure_ascii=False, sort_keys=True)


def build_multi_support_rows(
    rows: list[dict], split_name: str, support_count: int, seed: int
) -> tuple[list[dict], dict[str, int]]:
    support_pool_by_group = defaultdict(dict)
    row_groups = []
    for row in rows:
        group = _experience_group(row, _decode_payload(row))
        row_groups.append(group)
        support_pool_by_group[group].setdefault(_support_dedup_key(row), row)

    rng = random.Random(seed)
    output_rows = []
    num_rows_total = 0
    num_rows_kept = 0
    num_rows_skipped_insufficient_supports = 0
    num_unique_supports_total = sum(len(pool) for pool in support_pool_by_group.values())
    for row, group in zip(rows, row_groups):
        num_rows_total += 1
        env_key, category = group
        experience_key = f"{env_key}::{category}"
        payload = _decode_payload(row)
        original_support_key = _support_dedup_key(row)
        extra_support_rows = [
            support_row
            for support_key, support_row in support_pool_by_group[group].items()
            if support_key != original_support_key
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
            "category": category,
            "experience_key": experience_key,
            "supports": supports,
            "queries": [payload["query"]],
        }
        pair_id = f"{split_name}:{support_count}supports:{support_id_suffix}->{row['query_id']}"
        output_rows.append(
            {
                "data_source": "bfcl_meta_summary",
                "pair_id": pair_id,
                "env_key": env_key,
                "category": category,
                "experience_key": experience_key,
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
        "num_envs": len({env_key for env_key, _ in support_pool_by_group}),
        "num_experience_groups": len(support_pool_by_group),
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
    if not output_rows:
        parser.error(f"No {args.split} pairs have {args.support_count} supports in the same environment and category.")
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
