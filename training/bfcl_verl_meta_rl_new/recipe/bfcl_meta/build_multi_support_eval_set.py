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


def _env_experience_key(row: dict) -> tuple[str, str]:
    return row["env_key"], _experience_key(row)


def _support_dedup_key(row: dict) -> str:
    payload = _decode_payload(row)
    support_payload = payload["support"]
    support_id = support_payload.get("id")
    if support_id:
        return f"id:{support_id}"
    return "json:" + json.dumps(support_payload, ensure_ascii=False, sort_keys=True)


def build_multi_support_rows(rows: list[dict], split_name: str, support_count: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_env_experience_key(row)].append(row)

    rng = random.Random(seed)
    output_rows = []
    num_groups_total = 0
    num_groups_kept = 0
    num_unique_supports_total = 0
    for (env_key, experience_key), experience_rows in grouped.items():
        num_groups_total += 1
        deduped_rows_by_support = {}
        for row in experience_rows:
            deduped_rows_by_support.setdefault(_support_dedup_key(row), row)
        unique_support_rows = list(deduped_rows_by_support.values())
        num_unique_supports_total += len(unique_support_rows)

        shuffled_rows = unique_support_rows[:]
        rng.shuffle(shuffled_rows)

        for chunk_start in range(0, len(shuffled_rows), support_count):
            chunk_rows = shuffled_rows[chunk_start : chunk_start + support_count]
            if len(chunk_rows) < support_count:
                continue

            num_groups_kept += 1
            supports = [_decode_payload(row)["support"] for row in chunk_rows]
            support_id_suffix = ",".join(row["support_id"] for row in chunk_rows)
            queries = [_decode_payload(row)["query"] for row in chunk_rows]
            first_row = chunk_rows[0]
            first_payload = _decode_payload(first_row)
            pair_payload = {
                "pair_split": split_name,
                "env_key": env_key,
                "category": first_row.get("category") or first_payload.get("category"),
                "experience_key": experience_key,
                "supports": supports,
                "queries": queries,
            }
            pair_id = f"{split_name}:{support_count}supports:{support_id_suffix}"
            output_rows.append(
                {
                    "data_source": "bfcl_meta_summary",
                    "pair_id": pair_id,
                    "env_key": env_key,
                    "category": first_row.get("category") or first_payload.get("category"),
                    "experience_key": experience_key,
                    "support_id": support_id_suffix,
                    "query_id": ",".join(row["query_id"] for row in chunk_rows),
                    "flag": pair_id,
                    "reward_model": None,
                    "tools_kwargs": "",
                    "messages": "",
                    "total_messages": json.dumps(pair_payload, ensure_ascii=False),
                }
            )
    return output_rows, {
        "num_groups_total": num_groups_total,
        "num_groups_kept": num_groups_kept,
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
