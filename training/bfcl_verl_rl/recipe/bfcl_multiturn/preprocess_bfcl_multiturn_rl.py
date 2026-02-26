import argparse
import json
import os
import random
from collections import defaultdict

import datasets

from recipe.bfcl_multiturn.bfcl_config import MULTI_TURN_CATEGORIES
from recipe.bfcl_multiturn.bfcl_env import (
    load_dataset_entry,
    load_possible_answers,
    populate_test_case_with_predefined_functions,
)


def _env_key(entry: dict) -> str:
    return "|".join(sorted(entry["involved_classes"]))


def _tools_to_string(tools: list[dict]) -> str:
    return "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)


def _build_rows(categories: list[str]) -> list[dict]:
    rows = []
    for category in categories:
        entries = load_dataset_entry(category)
        answers = load_possible_answers(category)
        answer_map = {x["id"]: x["ground_truth"] for x in answers}

        for entry in entries:
            gt = answer_map.get(entry["id"])
            if gt is None:
                continue
            entry = populate_test_case_with_predefined_functions(entry)
            payload = {
                "id": entry["id"],
                "category": category,
                "initial_config": entry["initial_config"],
                "involved_classes": entry["involved_classes"],
                "question": entry["question"],
                "ground_truth": gt,
                "function": entry["function"],
                "missed_function": entry.get("missed_function", {}),
            }
            rows.append(
                {
                    "data_source": "bfcl_multi_turn",
                    "id": entry["id"],
                    "category": category,
                    "env_key": _env_key(entry),
                    "tools_kwargs": _tools_to_string(entry["function"]),
                    "flag": entry["id"],
                    "reward_model": json.dumps(gt, ensure_ascii=False),
                    "total_messages": json.dumps(payload, ensure_ascii=False),
                    "messages": json.dumps(entry["question"], ensure_ascii=False),
                }
            )
    return rows


def _split_by_env(rows: list[dict], unseen_env_ratio: float, seen_test_ratio: float, seed: int):
    env_to_rows = defaultdict(list)
    for row in rows:
        env_to_rows[row["env_key"]].append(row)

    envs = sorted(env_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(envs)

    num_unseen_env = max(1, int(len(envs) * unseen_env_ratio))
    unseen_envs = set(envs[:num_unseen_env])
    seen_envs = set(envs[num_unseen_env:])
    if not seen_envs:
        seen_envs = set(unseen_envs)
        unseen_envs = set()

    train, test_seen, test_unseen = [], [], []
    for env in seen_envs:
        env_rows = env_to_rows[env][:]
        rng.shuffle(env_rows)
        n_seen = max(1, int(len(env_rows) * seen_test_ratio))
        if len(env_rows) == 1:
            train.extend(env_rows)
            continue
        test_seen.extend(env_rows[:n_seen])
        train.extend(env_rows[n_seen:])

    for env in unseen_envs:
        test_unseen.extend(env_to_rows[env])

    return train, test_seen, test_unseen, seen_envs, unseen_envs


def _cap_train_rows(train_rows: list[dict], train_size: int, seed: int) -> list[dict]:
    if train_size <= 0 or len(train_rows) <= train_size:
        return train_rows
    rng = random.Random(seed)
    sampled = train_rows[:]
    rng.shuffle(sampled)
    return sampled[:train_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/bfcl_multiturn_rl")
    parser.add_argument("--categories", default=",".join(MULTI_TURN_CATEGORIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unseen_env_ratio", type=float, default=0.25)
    parser.add_argument("--seen_test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--train_size",
        type=int,
        default=512,
        help="Cap train split size. <=0 means no cap.",
    )
    args = parser.parse_args()

    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    rows = _build_rows(categories)
    train, test_seen, test_unseen, seen_envs, unseen_envs = _split_by_env(
        rows=rows,
        unseen_env_ratio=args.unseen_env_ratio,
        seen_test_ratio=args.seen_test_ratio,
        seed=args.seed,
    )
    train = _cap_train_rows(train, train_size=args.train_size, seed=args.seed)

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    datasets.Dataset.from_list(train).to_parquet(os.path.join(local_dir, "train.parquet"))
    datasets.Dataset.from_list(test_seen).to_parquet(os.path.join(local_dir, "test_seen.parquet"))
    datasets.Dataset.from_list(test_unseen).to_parquet(os.path.join(local_dir, "test_unseen.parquet"))

    summary = {
        "categories": categories,
        "num_total": len(rows),
        "num_train": len(train),
        "num_test_seen": len(test_seen),
        "num_test_unseen": len(test_unseen),
        "train_size_cap": args.train_size,
        "num_seen_envs": len(seen_envs),
        "num_unseen_envs": len(unseen_envs),
        "seen_envs": sorted(list(seen_envs)),
        "unseen_envs": sorted(list(unseen_envs)),
    }
    with open(os.path.join(local_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
