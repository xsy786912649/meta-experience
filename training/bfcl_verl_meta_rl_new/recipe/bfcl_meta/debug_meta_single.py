import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import aiohttp


async def _chat(base_url: str, model_name: str, messages: list[dict[str, str]], temperature: float, top_p: float):
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url=f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": "Bearer token-abc123", "x-request-id": uuid.uuid4().hex},
            json={
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            },
        ) as resp:
            data = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(str(data))
            from openai.types.chat.chat_completion import ChatCompletion

            return ChatCompletion(**data)


def _load_pair(path: str, pair_id: str | None, index: int | None):
    import datasets

    rows = datasets.load_dataset("parquet", data_files=path)["train"].to_list()
    if pair_id is not None:
        for row in rows:
            if row.get("pair_id") == pair_id:
                return row
        raise ValueError(f"pair_id not found: {pair_id}")
    if index is None:
        index = 0
    return rows[index]


async def main_async(args):
    repo_root = Path(args.repo_root).resolve()
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = repo_root / data_file

    sys.path.insert(0, str(repo_root))
    from recipe.bfcl_meta.meta_env import build_summary_prompt_messages
    from recipe.bfcl_meta.meta_rollout import MetaRolloutEngine, parse_summary_generation
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    pair_row = _load_pair(str(data_file), args.pair_id, args.index)
    payload = pair_row["total_messages"]
    if isinstance(payload, str):
        payload = json.loads(payload)

    async def chat_fn(messages, temperature):
        return await _chat(args.base_url, args.served_model, messages, temperature, args.top_p)

    engine = MetaRolloutEngine(
        tokenizer=tokenizer,
        chat_fn=chat_fn,
        max_model_len=args.max_model_len,
        max_assistant_turns=args.max_assistant_turns,
    )

    support_result = await engine.run_task_rollout(
        payload=payload["support"],
        temperature=args.support_temperature,
        experience_summary=None,
        max_model_len=args.support_max_model_len,
        exploration_mode=True,
    )
    summary_messages = build_summary_prompt_messages(
        tokenizer=tokenizer,
        trajectory_text=support_result["trajectory_text"],
        support_success=support_result["success"],
        checker=support_result["checker"],
        max_prompt_tokens=args.summary_prompt_budget,
    )
    summary_completion = await chat_fn(summary_messages, args.summary_temperature)
    summary_full_output, summary_context_text = parse_summary_generation(summary_completion)
    query_result = await engine.run_task_rollout(
        payload=payload["query"],
        temperature=args.query_temperature,
        experience_summary=summary_context_text,
    )

    print("=== pair ===")
    print(json.dumps({"pair_id": pair_row.get("pair_id"), "support_id": pair_row.get("support_id"), "query_id": pair_row.get("query_id")}, ensure_ascii=False, indent=2))
    print("=== support_success ===")
    print(support_result["success"])
    print("=== support_checker ===")
    print(json.dumps(support_result["checker"], ensure_ascii=False, indent=2))
    print("=== support_trajectory ===")
    print(support_result["trajectory_text"])
    print("=== summary_prompt ===")
    print(tokenizer.apply_chat_template(summary_messages, tokenize=False, add_generation_prompt=True))
    print("=== summary_full_output ===")
    print(summary_full_output)
    print("=== summary_context_text ===")
    print(summary_context_text)
    print("=== query_success ===")
    print(query_result["success"])
    print("=== query_checker ===")
    print(json.dumps(query_result["checker"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-file", default="data/bfcl_meta_rl/test_seen.parquet")
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:8010/v1")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="Qwen3-8B")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--support-temperature", type=float, default=1.0)
    parser.add_argument("--summary-temperature", type=float, default=1.0)
    parser.add_argument("--query-temperature", type=float, default=0.2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--support-max-model-len", type=int, default=28672)
    parser.add_argument("--max-assistant-turns", type=int, default=50)
    parser.add_argument("--summary-prompt-budget", type=int, default=29696)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
