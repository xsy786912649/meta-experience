# Copyright Sierra

import argparse
import json
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional

from litellm import completion, provider_list

from tau_bench.envs import get_env
from tau_bench.envs.user import UserStrategy
from tau_bench.run import agent_factory
from tau_bench.types import Action, EnvRunResult, RESPOND_ACTION_NAME, SolveResult


MEMO_HEADER = "Environment experience memo"


class AdaptedToolCallingAgent:
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        completion_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs or {}

    def solve(self, env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
                **self.completion_kwargs,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend([next_message, {"role": "user", "content": env_response.observation}])
            if env_response.done:
                break
        return SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)


def message_to_action(message: Dict[str, Any]) -> Action:
    if (
        "tool_calls" in message
        and message["tool_calls"] is not None
        and len(message["tool_calls"]) > 0
        and message["tool_calls"][0]["function"] is not None
    ):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})


def build_agent(tools_info: List[Dict[str, Any]], wiki: str, args: argparse.Namespace):
    if args.agent_completion_kwargs and args.agent_strategy != "tool-calling":
        raise ValueError("--agent-completion-kwargs currently supports --agent-strategy tool-calling only")
    if args.agent_strategy == "tool-calling":
        return AdaptedToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=args.model,
            provider=args.model_provider,
            temperature=args.temperature,
            completion_kwargs=args.agent_completion_kwargs,
        )
    return agent_factory(tools_info, wiki, args)


def format_trajectory(messages: List[Dict[str, Any]], reward: float, info: Dict[str, Any]) -> str:
    lines = [f"Reward: {reward}", f"Info: {json.dumps(info, ensure_ascii=False, default=str)}"]
    for message in messages:
        role = message.get("role", "unknown")
        if role == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                lines.append(
                    "assistant tool_call: "
                    + json.dumps(
                        {
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            content = message.get("content")
            if content is not None:
                lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_support_trajectories(
    support_results: List[Dict[str, Any]],
    model: str,
    provider: str,
    temperature: float,
    completion_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[str, float]:
    support_text = "\n\n".join(
        [
            f"### Support task {item['task_id']}\n"
            f"{format_trajectory(item['result'].messages, item['result'].reward, item['result'].info)}"
            for item in support_results
        ]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You extract reusable environment-specific guidance from completed tool-agent trajectories. "
                "Write a compact memo that helps the same agent solve later tasks in the same benchmark environment. "
                "Focus on policies, tool semantics, database constraints, common failure modes, and successful action patterns. "
                "Do not mention task ids unless needed. Do not invent facts beyond the trajectories."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the following support trajectories into an environment experience memo.\n\n"
                f"{support_text}"
            ),
        },
    ]
    res = completion(
        messages=messages,
        model=model,
        custom_llm_provider=provider,
        temperature=temperature,
        **(completion_kwargs or {}),
    )
    cost = res._hidden_params.get("response_cost") or 0.0
    return res.choices[0].message.content or "", cost


def add_memo_to_wiki(wiki: str, memo: str) -> str:
    if not memo.strip():
        return wiki
    return f"{wiki}\n\n## {MEMO_HEADER}\n{memo.strip()}\n"


def select_support_task_ids(
    query_task_id: int,
    num_tasks: int,
    support_count: int,
    rng: random.Random,
) -> List[int]:
    candidates = [idx for idx in range(num_tasks) if idx != query_task_id]
    rng.shuffle(candidates)
    return candidates[:support_count]


def pass_hat(results: List[EnvRunResult]) -> Dict[int, float]:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    c_per_task_id: dict[int, int] = {}
    for result in results:
        c_per_task_id.setdefault(result.task_id, 0)
        c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    out: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        out[k] = sum(comb(c, k) / comb(num_trials, k) for c in c_per_task_id.values()) / len(c_per_task_id)
    return out


def run_one_query(args: argparse.Namespace, query_task_id: int, trial: int, adaptation_count: int) -> EnvRunResult:
    base_env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
        task_split=args.task_split,
        task_index=0,
    )
    rng = random.Random(args.seed + trial * 100000 + query_task_id * 100 + adaptation_count)
    support_task_ids = select_support_task_ids(query_task_id, len(base_env.tasks), adaptation_count, rng)

    support_results: List[Dict[str, Any]] = []
    support_cost = 0.0
    memo = ""
    memo_cost = 0.0
    try:
        if adaptation_count > 0:
            support_agent = build_agent(base_env.tools_info, base_env.wiki, args)
            for support_task_id in support_task_ids:
                support_env = get_env(
                    args.env,
                    user_strategy=args.user_strategy,
                    user_model=args.user_model,
                    user_provider=args.user_model_provider,
                    task_split=args.task_split,
                    task_index=0,
                )
                support_result = support_agent.solve(
                    env=support_env,
                    task_index=support_task_id,
                    max_num_steps=args.max_support_steps,
                )
                support_cost += support_result.total_cost or 0.0
                support_results.append({"task_id": support_task_id, "result": support_result})
            memo, memo_cost = summarize_support_trajectories(
                support_results=support_results,
                model=args.summary_model or args.model,
                provider=args.summary_model_provider or args.model_provider,
                temperature=args.memo_temperature,
                completion_kwargs=args.summary_completion_kwargs,
            )

        query_wiki = add_memo_to_wiki(base_env.wiki, memo)
        query_agent = build_agent(base_env.tools_info, query_wiki, args)
        query_env = get_env(
            args.env,
            user_strategy=args.user_strategy,
            user_model=args.user_model,
            user_provider=args.user_model_provider,
            task_split=args.task_split,
            task_index=0,
        )
        query_result: SolveResult = query_agent.solve(
            env=query_env,
            task_index=query_task_id,
            max_num_steps=args.max_query_steps,
        )
        info = {
            **query_result.info,
            "adaptation_count": adaptation_count,
            "support_task_ids": support_task_ids,
            "support_rewards": [item["result"].reward for item in support_results],
            "memo": memo,
            "support_cost": support_cost,
            "memo_cost": memo_cost,
            "query_cost": query_result.total_cost or 0.0,
        }
        return EnvRunResult(
            task_id=query_task_id,
            reward=query_result.reward,
            info=info,
            traj=query_result.messages,
            trial=trial,
        )
    except Exception as exc:
        return EnvRunResult(
            task_id=query_task_id,
            reward=0.0,
            info={
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "adaptation_count": adaptation_count,
                "support_task_ids": support_task_ids,
                "support_rewards": [item["result"].reward for item in support_results],
                "memo": memo,
            },
            traj=[],
            trial=trial,
        )


def run(args: argparse.Namespace) -> List[EnvRunResult]:
    assert args.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert args.model_provider in provider_list, "Invalid model provider"
    assert args.user_model_provider in provider_list, "Invalid user model provider"
    assert args.agent_strategy in ["tool-calling", "act", "react", "few-shot"], "Invalid agent strategy"
    assert args.task_split in ["train", "test", "dev"], "Invalid task split"
    assert args.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    env = get_env(
        args.env,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
        task_split=args.task_split,
        task_index=0,
    )
    end_index = len(env.tasks) if args.end_index == -1 else min(args.end_index, len(env.tasks))
    task_ids = args.task_ids if args.task_ids else list(range(args.start_index, end_index))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(task_ids)

    os.makedirs(args.log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    all_results: List[EnvRunResult] = []
    summary_rows = []

    for adaptation_count in args.adaptation_counts:
        ckpt_path = (
            f"{args.log_dir}/env-adapt{adaptation_count}-{args.agent_strategy}-"
            f"{args.model.split('/')[-1]}-{args.temperature}_{args.env}_{args.task_split}_{time_str}.json"
        )
        results: List[EnvRunResult] = []
        print(f"Running adaptation_count={adaptation_count}, tasks={task_ids}, checkpoint={ckpt_path}")
        for trial in range(args.num_trials):
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                trial_results = list(
                    executor.map(
                        lambda task_id: run_one_query(args, task_id, trial, adaptation_count),
                        task_ids,
                    )
                )
            results.extend(trial_results)
            with open(ckpt_path, "w") as f:
                json.dump([result.model_dump() for result in results], f, indent=2)

        rewards = [result.reward for result in results]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        pass_hats = pass_hat(results) if results else {}
        summary = {
            "env": args.env,
            "task_split": args.task_split,
            "adaptation_count": adaptation_count,
            "num_results": len(results),
            "avg_reward": avg_reward,
            "pass_hat": pass_hats,
            "results_path": ckpt_path,
        }
        summary_rows.append(summary)
        all_results.extend(results)
        print(json.dumps(summary, indent=2))

    summary_path = f"{args.log_dir}/env-adaptation-summary-{args.env}-{args.task_split}_{time_str}.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"Summary saved to {summary_path}")
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--env", type=str, choices=["retail", "airline"], default="retail")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=provider_list, required=True)
    parser.add_argument("--user-model", type=str, default="gpt-4o")
    parser.add_argument("--user-model-provider", type=str, choices=provider_list, required=True)
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Model used to summarize support trajectories. Defaults to --model.",
    )
    parser.add_argument(
        "--summary-model-provider",
        type=str,
        choices=provider_list,
        default=None,
        help="Provider for --summary-model. Defaults to --model-provider.",
    )
    parser.add_argument("--agent-strategy", type=str, default="tool-calling", choices=["tool-calling", "act", "react", "few-shot"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--memo-temperature", type=float, default=0.0)
    parser.add_argument("--task-split", type=str, default="test", choices=["train", "test", "dev"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--adaptation-counts", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-support-steps", type=int, default=30)
    parser.add_argument("--max-query-steps", type=int, default=30)
    parser.add_argument("--log-dir", type=str, default="results_env_adaptation")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str)
    parser.add_argument(
        "--agent-completion-kwargs",
        type=json.loads,
        default={},
        help='JSON kwargs passed only to agent model calls, e.g. {"api_base":"http://host:8000/v1","api_key":"token"}.',
    )
    parser.add_argument(
        "--summary-completion-kwargs",
        type=json.loads,
        default=None,
        help="JSON kwargs passed only to memo summary model calls. Defaults to --agent-completion-kwargs.",
    )
    args = parser.parse_args()
    if args.summary_completion_kwargs is None:
        args.summary_completion_kwargs = dict(args.agent_completion_kwargs)
    return args


if __name__ == "__main__":
    run(parse_args())
