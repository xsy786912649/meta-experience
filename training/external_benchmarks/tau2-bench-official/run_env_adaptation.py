import argparse
import hashlib
import json
import os
import random
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from litellm import completion
from loguru import logger

from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.runner.build import _build_env_kwargs, build_agent, build_environment, build_user
from tau2.runner.helpers import get_options, get_tasks
from tau2.runner.simulation import run_simulation


MEMO_HEADER = "Environment experience memo"

SUPPORT_EXPLORATION_PREFIX = (
    "This is an early attempt in a new tool-calling environment. "
    "Try to solve the task, but also use careful exploration to learn tool behavior, constraints, and failure modes.\n\n"
)

QUERY_GUIDANCE_PREFIX = (
    "You are given a tool-using guidance extracted from a previous task in the tool-calling environment. "
    "Use it as heuristic guidance when it is relevant.\n\n"
    "# Guidance:\n"
)

SUMMARY_SYSTEM_PROMPT = "You are extracting an tool-using guidance from your own support tool-calling trajectory."

SUMMARY_USER_PROMPT_PREFIX = (
    "First think and analyze the trajectory step-by-step and identify, (i) the exact final failure point(s), (ii) the root cause (not just symptoms), (iii) the hidden preconditions that were violated, (iv) the incorrect assumptions and problematic patterns by the agent. \n"
    "Then, propose a guidance to avoid mistakes in future other tool-calling tasks. Keep your final guidance concise.\n\n"
    "The guidance potentially include two part:\n"
    "-1 guidance to avoid the identified mistakes, such as action-guiding rules, environment-specific constraints, and hidden preconditions\n"
    "-2 high-level and generalized strategy that applies across tasks and tools\n\n"
    "Do not:\n"
    "- answer the user\n"
    "- restate the whole trajectory\n"
    "- output any tool call, function call, XML tool tag, or executable next step\n"
    "- continue solving the task\n"
    "- include one-off IDs, values, or details unless they imply a reusable rule\n"
    "- add assistant-style closing language or filler\n\n"
)


def _message_to_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump(mode="json")
    if isinstance(message, dict):
        return message
    return {"repr": repr(message)}


def format_simulation(simulation: SimulationRun) -> str:
    reward = simulation.reward_info.reward if simulation.reward_info else None
    lines = [
        f"Task id: {simulation.task_id}",
        f"Reward: {reward}",
        f"Termination: {simulation.termination_reason}",
    ]
    for message in simulation.get_messages():
        payload = _message_to_dict(message)
        lines.append(json.dumps(payload, ensure_ascii=False, default=str))
    return "\n".join(lines)


def extract_summary_context_text(summary_output: str) -> str:
    text = (summary_output or "").strip()
    if not text:
        return ""
    if "</think>" in text:
        distilled = text.split("</think>")[-1].strip()
        if distilled:
            return distilled
    return text


def summarize_checker_result(simulation: SimulationRun) -> str:
    reward = simulation.reward_info.reward if simulation.reward_info else None
    if reward is not None and abs(float(reward) - 1.0) < 1e-6:
        return "The support trajectory successfully completed the task."
    parts = [
        "The support trajectory failed or did not receive full reward.",
        f"Reward: {reward}",
        f"Termination: {simulation.termination_reason}",
    ]
    if simulation.reward_info:
        parts.append(json.dumps(simulation.reward_info.model_dump(mode="json"), ensure_ascii=False, default=str)[:1200])
    return "\n".join(parts)


def summarize_support_simulations(
    support_simulations: list[SimulationRun],
    model: str,
    llm_args: dict,
) -> str:
    support_text = "\n\n".join(
        [
            f"### Support task {simulation.task_id}\n{format_simulation(simulation)}"
            for simulation in support_simulations
        ]
    )
    checker_text = "\n\n".join(
        [
            f"### Support task {simulation.task_id}\n{summarize_checker_result(simulation)}"
            for simulation in support_simulations
        ]
    )
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                SUMMARY_USER_PROMPT_PREFIX
                + "Support trajectory:\n"
                + support_text
                + "\n\nSupport checker summary:\n"
                + checker_text
                + "\n\nNow write the guidance."
            ),
        },
    ]
    response = completion(messages=messages, model=model, **llm_args)
    return extract_summary_context_text(response.choices[0].message.content or "")


def add_memo_to_policy(policy: str, memo: str) -> str:
    if not memo.strip():
        return policy
    return f"{policy}\n\n{QUERY_GUIDANCE_PREFIX}{memo.strip()}"


def add_exploration_to_policy(policy: str) -> str:
    return f"{policy}\n\n{SUPPORT_EXPLORATION_PREFIX.strip()}"


def build_text_orchestrator_with_policy(
    config: TextRunConfig,
    task: Task,
    *,
    policy_override: Optional[str] = None,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_persona_config: Optional[PersonaConfig] = None,
) -> Orchestrator:
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())
    if seed is None:
        seed = config.seed

    solo_mode = registry.get_agent_metadata(
        config.effective_agent, "solo_mode", default=False
    )
    env_kwargs = _build_env_kwargs(config, task)
    environment = build_environment(config.domain, solo_mode=solo_mode, env_kwargs=env_kwargs)

    if policy_override is not None:
        environment.get_policy = lambda: policy_override

    agent = build_agent(
        config.effective_agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
    )
    user = build_user(
        config.effective_user,
        environment,
        task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        persona_config=user_persona_config,
        solo_mode=solo_mode,
    )
    return Orchestrator(
        domain=config.domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        seed=seed,
        solo_mode=solo_mode,
        simulation_id=simulation_id,
        validate_communication=config.enforce_communication_protocol,
        timeout=config.timeout,
    )


def run_task_with_policy(
    config: TextRunConfig,
    task: Task,
    *,
    policy_override: Optional[str],
    seed: int,
) -> SimulationRun:
    orchestrator = build_text_orchestrator_with_policy(
        config,
        task,
        policy_override=policy_override,
        seed=seed,
    )
    env_kwargs = _build_env_kwargs(config, task) or None
    return run_simulation(
        orchestrator,
        evaluation_type=EvaluationType.ALL,
        env_kwargs=env_kwargs,
    )


def select_support_tasks(
    tasks: list[Task],
    query_task: Task,
    adaptation_count: int,
    rng: random.Random,
) -> list[Task]:
    candidates = [task for task in tasks if task.id != query_task.id]
    rng.shuffle(candidates)
    return candidates[:adaptation_count]


def run_one_query(
    config: TextRunConfig,
    tasks: list[Task],
    query_task: Task,
    adaptation_count: int,
    trial: int,
    args: argparse.Namespace,
) -> dict:
    stable_task_hash = int(hashlib.sha256(str(query_task.id).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random((args.seed or 0) + trial * 100000 + stable_task_hash % 100000)
    support_tasks = select_support_tasks(tasks, query_task, adaptation_count, rng)
    support_simulations: list[SimulationRun] = []
    memo = ""
    seed_base = (args.seed or 0) + trial * 100000

    try:
        for idx, support_task in enumerate(support_tasks):
            support_env = build_environment(config.domain, env_kwargs=_build_env_kwargs(config, support_task))
            support_simulations.append(
                run_task_with_policy(
                    config,
                    support_task,
                    policy_override=add_exploration_to_policy(support_env.get_policy()),
                    seed=seed_base + idx,
                )
            )

        policy_override = None
        if adaptation_count > 0:
            memo_args = dict(args.summary_llm_args)
            memo = summarize_support_simulations(
                support_simulations=support_simulations,
                model=args.summary_llm or args.agent_llm,
                llm_args=memo_args,
            )
            base_env = build_environment(config.domain, env_kwargs=_build_env_kwargs(config, query_task))
            policy_override = add_memo_to_policy(base_env.get_policy(), memo)

        query_simulation = run_task_with_policy(
            config,
            query_task,
            policy_override=policy_override,
            seed=seed_base + 9999,
        )
        reward = query_simulation.reward_info.reward if query_simulation.reward_info else None
        return {
            "task_id": query_task.id,
            "trial": trial,
            "adaptation_count": adaptation_count,
            "reward": reward,
            "support_task_ids": [task.id for task in support_tasks],
            "support_rewards": [
                simulation.reward_info.reward if simulation.reward_info else None
                for simulation in support_simulations
            ],
            "memo": memo,
            "query_simulation": query_simulation.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.exception(f"Failed task={query_task.id} adaptation_count={adaptation_count}")
        return {
            "task_id": query_task.id,
            "trial": trial,
            "adaptation_count": adaptation_count,
            "reward": 0.0,
            "support_task_ids": [task.id for task in support_tasks],
            "support_rewards": [
                simulation.reward_info.reward if simulation.reward_info else None
                for simulation in support_simulations
            ],
            "memo": memo,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run(args: argparse.Namespace) -> list[dict]:
    task_set_name = args.task_set_name or args.domain
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
    )
    config = TextRunConfig(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        agent=args.agent,
        llm_agent=args.agent_llm,
        llm_args_agent=args.agent_llm_args,
        user=args.user,
        llm_user=args.user_llm,
        llm_args_user=args.user_llm_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        enforce_communication_protocol=args.enforce_communication_protocol,
        retrieval_config=args.retrieval_config,
        retrieval_config_kwargs=args.retrieval_config_kwargs,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []
    summary: list[dict] = []

    for adaptation_count in args.adaptation_counts:
        rows: list[dict] = []
        output_path = Path(args.log_dir) / (
            f"tau2-env-adapt{adaptation_count}-{args.domain}-{args.task_split_name}-{time_str}.json"
        )
        print(f"Running tau2 env adaptation: domain={args.domain}, adaptation_count={adaptation_count}")
        for trial in range(args.num_trials):
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                futures = {
                    executor.submit(run_one_query, config, tasks, task, adaptation_count, trial, args): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    row = future.result()
                    rows.append(row)
                    print(
                        f"[tau2-bench] adapt={adaptation_count} trial={trial} "
                        f"task={task.id} reward={row.get('reward')} checkpoint={output_path}",
                        flush=True,
                    )
                    with open(output_path, "w") as f:
                        json.dump(rows, f, indent=2)

        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)

        rewards = [float(row.get("reward") or 0.0) for row in rows]
        summary_row = {
            "domain": args.domain,
            "task_split_name": args.task_split_name,
            "adaptation_count": adaptation_count,
            "num_results": len(rows),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "successes": sum(1 for reward in rewards if abs(reward - 1.0) < 1e-6),
            "results_path": str(output_path),
        }
        summary.append(summary_row)
        all_rows.extend(rows)
        print(json.dumps(summary_row, indent=2))

    summary_path = Path(args.log_dir) / f"tau2-env-adaptation-summary-{args.domain}-{time_str}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    options = get_options()
    parser.add_argument("--domain", "-d", choices=options.domains, required=True)
    parser.add_argument("--agent", default="llm_agent", choices=options.agents)
    parser.add_argument("--agent-llm", default="gpt-4.1")
    parser.add_argument("--agent-llm-args", type=json.loads, default={"temperature": 0})
    parser.add_argument("--user", default="user_simulator", choices=options.users)
    parser.add_argument("--user-llm", default="gpt-4.1")
    parser.add_argument("--user-llm-args", type=json.loads, default={"temperature": 0})
    parser.add_argument("--summary-llm", default=None)
    parser.add_argument("--summary-llm-args", type=json.loads, default={"temperature": 0})
    parser.add_argument("--task-set-name", default=None, choices=options.task_sets)
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--adaptation-counts", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default="data/env_adaptation")
    parser.add_argument("--enforce-communication-protocol", action="store_true", default=False)
    parser.add_argument("--retrieval-config", default=None)
    parser.add_argument("--retrieval-config-kwargs", type=json.loads, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
