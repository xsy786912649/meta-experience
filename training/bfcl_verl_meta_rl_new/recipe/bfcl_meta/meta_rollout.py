import json
import uuid
from typing import Any, Awaitable, Callable
from copy import deepcopy

from recipe.bfcl_meta.meta_env import (
    _make_json_serializable,
    build_task_system_prompt,
    build_trajectory_text,
    extract_summary_context_text,
    snapshot_state_items,
)
from recipe.bfcl_multiturn.bfcl_completion_callback import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    _normalize_tool_calls,
    _to_tool_response_message,
)
from recipe.bfcl_multiturn.bfcl_env import (
    decode_tool_calls,
    execute_multi_turn_func_call,
    multi_turn_checker,
)


def _is_context_overflow_error(msg: str) -> bool:
    lowered = (msg or "").lower()
    return "maximum context length" in lowered and "tokens" in lowered


class MetaRolloutEngine:
    def __init__(
        self,
        tokenizer,
        chat_fn: Callable[[list[dict[str, str]], float], Awaitable[Any]],
        max_model_len: int,
        max_assistant_turns: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_fn = chat_fn
        self.max_model_len = int(max_model_len)
        self.max_assistant_turns = int(max_assistant_turns)

    def _token_len(self, messages: list[dict[str, str]]) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    @staticmethod
    def _rollback_trailing_users(messages: list[dict[str, str]], min_len: int = 0) -> None:
        while len(messages) > min_len and messages[-1].get("role") == "user":
            messages.pop()

    @staticmethod
    def _rollback_trailing_tool_responses(messages: list[dict[str, str]]) -> None:
        while (
            messages
            and messages[-1].get("role") == "user"
            and isinstance(messages[-1].get("content"), str)
            and messages[-1]["content"].startswith("<tool_response>")
        ):
            messages.pop()

    async def run_task_rollout(
        self,
        payload: dict[str, Any],
        temperature: float,
        experience_summary: str | None,
        max_model_len: int | None = None,
        exploration_mode: bool = False,
    ) -> dict[str, Any]:
        effective_max_model_len = int(max_model_len) if max_model_len is not None else self.max_model_len
        run_namespace = f"meta_{uuid.uuid4().hex[:10]}"
        system_prompt = build_task_system_prompt(
            payload["function"],
            experience_summary=experience_summary,
            exploration_mode=exploration_mode,
        )
        messages = [{"role": "system", "content": system_prompt}]
        system_prompt_len = len(messages)
        inference_log: list[Any] = []
        model_result_decoded = [[] for _ in payload["ground_truth"]]
        force_quit = False
        assistant_turns = 0

        _, involved_instances = execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=payload["initial_config"],
            involved_classes=payload["involved_classes"],
            run_namespace=run_namespace,
            test_entry_id=payload["id"],
            long_context=("long_context" in payload["category"]),
            is_eval_run=False,
        )

        initial_state = snapshot_state_items(involved_instances)
        if initial_state:
            inference_log.append(initial_state)

        for turn_idx, current_turn_messages in enumerate(payload["question"]):
            if str(turn_idx) in payload.get("missed_function", {}):
                current_turn_messages = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                            functions=json.dumps(payload["missed_function"][str(turn_idx)], ensure_ascii=False)
                        ),
                    }
                ]

            messages.extend(current_turn_messages)
            if self._token_len(messages) > effective_max_model_len:
                self._rollback_trailing_users(messages, min_len=system_prompt_len)
                force_quit = True
                inference_log.append(
                    {
                        "begin_of_turn_query": current_turn_messages,
                        "step_0": [
                            {
                                "role": "handler_log",
                                "content": "Support/query rollout exceeded context after appending user turn.",
                            }
                        ],
                    }
                )
                break

            turn_log = {"begin_of_turn_query": current_turn_messages}
            step_idx = 0

            while True:
                if assistant_turns >= self.max_assistant_turns:
                    force_quit = True
                    turn_log[f"step_{step_idx}"] = [
                        {
                            "role": "handler_log",
                            "content": f"Support/query rollout exceeded {self.max_assistant_turns} assistant steps.",
                        }
                    ]
                    break

                step_events = []
                turn_log[f"step_{step_idx}"] = step_events
                try:
                    completions = await self.chat_fn(messages, temperature)
                except Exception as exc:
                    if _is_context_overflow_error(str(exc)):
                        self._rollback_trailing_users(messages, min_len=system_prompt_len)
                        step_events.append(
                            {
                                "role": "handler_log",
                                "content": "Model query failed with context overflow during support/query rollout.",
                                "error": str(exc),
                            }
                        )
                    else:
                        step_events.append(
                            {
                                "role": "handler_log",
                                "content": "Model query failed during support/query rollout.",
                                "error": str(exc),
                            }
                        )
                    step_events.append(
                        {
                            "role": "handler_log",
                            "content": "Rollout terminated as failure.",
                        }
                    )
                    force_quit = True
                    break

                finish_reason = completions.choices[0].finish_reason
                content = _normalize_tool_calls(completions.choices[0].message.content or "")
                messages.append({"role": "assistant", "content": content})
                step_events.append({"role": "assistant", "content": content})
                assistant_turns += 1

                if finish_reason == "length":
                    force_quit = True
                    break

                decoded_calls = decode_tool_calls(content)
                if not decoded_calls:
                    break

                model_result_decoded[turn_idx].append(decoded_calls)
                execution_results, involved_instances = execute_multi_turn_func_call(
                    func_call_list=decoded_calls,
                    initial_config=payload["initial_config"],
                    involved_classes=payload["involved_classes"],
                    run_namespace=run_namespace,
                    test_entry_id=payload["id"],
                    long_context=("long_context" in payload["category"]),
                    is_eval_run=False,
                )

                for execution_result in execution_results:
                    messages.append(_to_tool_response_message(execution_result))
                    step_events.append({"role": "tool", "content": execution_result})

                if self._token_len(messages) > effective_max_model_len:
                    self._rollback_trailing_tool_responses(messages)
                    step_events.append(
                        {
                            "role": "handler_log",
                            "content": "Tool responses were rolled back because they exceeded context.",
                        }
                    )
                    force_quit = True
                    break

                step_idx += 1

            inference_log.append(turn_log)
            state_items = snapshot_state_items(involved_instances)
            if state_items:
                inference_log.append(state_items)

            if force_quit:
                break

        checker = multi_turn_checker(
            multi_turn_model_result_list_decoded=model_result_decoded,
            multi_turn_ground_truth_list=payload["ground_truth"],
            test_entry=payload,
            run_namespace=run_namespace,
        )
        if force_quit and checker.get("valid"):
            checker = {"valid": False, "error_type": "meta:forced_stop"}
        checker = _make_json_serializable(checker)

        return {
            "success": bool(checker.get("valid")),
            "checker": checker,
            "trajectory_text": build_trajectory_text(system_prompt=system_prompt, inference_log=inference_log),
            "conversation": deepcopy(messages),
        }


def parse_summary_generation(completions: Any) -> tuple[str, str]:
    full_output = (completions.choices[0].message.content or "").strip()
    context_text = extract_summary_context_text(full_output)
    return full_output, context_text
