import json
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback

from recipe.bfcl_multiturn.bfcl_env import (
    bounded_should_stop,
    build_reward_vector,
    convert_to_function_calls,
    decode_tool_calls,
    execute_multi_turn_func_call,
    extract_tool_calls_from_text,
    multi_turn_checker,
)

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING = (
    "{functions}\nI have updated some more functions you can choose from. What about now?"
)


def keep_last_one_rows(x: torch.Tensor) -> torch.Tensor:
    x_np = np.asarray(x.clone())
    next_col = np.concatenate([x_np[:, 1:], np.zeros((x_np.shape[0], 1), dtype=x_np.dtype)], axis=1)
    result = ((x_np == 1) & (next_col == 0)).astype(np.float32)
    return torch.tensor(result, dtype=torch.float32)


def _normalize_tool_calls(text: str) -> str:
    if not text:
        return ""
    if "</tool_call>" in text:
        last_close = text.rfind("</tool_call>")
        text = text[: last_close + len("</tool_call>")]
    return text.strip().replace("\n\n", "\n")


def _to_tool_response_message(content: str) -> dict:
    return {"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"}


class BFCLMultiTurnCompletionCallback(ToolCompletionCallback):
    def __init__(self, config: DictConfig, scheduler: ChatCompletionScheduler):
        super().__init__(config, scheduler)
        self.max_assistant_turns = config.actor_rollout_ref.rollout.max_assistant_turns

    def _init_episode(self, info: dict, payload: dict):
        run_namespace = f"bfcl_{uuid.uuid4().hex[:10]}"
        info["__bfcl_state__"] = {
            "payload": payload,
            "run_namespace": run_namespace,
            "turn_idx": 0,
            "assistant_turns": 0,
            "model_result_decoded": [[] for _ in payload["ground_truth"]],
        }
        execute_multi_turn_func_call(
            func_call_list=[],
            initial_config=payload["initial_config"],
            involved_classes=payload["involved_classes"],
            run_namespace=run_namespace,
            test_entry_id=payload["id"],
            long_context=("long_context" in payload["category"]),
            is_eval_run=False,
        )

    def _finish_episode(self, messages: list[dict], info: dict, success: bool):
        state = info["__bfcl_state__"]
        rewards = build_reward_vector(state["assistant_turns"], success=success)
        messages.append({"reward": rewards})

    async def __call__(
        self,
        messages: list[dict[str, str]],
        completions,
        info: dict[str, Any],
        flag,
        reward_reference,
        total_messages,
    ):
        payload = total_messages if isinstance(total_messages, dict) else json.loads(total_messages)
        if "__bfcl_state__" not in info:
            self._init_episode(info, payload)
        state = info["__bfcl_state__"]

        content = _normalize_tool_calls(completions.choices[0].message.content or "")
        messages.append({"role": "assistant", "content": content})
        state["assistant_turns"] += 1

        if bounded_should_stop(state["assistant_turns"]) or state["assistant_turns"] >= self.max_assistant_turns:
            self._finish_episode(messages, info, success=False)
            return

        turn_idx = state["turn_idx"]
        decoded_calls = decode_tool_calls(content)
        if decoded_calls:
            state["model_result_decoded"][turn_idx].append(decoded_calls)
            execution_results, _ = execute_multi_turn_func_call(
                func_call_list=decoded_calls,
                initial_config=payload["initial_config"],
                involved_classes=payload["involved_classes"],
                run_namespace=state["run_namespace"],
                test_entry_id=payload["id"],
                long_context=("long_context" in payload["category"]),
                is_eval_run=False,
            )
            for res in execution_results:
                messages.append(_to_tool_response_message(res))
            self.scheduler.submit_chat_completions(
                messages=messages,
                request_id=completions.id,
                info=info,
                flag=flag,
                reward_reference=reward_reference,
                total_messages=total_messages,
            )
            return

        state["turn_idx"] += 1
        if state["turn_idx"] >= len(payload["question"]):
            checker = multi_turn_checker(
                multi_turn_model_result_list_decoded=state["model_result_decoded"],
                multi_turn_ground_truth_list=payload["ground_truth"],
                test_entry=payload,
                run_namespace=state["run_namespace"],
            )
            self._finish_episode(messages, info, success=bool(checker.get("valid")))
            return

        next_turn_idx = state["turn_idx"]
        holdout_function = payload.get("missed_function", {})
        if str(next_turn_idx) in holdout_function:
            prompt = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                functions=json.dumps(holdout_function[str(next_turn_idx)], ensure_ascii=False)
            )
            messages.append({"role": "user", "content": prompt})
        else:
            for msg in payload["question"][next_turn_idx]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        self.scheduler.submit_chat_completions(
            messages=messages,
            request_id=completions.id,
            info=info,
            flag=flag,
            reward_reference=reward_reference,
            total_messages=total_messages,
        )

    def postprocess(
        self,
        batch: DataProto,
        batch_conversations: list[list[dict[str, str]]],
        batch_tools_content,
        batch_flag,
        batch_reward_reference,
        batch_total_messages,
        batch_reward,
        n: int,
    ) -> DataProto:
        prompts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in batch.non_tensor_batch["raw_prompt"]
        ]
        assert len(batch_conversations) == len(prompts) * n

        sequences = [
            self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            for conversation in batch_conversations
        ]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        response_mask = self._mask_out_tools_calling_tokens(
            batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0),
            batch_conversations,
            responses["input_ids"],
            responses["attention_mask"],
        )
        reward_mask = keep_last_one_rows(response_mask)

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch_td = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "response_mask": response_mask,
                "reward_mask": reward_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )
        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
        rewards = np.array([reward for reward in batch_reward], dtype=object)
        return DataProto(batch=batch_td, non_tensor_batch={"__num_turns__": num_turns, "__reward__": rewards})

