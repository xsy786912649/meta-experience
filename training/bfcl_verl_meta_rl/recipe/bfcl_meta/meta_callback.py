import asyncio
import copy
import json
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback

from recipe.bfcl_meta.meta_env import (
    build_summary_prompt_messages,
)
from recipe.bfcl_meta.meta_rollout import MetaRolloutEngine, parse_summary_generation
from recipe.bfcl_multiturn.bfcl_completion_callback import (
    _normalize_tool_calls,
)

SUPPORT_TEMPERATURE = 1.0
QUERY_TEMPERATURE = 0.2


def _keep_last_token(mask: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(mask)
    if mask.size(1) == 0:
        return out
    last_indices = mask.sum(dim=1).long() - 1
    for row_idx, col_idx in enumerate(last_indices.tolist()):
        if col_idx >= 0:
            out[row_idx, col_idx] = 1
    return out


class BFCLMetaCompletionCallback(ToolCompletionCallback):
    def __init__(self, config: DictConfig, scheduler: ChatCompletionScheduler):
        super().__init__(config, scheduler)
        self.max_model_len = int(config.actor_rollout_ref.rollout.max_model_len)
        response_length = int(config.actor_rollout_ref.rollout.response_length)
        self.max_assistant_turns = int(config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns)
        self.summary_safety_margin = int(
            config.actor_rollout_ref.rollout.multi_turn.get("summary_safety_margin", 1024)
        )
        self.support_max_model_len = int(
            config.actor_rollout_ref.rollout.multi_turn.get(
                "support_max_model_len",
                max(2048, self.max_model_len - response_length - 1024),
            )
        )
        self.summary_prompt_budget = max(
            2048,
            self.max_model_len - response_length - self.summary_safety_margin,
        )
        self._prepared_samples: dict[str, dict[str, Any]] = {}
        self._direct_server_addresses = [address for _, address in scheduler.weighted_addresses]
        self._direct_request_counter = 0
        self._direct_request_lock = asyncio.Lock()
        self.rollout_engine = MetaRolloutEngine(
            tokenizer=self.tokenizer,
            chat_fn=self._chat_once,
            max_model_len=self.max_model_len,
            max_assistant_turns=self.max_assistant_turns,
        )

    async def prepare_batch_conversations(self, batch: DataProto, n: int) -> list[list[dict[str, str]]]:
        prepared = []
        payloads = [
            payload if isinstance(payload, dict) else json.loads(payload)
            for payload in batch.non_tensor_batch["total_messages"]
        ]
        results = await asyncio.gather(*(self._prepare_single_payload(payload, repeat_count=n) for payload in payloads))

        for result in results:
            self._prepared_samples[result["meta_instance_id"]] = result
            for _ in range(n):
                prepared.append(copy.deepcopy(result["summary_messages"]))
        return prepared

    async def _prepare_single_payload(self, payload: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        support_result = await self.rollout_engine.run_task_rollout(
            payload=payload["support"],
            temperature=SUPPORT_TEMPERATURE,
            experience_summary=None,
            max_model_len=self.support_max_model_len,
            exploration_mode=True,
        )
        summary_messages = build_summary_prompt_messages(
            tokenizer=self.tokenizer,
            trajectory_text=support_result["trajectory_text"],
            support_success=support_result["success"],
            checker=support_result["checker"],
            max_prompt_tokens=self.summary_prompt_budget,
        )
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "summary_messages": summary_messages,
            "remaining": repeat_count,
            "support_result": support_result,
        }

    async def handle_request_error(
        self,
        messages: list[dict[str, str]],
        info: dict[str, Any],
        total_messages,
        error_type: str,
        error_message: str,
    ) -> None:
        payload = total_messages if isinstance(total_messages, dict) else json.loads(total_messages)
        prepared = self._prepared_samples.get(payload["meta_instance_id"])
        messages.append({"role": "assistant", "content": ""})
        messages.append({"reward": [-0.001]})
        if prepared is not None:
            prepared["remaining"] -= 1
            if prepared["remaining"] <= 0:
                self._prepared_samples.pop(payload["meta_instance_id"], None)

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
        prepared = self._prepared_samples[payload["meta_instance_id"]]

        full_output, summary_context_text = parse_summary_generation(completions)
        messages.append({"role": "assistant", "content": full_output})

        reward = -0.001
        if summary_context_text:
            query_result = await self.rollout_engine.run_task_rollout(
                payload=payload["query"],
                temperature=QUERY_TEMPERATURE,
                experience_summary=summary_context_text,
            )
            reward = 1.0 if query_result["success"] else -0.001

        messages.append({"reward": [reward]})

        prepared["remaining"] -= 1
        if prepared["remaining"] <= 0:
            self._prepared_samples.pop(payload["meta_instance_id"], None)

    async def _chat_once(self, messages: list[dict[str, str]], temperature: float):
        async with self._direct_request_lock:
            address = self._direct_server_addresses[
                self._direct_request_counter % len(self._direct_server_addresses)
            ]
            self._direct_request_counter += 1
        sampling_params = {
            "model": self.scheduler.model_name,
            "temperature": temperature,
            "top_p": self.config.actor_rollout_ref.rollout.top_p,
        }
        return await self.scheduler._chat_completions_aiohttp(
            address,
            messages=messages,
            extra_body=self.extra_body,
            extra_headers={"x-request-id": uuid.uuid4().hex},
            **sampling_params,
        )

    def _token_len(self, messages: list[dict[str, str]]) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

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
        prompts = [conversation[:-1] for conversation in batch_conversations]
        prompt_texts = [
            self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            for prompt in prompts
        ]
        sequences = [
            self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
            for conversation in batch_conversations
        ]
        responses = [sequence[len(prompt_texts[i]) :] for i, sequence in enumerate(sequences)]

        prompt_tensors = self.tokenizer(prompt_texts, return_tensors="pt", padding="longest", padding_side="left")
        response_tensors = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")

        response_mask = response_tensors["attention_mask"].to(dtype=torch.float32)
        terminal_mask = _keep_last_token(response_mask)

        input_ids = torch.cat([prompt_tensors["input_ids"], response_tensors["input_ids"]], dim=1)
        attention_mask = torch.cat(
            [prompt_tensors["attention_mask"], response_tensors["attention_mask"]], dim=1
        )
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch_td = TensorDict(
            {
                "prompts": prompt_tensors["input_ids"],
                "responses": response_tensors["input_ids"],
                "response_mask": response_mask,
                "reward_mask": terminal_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )
        rewards = np.array([reward for reward in batch_reward], dtype=object)
        return DataProto(batch=batch_td, non_tensor_batch={"__reward__": rewards})
