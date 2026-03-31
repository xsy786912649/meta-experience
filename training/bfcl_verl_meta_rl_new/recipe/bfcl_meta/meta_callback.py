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

from recipe.bfcl_meta.meta_env import build_summary_append_user_message
from recipe.bfcl_meta.meta_rollout import MetaRolloutEngine, parse_summary_generation

SUPPORT_TEMPERATURE = 1.0
QUERY_TEMPERATURE = 0.2

PIPELINE_SHARED = "shared_support_summary"
PIPELINE_FULL = "full_support_summary"
PIPELINE_ALTERNATING = "alternating"


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
        self.train_pipeline_mode = str(
            config.actor_rollout_ref.rollout.multi_turn.get("pipeline_mode", PIPELINE_ALTERNATING)
        )
        self.val_pipeline_mode = str(
            config.actor_rollout_ref.rollout.multi_turn.get("val_pipeline_mode", PIPELINE_SHARED)
        )
        self._prepare_counter = 0
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

    def _select_batch_mode(self, validate: bool) -> str:
        requested = self.val_pipeline_mode if validate else self.train_pipeline_mode
        if requested == PIPELINE_ALTERNATING:
            mode = PIPELINE_SHARED if self._prepare_counter % 2 == 0 else PIPELINE_FULL
            self._prepare_counter += 1
            return mode
        if requested in {PIPELINE_SHARED, PIPELINE_FULL}:
            return requested
        raise ValueError(f"Unsupported bfcl_meta pipeline mode: {requested}")

    def _append_summary_request(
        self,
        support_conversation: list[dict[str, str]],
        support_success: bool,
        checker: dict[str, Any],
    ) -> list[dict[str, str]]:
        conversation = copy.deepcopy(support_conversation)
        conversation.append(build_summary_append_user_message(support_success=support_success, checker=checker))
        return conversation

    async def prepare_batch_conversations(self, batch: DataProto, n: int) -> list[list[dict[str, str]]]:
        batch_mode = self._select_batch_mode(validate=bool(batch.meta_info.get("validate", False)))
        batch.non_tensor_batch["pipeline_mode"] = np.array([batch_mode] * len(batch), dtype=object)

        prepared = []
        payloads = [
            payload if isinstance(payload, dict) else json.loads(payload)
            for payload in batch.non_tensor_batch["total_messages"]
        ]

        if batch_mode == PIPELINE_SHARED:
            results = await asyncio.gather(*(self._prepare_shared_payload(payload, repeat_count=n) for payload in payloads))
        else:
            results = await asyncio.gather(*(self._prepare_full_payload(payload, repeat_count=n) for payload in payloads))

        for result in results:
            self._prepared_samples[result["meta_instance_id"]] = {
                "remaining": result["remaining"],
                "pipeline_mode": batch_mode,
            }
            prepared.extend(result["conversations"])
        return prepared

    async def _prepare_shared_payload(self, payload: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        support_result = await self.rollout_engine.run_task_rollout(
            payload=payload["support"],
            temperature=SUPPORT_TEMPERATURE,
            experience_summary=None,
            max_model_len=self.support_max_model_len,
            exploration_mode=True,
        )
        summary_conversation = self._append_summary_request(
            support_conversation=support_result["conversation"],
            support_success=support_result["success"],
            checker=support_result["checker"],
        )
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "conversations": [copy.deepcopy(summary_conversation) for _ in range(repeat_count)],
            "remaining": repeat_count,
        }

    async def _prepare_full_payload(self, payload: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        support_results = await asyncio.gather(
            *(
                self.rollout_engine.run_task_rollout(
                    payload=payload["support"],
                    temperature=SUPPORT_TEMPERATURE,
                    experience_summary=None,
                    max_model_len=self.support_max_model_len,
                    exploration_mode=True,
                )
                for _ in range(repeat_count)
            )
        )
        conversations = [
            self._append_summary_request(
                support_conversation=result["conversation"],
                support_success=result["success"],
                checker=result["checker"],
            )
            for result in support_results
        ]
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "conversations": conversations,
            "remaining": repeat_count,
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

    def _get_prompt_message_count(self, conversation: list[dict[str, str]], pipeline_mode: str) -> int:
        if pipeline_mode == PIPELINE_SHARED:
            return max(1, len(conversation) - 1)
        for idx, message in enumerate(conversation):
            if message.get("role") == "assistant":
                return idx
        return max(1, len(conversation) - 1)

    def _encode_training_example(
        self,
        conversation: list[dict[str, str]],
        pipeline_mode: str,
    ) -> tuple[list[int], list[int], list[float]]:
        prompt_message_count = self._get_prompt_message_count(conversation, pipeline_mode)
        prompt_messages = conversation[:prompt_message_count]

        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )
        response_ids = full_ids[len(prompt_ids):]

        response_mask = []
        previous_prefix_ids = prompt_ids
        for end_idx in range(prompt_message_count + 1, len(conversation) + 1):
            prefix_ids = self.tokenizer.apply_chat_template(
                conversation[:end_idx],
                tokenize=True,
                add_generation_prompt=False,
            )
            segment_len = len(prefix_ids) - len(previous_prefix_ids)
            role = conversation[end_idx - 1].get("role")
            include_segment = role == "assistant" and (
                pipeline_mode == PIPELINE_FULL or end_idx - 1 == len(conversation) - 1
            )
            response_mask.extend([1.0 if include_segment else 0.0] * max(segment_len, 0))
            previous_prefix_ids = prefix_ids

        if len(response_mask) != len(response_ids):
            raise ValueError(
                f"Response mask/token mismatch for pipeline={pipeline_mode}: "
                f"{len(response_mask)=} vs {len(response_ids)=}"
            )

        return prompt_ids, response_ids, response_mask

    def _pad_left(self, sequences: list[list[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        ids = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row_idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            ids[row_idx, -len(seq) :] = seq_tensor
            mask[row_idx, -len(seq) :] = 1
        return ids, mask

    def _pad_right_ids(self, sequences: list[list[int]], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        ids = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row_idx, seq in enumerate(sequences):
            if not seq:
                continue
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            ids[row_idx, : len(seq)] = seq_tensor
            mask[row_idx, : len(seq)] = 1
        return ids, mask

    def _pad_right_float(self, sequences: list[list[float]]) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        out = torch.zeros((len(sequences), max_len), dtype=torch.float32)
        for row_idx, seq in enumerate(sequences):
            if not seq:
                continue
            out[row_idx, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
        return out

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
        pipeline_modes = batch.non_tensor_batch.get("pipeline_mode")
        if pipeline_modes is None:
            pipeline_modes = np.array([PIPELINE_SHARED] * len(batch), dtype=object)
        expanded_pipeline_modes = np.repeat(pipeline_modes, n)

        prompt_id_rows = []
        response_id_rows = []
        response_mask_rows = []
        for conversation, pipeline_mode in zip(batch_conversations, expanded_pipeline_modes):
            prompt_ids, response_ids, response_mask = self._encode_training_example(
                conversation=conversation,
                pipeline_mode=str(pipeline_mode),
            )
            prompt_id_rows.append(prompt_ids)
            response_id_rows.append(response_ids)
            response_mask_rows.append(response_mask)

        prompt_input_ids, prompt_attention_mask = self._pad_left(
            prompt_id_rows,
            pad_value=self.tokenizer.pad_token_id,
        )
        response_input_ids, response_attention_mask = self._pad_right_ids(
            response_id_rows,
            pad_value=self.tokenizer.pad_token_id,
        )
        response_mask = self._pad_right_float(response_mask_rows)
        reward_mask = _keep_last_token(response_mask)

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0) * attention_mask

        batch_td = TensorDict(
            {
                "prompts": prompt_input_ids,
                "responses": response_input_ids,
                "response_mask": response_mask,
                "reward_mask": reward_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(input_ids),
        )
        rewards = np.array([reward for reward in batch_reward], dtype=object)
        return DataProto(
            batch=batch_td,
            non_tensor_batch={
                "__reward__": rewards,
                "pipeline_mode": expanded_pipeline_modes,
            },
        )
