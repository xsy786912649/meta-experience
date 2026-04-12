import asyncio
import copy
import hashlib
import json
import re
import uuid
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback

from recipe.bfcl_meta.meta_env import build_summary_prompt_messages
from recipe.bfcl_meta.meta_rollout import MetaRolloutEngine, parse_summary_generation

PIPELINE_SUMMARY = "summary_only"
PIPELINE_SUPPORT = "support_only"
PIPELINE_PHASED = "phased"


def _has_explicit_action_output(text: str) -> bool:
    if not text:
        return False
    final_text = text.split("</think>")[-1] if "</think>" in text else text
    return bool(re.search(r"<tool_call>\s*.*?\s*</tool_call>", final_text, re.DOTALL))


def _build_reward_payload(
    reward: float,
    *,
    query_success: bool,
    query_ran: bool,
    explicit_action_output: bool,
    used_memo: bool,
    summary_context_text: str,
) -> dict[str, Any]:
    return {
        "reward": [float(reward)],
        "query_success": bool(query_success),
        "query_ran": bool(query_ran),
        "explicit_action_output": bool(explicit_action_output),
        "used_memo": bool(used_memo),
        "summary_context_nonempty": bool((summary_context_text or "").strip()),
    }


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
        self.train_pipeline_mode = str(
            config.actor_rollout_ref.rollout.multi_turn.get("pipeline_mode", PIPELINE_PHASED)
        )
        self.val_pipeline_mode = str(
            config.actor_rollout_ref.rollout.multi_turn.get("val_pipeline_mode", PIPELINE_SUMMARY)
        )
        self.summary_phase_epochs = int(
            config.actor_rollout_ref.rollout.multi_turn.get("summary_phase_epochs", 3)
        )
        self.support_phase_epochs = int(
            config.actor_rollout_ref.rollout.multi_turn.get("support_phase_epochs", 1)
        )
        self.disable_query_memo = bool(
            config.actor_rollout_ref.rollout.multi_turn.get("disable_query_memo", False)
        )
        self.support_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("support_temperature", 1.0)
        )
        self.query_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("query_temperature", 0.0)
        )
        self._prepared_samples: dict[str, dict[str, Any]] = {}
        self._prepared_requests: dict[str, list[dict[str, Any]]] = {}
        self._prepared_request_lock = asyncio.Lock()
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
        if requested == PIPELINE_PHASED:
            summary_epochs = max(self.summary_phase_epochs, 0)
            support_epochs = max(self.support_phase_epochs, 0)
            cycle_len = summary_epochs + support_epochs
            if cycle_len <= 0:
                return PIPELINE_SUMMARY
            current_epoch = int(getattr(self, "_current_train_epoch", 0))
            cycle_epoch = current_epoch % cycle_len
            return PIPELINE_SUMMARY if cycle_epoch < summary_epochs else PIPELINE_SUPPORT
        if requested in {PIPELINE_SUMMARY, PIPELINE_SUPPORT}:
            return requested
        raise ValueError(f"Unsupported bfcl_meta pipeline mode: {requested}")

    @staticmethod
    def _conversation_key(messages: list[dict[str, str]]) -> str:
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    async def _register_prepared_request(self, messages: list[dict[str, str]], metadata: dict[str, Any]) -> None:
        key = self._conversation_key(messages)
        async with self._prepared_request_lock:
            self._prepared_requests.setdefault(key, []).append(metadata)

    async def _consume_prepared_request(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        key = self._conversation_key(messages)
        async with self._prepared_request_lock:
            queue = self._prepared_requests.get(key)
            if not queue:
                return None
            metadata = queue.pop(0)
            if not queue:
                self._prepared_requests.pop(key, None)
            return metadata

    async def prepare_batch_conversations(self, batch: DataProto, n: int) -> list[list[dict[str, str]]]:
        self._current_train_epoch = int(batch.meta_info.get("train_epoch", 0))
        batch_mode = self._select_batch_mode(validate=bool(batch.meta_info.get("validate", False)))
        batch.non_tensor_batch["pipeline_mode"] = np.array([batch_mode] * len(batch), dtype=object)

        prepared = []
        payloads = [
            payload if isinstance(payload, dict) else json.loads(payload)
            for payload in batch.non_tensor_batch["total_messages"]
        ]

        if batch_mode == PIPELINE_SUMMARY:
            results = await asyncio.gather(*(self._prepare_summary_payload(payload, repeat_count=n) for payload in payloads))
        else:
            results = await asyncio.gather(*(self._prepare_support_payload(payload, repeat_count=n) for payload in payloads))

        for result in results:
            self._prepared_samples[result["meta_instance_id"]] = {"remaining": result["remaining"]}
            prepared.extend(result["conversations"])
        return prepared

    async def _prepare_summary_payload(self, payload: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        support_result = await self.rollout_engine.run_task_rollout(
            payload=payload["support"],
            temperature=self.support_temperature,
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
        conversations = []
        for _ in range(repeat_count):
            prepared_conversation = copy.deepcopy(summary_messages)
            await self._register_prepared_request(
                prepared_conversation,
                {
                    "pipeline_mode": PIPELINE_SUMMARY,
                    "meta_instance_id": payload["meta_instance_id"],
                },
            )
            conversations.append(prepared_conversation)
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "conversations": conversations,
            "remaining": repeat_count,
        }

    async def _prepare_support_payload(self, payload: dict[str, Any], repeat_count: int) -> dict[str, Any]:
        support_results = await asyncio.gather(
            *(
                self.rollout_engine.run_task_rollout(
                    payload=payload["support"],
                    temperature=self.support_temperature,
                    experience_summary=None,
                    max_model_len=self.support_max_model_len,
                    exploration_mode=True,
                )
                for _ in range(repeat_count)
            )
        )
        conversations = [
            build_summary_prompt_messages(
                tokenizer=self.tokenizer,
                trajectory_text=result["trajectory_text"],
                support_success=result["success"],
                checker=result["checker"],
                max_prompt_tokens=self.summary_prompt_budget,
            )
            for result in support_results
        ]
        for prepared_conversation, support_result in zip(conversations, support_results):
            await self._register_prepared_request(
                prepared_conversation,
                {
                    "pipeline_mode": PIPELINE_SUPPORT,
                    "meta_instance_id": payload["meta_instance_id"],
                    "support_conversation": copy.deepcopy(support_result["conversation"]),
                },
            )
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
        request_meta = await self._consume_prepared_request(messages)
        if request_meta and request_meta.get("pipeline_mode") == PIPELINE_SUPPORT:
            messages[:] = copy.deepcopy(request_meta["support_conversation"])
        messages.append({"role": "assistant", "content": ""})
        messages.append(
            {
                "reward": _build_reward_payload(
                    -0.001,
                    query_success=False,
                    query_ran=False,
                    explicit_action_output=False,
                    used_memo=False,
                    summary_context_text="",
                )
            }
        )
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
        request_meta = await self._consume_prepared_request(messages)
        pipeline_mode = request_meta["pipeline_mode"] if request_meta else PIPELINE_SUMMARY

        full_output, summary_context_text = parse_summary_generation(completions)

        explicit_action_output = _has_explicit_action_output(full_output)
        used_memo = not self.disable_query_memo and bool(summary_context_text)
        query_result = await self.rollout_engine.run_task_rollout(
            payload=payload["query"],
            temperature=self.query_temperature,
            experience_summary=None if self.disable_query_memo else (summary_context_text or None),
        )
        reward = 1.0 if query_result["success"] else -0.001
        if explicit_action_output:
            reward = -0.5

        if pipeline_mode == PIPELINE_SUPPORT and request_meta is not None:
            messages[:] = copy.deepcopy(request_meta["support_conversation"])
        else:
            messages.append({"role": "assistant", "content": full_output})
        messages.append(
            {
                "reward": _build_reward_payload(
                    reward,
                    query_success=bool(query_result["success"]),
                    query_ran=True,
                    explicit_action_output=explicit_action_output,
                    used_memo=used_memo,
                    summary_context_text=summary_context_text,
                )
            }
        )

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
        if pipeline_mode == PIPELINE_SUMMARY:
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
        response_ids: list[int] = []
        response_mask = []
        for message_idx in range(prompt_message_count, len(conversation)):
            prev_messages = conversation[:message_idx]
            cur_messages = conversation[: message_idx + 1]
            role = conversation[message_idx].get("role")
            include_assistant_content = role == "assistant" and (
                pipeline_mode == PIPELINE_SUPPORT or message_idx == len(conversation) - 1
            )

            prev_text = self.tokenizer.apply_chat_template(
                prev_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            cur_text = self.tokenizer.apply_chat_template(
                cur_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if role == "assistant":
                prev_text_with_generation_prompt = self.tokenizer.apply_chat_template(
                    prev_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                if message_idx == prompt_message_count:
                    segment_text = cur_text[len(prev_text_with_generation_prompt) :]
                    segment_tokens = self.tokenizer.encode(segment_text, add_special_tokens=False)
                    response_ids.extend(segment_tokens)
                    response_mask.extend([1.0 if include_assistant_content else 0.0] * len(segment_tokens))
                else:
                    generation_prompt_text = prev_text_with_generation_prompt[len(prev_text) :]
                    generation_prompt_tokens = self.tokenizer.encode(
                        generation_prompt_text,
                        add_special_tokens=False,
                    )
                    message_tokens = self.tokenizer.encode(
                        cur_text[len(prev_text_with_generation_prompt) :],
                        add_special_tokens=False,
                    )
                    response_ids.extend(generation_prompt_tokens)
                    response_mask.extend([0.0] * len(generation_prompt_tokens))
                    response_ids.extend(message_tokens)
                    response_mask.extend([1.0 if include_assistant_content else 0.0] * len(message_tokens))
            else:
                segment_text = cur_text[len(prev_text) :]
                segment_tokens = self.tokenizer.encode(segment_text, add_special_tokens=False)
                response_ids.extend(segment_tokens)
                response_mask.extend([0.0] * len(segment_tokens))

        if len(response_mask) != len(response_ids):
            raise ValueError(
                f"Response mask/token mismatch for pipeline={pipeline_mode}: "
                f"{len(response_mask)=} vs {len(response_ids)=}, "
                f"{prompt_message_count=}, {len(conversation)=}"
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
            pipeline_modes = np.array([PIPELINE_SUMMARY] * len(batch), dtype=object)
        expanded_pipeline_modes = np.repeat(pipeline_modes, n)

        batch_mode = str(expanded_pipeline_modes[0]) if len(expanded_pipeline_modes) > 0 else PIPELINE_SUMMARY
        if batch_mode == PIPELINE_SUMMARY:
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
            reward_mask = _keep_last_token(response_mask)
            input_ids = torch.cat([prompt_tensors["input_ids"], response_tensors["input_ids"]], dim=1)
            attention_mask = torch.cat(
                [prompt_tensors["attention_mask"], response_tensors["attention_mask"]],
                dim=1,
            )
            position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0) * attention_mask

            batch_td = TensorDict(
                {
                    "prompts": prompt_tensors["input_ids"],
                    "responses": response_tensors["input_ids"],
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
