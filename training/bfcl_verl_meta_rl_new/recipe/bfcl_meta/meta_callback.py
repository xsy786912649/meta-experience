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

from recipe.bfcl_meta.meta_env import build_summary_prompt_messages, build_task_system_prompt
from recipe.bfcl_meta.meta_rollout import MetaRolloutEngine, parse_summary_generation

PIPELINE_SUMMARY = "summary_only"
PIPELINE_SUPPORT = "support_only"
PIPELINE_QUERY = "query_only"
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
        "query_success_count": 1 if query_success else 0,
        "query_ran_count": 1 if query_ran else 0,
        "query_total": 1,
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
        self.train_summary_assistant = bool(
            config.actor_rollout_ref.rollout.multi_turn.get("train_summary_assistant", True)
        )
        self.support_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("support_temperature", 1.0)
        )
        self.query_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("query_temperature", 1.0)
        )
        self.summary_temperature = float(config.actor_rollout_ref.rollout.get("temperature", 1.0))
        self.val_support_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("val_support_temperature", 0.0)
        )
        self.val_query_temperature = float(
            config.actor_rollout_ref.rollout.multi_turn.get("val_query_temperature", 0.0)
        )
        self.val_summary_temperature = float(config.actor_rollout_ref.rollout.val_kwargs.get("temperature", 0.0))
        self._prepared_samples: dict[str, dict[str, Any]] = {}
        self._prepared_requests: dict[str, list[dict[str, Any]]] = {}
        self._query_training_examples: dict[str, list[dict[str, Any]]] = {}
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

    def _log_request_error_context(
        self,
        *,
        payload: dict[str, Any],
        request_meta: dict[str, Any] | None,
        messages: list[dict[str, str]],
        error_type: str,
        error_message: str,
    ) -> None:
        try:
            pipeline_mode = request_meta["pipeline_mode"] if request_meta else PIPELINE_SUMMARY
            local_tokens = len(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            role_summaries = [
                f"{message.get('role', 'unknown')}:{len(str(message.get('content', '')))}"
                for message in messages
            ]
            last_content = str(messages[-1].get("content", "")) if messages else ""
            print(
                "[bfcl_meta_request_error] "
                f"meta_instance_id={payload.get('meta_instance_id', 'unknown')} "
                f"pipeline_mode={pipeline_mode} "
                f"error_type={error_type} "
                f"local_tokens={local_tokens} "
                f"max_model_len={self.max_model_len} "
                f"message_count={len(messages)} "
                f"roles={role_summaries} "
                f"last_content_tail={last_content[-300:]!r} "
                f"error={error_message}"
            )
        except Exception as log_exc:
            print(
                "[bfcl_meta_request_error_logging_failed] "
                f"meta_instance_id={payload.get('meta_instance_id', 'unknown')} "
                f"error_type={error_type} "
                f"log_error={log_exc!r}"
            )

    @staticmethod
    def _format_conversation_as_trajectory(conversation: list[dict[str, str]]) -> str:
        return "".join(
            f"<|im_start|>{message.get('role', 'user')}\n{message.get('content', '')}<|im_end|>\n"
            for message in conversation
        )

    def _query_example_key(self, meta_instance_id: str, prompt_messages: list[dict[str, str]]) -> str:
        return f"{meta_instance_id}:{self._conversation_key(prompt_messages)}"

    def _store_query_training_example(
        self,
        meta_instance_id: str,
        prompt_messages: list[dict[str, str]],
        conversation: list[dict[str, str]],
        reward_payload: dict[str, Any],
    ) -> None:
        key = self._query_example_key(meta_instance_id, prompt_messages)
        self._query_training_examples.setdefault(key, []).append(
            {
                "conversation": copy.deepcopy(conversation),
                "reward": copy.deepcopy(reward_payload),
            }
        )

    def _consume_query_training_example(
        self,
        meta_instance_id: str,
        prompt_messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        key = self._query_example_key(meta_instance_id, prompt_messages)
        queue = self._query_training_examples.get(key)
        if not queue:
            return None
        example = queue.pop(0)
        if not queue:
            self._query_training_examples.pop(key, None)
        return example

    @staticmethod
    def _summary_prompt_messages_from_completed_conversation(
        conversation: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        messages = conversation
        if messages and "reward" in messages[-1] and "role" not in messages[-1]:
            messages = messages[:-1]
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return messages

    def _build_minimal_support_conversation(self, support_payload: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = build_task_system_prompt(
            support_payload["function"],
            experience_summary=None,
            exploration_mode=True,
        )
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.append({"role": "assistant", "content": "\n"})
        return conversation

    @staticmethod
    def _get_support_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
        supports = payload.get("supports")
        if isinstance(supports, list) and supports:
            return supports
        return [payload["support"]]

    def _build_minimal_query_conversation(
        self,
        query_payload: dict[str, Any],
        experience_summary: str | None,
    ) -> list[dict[str, str]]:
        system_prompt = build_task_system_prompt(
            query_payload["function"],
            experience_summary=experience_summary,
            exploration_mode=False,
        )
        conversation = [{"role": "system", "content": system_prompt}]
        conversation.append({"role": "assistant", "content": "\n"})
        return conversation

    @staticmethod
    def _get_query_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
        queries = payload.get("queries")
        if isinstance(queries, list) and queries:
            return queries
        return [payload["query"]]

    def _build_support_exception_result(self, support_payload: dict[str, Any], error_message: str) -> dict[str, Any]:
        conversation = self._build_minimal_support_conversation(support_payload)
        return {
            "success": False,
            "checker": {
                "valid": False,
                "error_type": "support_rollout_exception",
                "details": {"error": str(error_message)[:600]},
            },
            "trajectory_text": self._format_conversation_as_trajectory(conversation),
            "conversation": conversation,
        }

    def _normalize_support_result(self, support_payload: dict[str, Any], support_result: dict[str, Any]) -> dict[str, Any]:
        conversation = support_result.get("conversation") or []
        if any(message.get("role") == "assistant" for message in conversation):
            return support_result

        normalized = dict(support_result)
        minimal_conversation = self._build_minimal_support_conversation(support_payload)
        normalized["conversation"] = minimal_conversation
        normalized["trajectory_text"] = self._format_conversation_as_trajectory(minimal_conversation)
        return normalized

    def _build_query_exception_result(
        self,
        query_payload: dict[str, Any],
        experience_summary: str | None,
        error_message: str,
    ) -> dict[str, Any]:
        conversation = self._build_minimal_query_conversation(query_payload, experience_summary)
        return {
            "success": False,
            "checker": {
                "valid": False,
                "error_type": "query_rollout_exception",
                "details": {"error": str(error_message)[:600]},
            },
            "trajectory_text": self._format_conversation_as_trajectory(conversation),
            "conversation": conversation,
        }

    def _normalize_query_result(
        self,
        query_payload: dict[str, Any],
        experience_summary: str | None,
        query_result: dict[str, Any],
    ) -> dict[str, Any]:
        conversation = query_result.get("conversation") or []
        if any(message.get("role") == "assistant" for message in conversation):
            return query_result

        normalized = dict(query_result)
        minimal_conversation = self._build_minimal_query_conversation(query_payload, experience_summary)
        normalized["conversation"] = minimal_conversation
        normalized["trajectory_text"] = self._format_conversation_as_trajectory(minimal_conversation)
        return normalized

    async def _run_support_rollout_safe(self, support_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            support_result = await self.rollout_engine.run_task_rollout(
                payload=support_payload,
                temperature=self.support_temperature,
                experience_summary=None,
                max_model_len=self.support_max_model_len,
                exploration_mode=True,
                preserve_truncated_assistant=True,
            )
            return self._normalize_support_result(support_payload, support_result)
        except Exception as exc:
            return self._build_support_exception_result(support_payload, str(exc))

    async def _run_query_rollout_safe(
        self,
        query_payload: dict[str, Any],
        experience_summary: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        try:
            query_result = await self.rollout_engine.run_task_rollout(
                payload=query_payload,
                temperature=temperature,
                experience_summary=experience_summary,
                preserve_truncated_assistant=False,
            )
            return self._normalize_query_result(query_payload, experience_summary, query_result)
        except Exception as exc:
            return self._build_query_exception_result(query_payload, experience_summary, str(exc))

    async def _run_query_rollouts_aggregate(
        self,
        query_payloads: list[dict[str, Any]],
        experience_summary: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        query_results = [
            await self._run_query_rollout_safe(query_payload, experience_summary, temperature)
            for query_payload in query_payloads
        ]
        query_success_count = sum(bool(result["success"]) for result in query_results)
        query_ran_count = len(query_results)
        query_total = len(query_results)
        query_success_rate = (query_success_count / query_total) if query_total else 0.0
        return {
            "results": query_results,
            "query_success_count": query_success_count,
            "query_ran_count": query_ran_count,
            "query_total": query_total,
            "query_success_rate": query_success_rate,
        }

    def _select_support_temperature(self, validate: bool) -> float:
        return self.val_support_temperature if validate else self.support_temperature

    def _select_query_temperature(self, validate: bool) -> float:
        return self.val_query_temperature if validate else self.query_temperature

    def _select_summary_temperature(self, validate: bool) -> float:
        return self.val_summary_temperature if validate else self.summary_temperature

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

        validate = bool(batch.meta_info.get("validate", False))
        if batch_mode == PIPELINE_SUMMARY:
            results = await asyncio.gather(
                *(self._prepare_summary_payload(payload, repeat_count=n, validate=validate) for payload in payloads)
            )
        else:
            results = await asyncio.gather(
                *(self._prepare_support_payload(payload, repeat_count=n, validate=validate) for payload in payloads)
            )

        for result in results:
            self._prepared_samples[result["meta_instance_id"]] = {"remaining": result["remaining"]}
            prepared.extend(result["conversations"])
        return prepared

    async def _prepare_summary_payload(self, payload: dict[str, Any], repeat_count: int, validate: bool) -> dict[str, Any]:
        support_payloads = self._get_support_payloads(payload)
        support_result = await self._run_support_rollout_safe_with_temperature(
            support_payloads[0],
            self._select_support_temperature(validate),
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
                    "validate": validate,
                    "multi_support_eval": bool(validate and len(support_payloads) > 1),
                    "extra_support_payloads": copy.deepcopy(support_payloads[1:]),
                },
            )
            conversations.append(prepared_conversation)
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "conversations": conversations,
            "remaining": repeat_count,
        }

    async def _prepare_support_payload(self, payload: dict[str, Any], repeat_count: int, validate: bool) -> dict[str, Any]:
        support_results = await asyncio.gather(
            *(
                self._run_support_rollout_safe_with_temperature(
                    payload["support"],
                    self._select_support_temperature(validate),
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
                    "validate": validate,
                },
            )
        return {
            "meta_instance_id": payload["meta_instance_id"],
            "conversations": conversations,
            "remaining": repeat_count,
        }

    async def _run_support_rollout_safe_with_temperature(
        self,
        support_payload: dict[str, Any],
        temperature: float,
    ) -> dict[str, Any]:
        try:
            support_result = await self.rollout_engine.run_task_rollout(
                payload=support_payload,
                temperature=temperature,
                experience_summary=None,
                max_model_len=self.support_max_model_len,
                exploration_mode=True,
                preserve_truncated_assistant=True,
            )
            return self._normalize_support_result(support_payload, support_result)
        except Exception as exc:
            return self._build_support_exception_result(support_payload, str(exc))

    async def _run_summary_generation_safe(
        self,
        summary_messages: list[dict[str, str]],
        temperature: float,
    ) -> tuple[str, bool]:
        try:
            completion = await self._chat_once(summary_messages, temperature)
            full_output, _ = parse_summary_generation(completion)
            return full_output or "empty", True
        except Exception:
            return "empty", False

    @staticmethod
    def _combine_summary_outputs(summary_outputs: list[str]) -> str:
        cleaned = [(text or "").strip() for text in summary_outputs]
        return "\n\n".join(
            f"Memo {idx + 1}:\n{text if text else 'empty'}"
            for idx, text in enumerate(cleaned)
        )

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
        self._log_request_error_context(
            payload=payload,
            request_meta=request_meta,
            messages=messages,
            error_type=error_type,
            error_message=error_message,
        )
        if request_meta and request_meta.get("pipeline_mode") == PIPELINE_SUPPORT:
            messages[:] = copy.deepcopy(request_meta["support_conversation"])
        if self.disable_query_memo:
            query_summary_text = None
        else:
            query_summary_text = "empty"
        used_memo = query_summary_text is not None
        validate = bool(request_meta.get("validate", False)) if request_meta else False
        query_aggregate = await self._run_query_rollouts_aggregate(
            self._get_query_payloads(payload),
            query_summary_text,
            self._select_query_temperature(validate),
        )
        query_success_count = query_aggregate["query_success_count"]
        query_ran_count = query_aggregate["query_ran_count"]
        query_total = query_aggregate["query_total"]
        query_success_rate = query_aggregate["query_success_rate"]
        query_success = query_success_count == query_total and query_total > 0
        query_ran = query_ran_count > 0
        query_reward_payload = _build_reward_payload(
            query_success_rate if query_total > 0 else -0.001,
            query_success=query_success,
            query_ran=query_ran,
            explicit_action_output=False,
            used_memo=used_memo,
            summary_context_text=query_summary_text or "",
        )
        query_reward_payload["query_success_count"] = query_success_count
        query_reward_payload["query_ran_count"] = query_ran_count
        query_reward_payload["query_total"] = query_total
        query_reward_payload["query_success_rate"] = query_success_rate
        if request_meta and request_meta.get("pipeline_mode") == PIPELINE_SUMMARY:
            query_payloads = self._get_query_payloads(payload)
            if len(query_payloads) == 1:
                self._store_query_training_example(
                    payload["meta_instance_id"],
                    messages,
                    query_aggregate["results"][0]["conversation"],
                    query_reward_payload,
                )
        messages.append({"role": "assistant", "content": "\n"})
        messages.append(
            {
                "reward": {
                    **_build_reward_payload(
                    query_success_rate if query_total > 0 else -0.001,
                    query_success=query_success,
                    query_ran=query_ran,
                    explicit_action_output=False,
                    used_memo=used_memo,
                    summary_context_text=query_summary_text or "",
                    ),
                    "query_success_count": query_success_count,
                    "query_ran_count": query_ran_count,
                    "query_total": query_total,
                    "query_success_rate": query_success_rate,
                }
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
        validate = bool(request_meta.get("validate", False)) if request_meta else False

        full_output, summary_context_text = parse_summary_generation(completions)
        summary_outputs = [full_output or "empty"]

        if request_meta and request_meta.get("multi_support_eval"):
            extra_support_payloads = request_meta.get("extra_support_payloads", [])
            support_temperature = self._select_support_temperature(validate)
            summary_temperature = self._select_summary_temperature(validate)
            for extra_support_payload in extra_support_payloads:
                extra_support_result = await self._run_support_rollout_safe_with_temperature(
                    extra_support_payload,
                    support_temperature,
                )
                extra_summary_messages = build_summary_prompt_messages(
                    tokenizer=self.tokenizer,
                    trajectory_text=extra_support_result["trajectory_text"],
                    support_success=extra_support_result["success"],
                    checker=extra_support_result["checker"],
                    max_prompt_tokens=self.summary_prompt_budget,
                )
                extra_full_output, _ = await self._run_summary_generation_safe(
                    extra_summary_messages,
                    summary_temperature,
                )
                summary_outputs.append(extra_full_output)

        combined_summary_text = (
            self._combine_summary_outputs(summary_outputs)
            if request_meta and request_meta.get("multi_support_eval")
            else (full_output or "empty")
        )
        explicit_action_output = any(_has_explicit_action_output(text) for text in summary_outputs)
        if self.disable_query_memo:
            query_summary_text = None
        else:
            query_summary_text = combined_summary_text if combined_summary_text else "empty"
        used_memo = query_summary_text is not None
        query_aggregate = await self._run_query_rollouts_aggregate(
            self._get_query_payloads(payload),
            query_summary_text,
            self._select_query_temperature(validate),
        )
        query_success_count = query_aggregate["query_success_count"]
        query_ran_count = query_aggregate["query_ran_count"]
        query_total = query_aggregate["query_total"]
        query_success_rate = query_aggregate["query_success_rate"]
        query_success = query_success_count == query_total and query_total > 0
        query_reward = query_success_rate if query_total > 0 else -0.001
        reward = query_reward
        if explicit_action_output:
            reward = -0.5
        query_reward_payload = _build_reward_payload(
            query_reward,
            query_success=query_success,
            query_ran=query_ran_count > 0,
            explicit_action_output=False,
            used_memo=used_memo,
            summary_context_text=query_summary_text or "",
        )
        query_reward_payload["query_success_count"] = query_success_count
        query_reward_payload["query_ran_count"] = query_ran_count
        query_reward_payload["query_total"] = query_total
        query_reward_payload["query_success_rate"] = query_success_rate
        if pipeline_mode == PIPELINE_SUMMARY:
            query_payloads = self._get_query_payloads(payload)
            if len(query_payloads) == 1:
                self._store_query_training_example(
                    payload["meta_instance_id"],
                    messages,
                    query_aggregate["results"][0]["conversation"],
                    query_reward_payload,
                )

        if pipeline_mode == PIPELINE_SUPPORT and request_meta is not None:
            messages[:] = copy.deepcopy(request_meta["support_conversation"])
        else:
            messages.append({"role": "assistant", "content": combined_summary_text})
        messages.append(
            {
                "reward": {
                    **_build_reward_payload(
                        reward,
                        query_success=query_success,
                        query_ran=query_ran_count > 0,
                        explicit_action_output=explicit_action_output,
                        used_memo=used_memo,
                        summary_context_text=query_summary_text or "",
                    ),
                    "query_success_count": query_success_count,
                    "query_ran_count": query_ran_count,
                    "query_total": query_total,
                    "query_success_rate": query_success_rate,
                }
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
                pipeline_mode in {PIPELINE_SUPPORT, PIPELINE_QUERY} or message_idx == len(conversation) - 1
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
            expanded_conversations = []
            expanded_rewards = []
            expanded_modes = []
            expanded_uids = []

            for conversation, reward, total_messages in zip(batch_conversations, batch_reward, batch_total_messages):
                payload = total_messages if isinstance(total_messages, dict) else json.loads(total_messages)
                meta_instance_id = payload["meta_instance_id"]

                if self.train_summary_assistant:
                    expanded_conversations.append(conversation)
                    expanded_rewards.append(reward)
                    expanded_modes.append(PIPELINE_SUMMARY)
                    expanded_uids.append(f"{meta_instance_id}::summary")

                summary_prompt_messages = self._summary_prompt_messages_from_completed_conversation(conversation)
                query_example = self._consume_query_training_example(meta_instance_id, summary_prompt_messages)
                if query_example is None:
                    fallback_summary = None if self.disable_query_memo else "empty"
                    fallback_query_payload = self._get_query_payloads(payload)[0]
                    query_conversation = self._build_minimal_query_conversation(
                        fallback_query_payload,
                        fallback_summary,
                    )
                    query_reward = _build_reward_payload(
                        -0.001,
                        query_success=False,
                        query_ran=False,
                        explicit_action_output=False,
                        used_memo=fallback_summary is not None,
                        summary_context_text=fallback_summary or "",
                    )
                    query_total = len(self._get_query_payloads(payload))
                    query_reward["query_success_count"] = 0
                    query_reward["query_ran_count"] = 0
                    query_reward["query_total"] = query_total
                    query_reward["query_success_rate"] = 0.0
                else:
                    query_conversation = query_example["conversation"]
                    query_reward = query_example["reward"]

                expanded_conversations.append(query_conversation)
                expanded_rewards.append(query_reward)
                expanded_modes.append(PIPELINE_QUERY)
                expanded_uids.append(f"{meta_instance_id}::query")

            batch_conversations = expanded_conversations
            batch_reward = expanded_rewards
            expanded_pipeline_modes = np.array(expanded_modes, dtype=object)
            expanded_uids = np.array(expanded_uids, dtype=object)
        else:
            expanded_uids = None

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
                **({"uid": expanded_uids} if expanded_uids is not None else {}),
            },
        )
