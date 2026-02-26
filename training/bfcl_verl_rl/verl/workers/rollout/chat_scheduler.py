# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import importlib
import itertools
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import uuid4
import os

import aiohttp
import numpy as np
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import deprecated
logger = logging.getLogger(__file__)


def _short_err_msg(err: Exception, max_len: int = 400) -> str:
    msg = str(err).replace("\n", " ")
    if len(msg) > max_len:
        return msg[:max_len] + "...(truncated)"
    return msg


def _is_context_overflow_error(msg: str) -> bool:
    lowered = (msg or "").lower()
    return "maximum context length" in lowered and "tokens" in lowered


def _build_empty_completion(request_id: str, model_name: str) -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{request_id}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
    )


def _estimate_chat_tokens(tokenizer, messages: List[Dict[str, Any]]) -> int:
    try:
        if tokenizer is None:
            return -1
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    except Exception:
        return -1


def _safe_message_copy(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for msg in messages:
        item = dict(msg)
        if "content" in item and item["content"] is not None and not isinstance(item["content"], str):
            item["content"] = str(item["content"])
        copied.append(item)
    return copied



class CompletionCallback(ABC):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        self.config = config
        self.scheduler = scheduler

        # Initialize tools from config file
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        #tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        #self.tools = {tool.name: tool for tool in tool_list}
        #self._tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        #print(f"Initialized tools: {self.tools}", flush=True)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

    @property
    def tool_schemas(self):
        """OpenAI JSON tool schemas."""
        return self._tool_schemas

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None

    @abstractmethod
    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        """Call back function to process completions.

        Args:
            messages: List of messages including raw prompt and assistant, tool response generated so far.
            completions: Chat completions from OpenAI compatible server.
            info: Any other auxiliary information pass across multi-turn.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask",
            "position_ids"].
        """
        raise NotImplementedError


class ToolCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)

        # TODO: add reward manager to calculate reward score once a sample finish

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        '''
        message = completions.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
        if "content" not in message:
            message["content"] = ""
        messages.append(message)
        finish_reason = completions.choices[0].finish_reason

        # STEP 0: check if we reach max turns
        if self.max_assistant_turns and len(messages) >= self.max_assistant_turns:
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")
            return

        # STEP 1: check if the model called tools
        if finish_reason != "tool_calls":
            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")
            return

        # STEP 2: call tools
        tool_calls = completions.choices[0].message.tool_calls
        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")
        tasks = []
        for tool_call in tool_calls:
            tasks.append(self._call_tool(tool_call))
        tool_responses = await asyncio.gather(*tasks)
        if any(isinstance(item, Exception) for item in tool_responses):
            print(
                f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, "
                f"done!"
            )
            return
        messages.extend(tool_responses)

        # STEP 3: resubmit completion request with tool responses
        self.scheduler.submit_chat_completions(messages=messages, tools_content = tools_content, request_id=completions.id, info=info)
        '''

    async def _call_tool(self, tool_call) -> Dict[str, str]:
        """Call tool and return tool response."""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool = self.tools[tool_name]

        instance_id = await tool.create()
        try:
            tool_response, tool_reward_score, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.exception(f"Error when executing tool: {e}")
            return e
        finally:
            await tool.release(instance_id)

        return {
            "role": "tool",
            "content": tool_response,
            "tool_call_id": tool_call.id,
        }

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts: [prompt] from input dataset
        '''
        prompts = [
            self.tokenizer.apply_chat_template(
                prompt, tools=tools_content, add_generation_prompt=True, tokenize=False
            )
            for prompt,tools_content in zip(batch.non_tensor_batch["raw_prompt"], batch.non_tensor_batch["tools_kwargs"])
        ]
        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = [
            self.tokenizer.apply_chat_template(
                conversation, tools=tools_content, add_generation_prompt=False, tokenize=False
            )
            for conversation,tools_content in zip(batch_conversations, batch.non_tensor_batch["tools_kwargs"])
        ]

        # responses: [response]
        responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]

        prompts = self.tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
        responses = self.tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if n > 1:
            prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
            prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

        # response_mask: response mask with tools calling masked out
        response_mask = self._mask_out_tools_calling_tokens(
            batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0),
            batch_conversations,
            responses["input_ids"],
            responses["attention_mask"],
        )

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32)
    
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns})
        '''
        return 

    def _mask_out_tools_calling_tokens(
        self,
        raw_prompts: List[List[Dict[str, str]]],
        batch_conversations: List[List[Dict[str, str]]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mask out tools calling tokens in the responses.

        Args:
            raw_prompts: [prompt] from input dataset
            batch_conversations: [prompt + response]
            input_ids: responses tokens
            attention_mask: responses attention mask

        Returns:
            mask: (batch_size, response_length)
        """
        batch_size = input_ids.size(0)
        assert len(raw_prompts) == batch_size, f"{len(raw_prompts)} != {batch_size}"
        assert len(batch_conversations) == batch_size, f"{len(batch_conversations)} != {batch_size}"

        # Deduplicate adjacent tool calls, since they're merged into one turn.
        # [user, assistant, tool, tool, assistant] -> [user, assistant, tool, assistant]
        # TODO: it's chat_template specific, find a more generic way to do this.
        def deduplicate_adjacent_tool_calls(roles):
            result = []
            for role, group in itertools.groupby(roles):
                if role == "tool":
                    result.append(role)
                else:
                    result.extend(group)
            return result

        loss_mask = attention_mask.clone()
        for i in range(batch_size):
            responses = batch_conversations[i][len(raw_prompts[i]) :]
            assert len(responses) > 0, f"responses is empty: {responses}"

            roles = deduplicate_adjacent_tool_calls([response["role"] for response in responses])
            # Each turn should be: [BOS]...[EOS]
            eos_indices = input_ids[i].eq(self.tokenizer.eos_token_id).nonzero().squeeze(1)[: len(roles)]
            assert len(roles) == len(eos_indices), "msimsimsimsi"
            for j in range(len(roles)):
                if roles[j] == "tool" or roles[j] == "user":
                    bos = eos_indices[j - 1] + 1 if j > 0 else 0
                    eos = eos_indices[j]
                    loss_mask[i, bos : eos + 5] = 0 # qwen2.5-7b
                else:
                    eos = eos_indices[j]
                    loss_mask[i, eos + 1 : eos + 2] = 0 #qwen2.5-7b

        return loss_mask


@deprecated("verl.experimental.agent_loop.AgentLoopManager")
class ChatCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig.
            server_addresses: List[str], OpenAI compatible server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config.actor_rollout_ref.rollout
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        self.background_tasks = set()
        if self.config.multi_turn.completion_callback is None:
            self.completion_callback = ToolCompletionCallback(config, self)
            logger.warning("completion_callback is None, use ToolCompletionCallback")
        else:
            module_path, class_name = self.config.multi_turn.completion_callback.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.completion_callback = getattr(module, class_name)(config, self)

    def _trim_messages_for_max_model_len(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Best-effort trim before request to avoid vLLM context overflow."""
        max_model_len = int(getattr(self.config, "max_model_len", 0) or 0)
        if max_model_len <= 0:
            return messages

        tokenizer = getattr(self.completion_callback, "tokenizer", None)
        if tokenizer is None:
            return messages

        trimmed = _safe_message_copy(messages)

        def _num_tokens(msgs: List[Dict[str, Any]]) -> int:
            return len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))

        # Keep system prompt; evict oldest non-system turns first.
        while len(trimmed) > 1 and _num_tokens(trimmed) > max_model_len:
            if trimmed[0].get("role") == "system":
                del trimmed[1]
            else:
                del trimmed[0]

        if _num_tokens(trimmed) <= max_model_len:
            return trimmed

        # Last-resort: truncate the final content payload.
        last = dict(trimmed[-1])
        content = last.get("content", "")
        if isinstance(content, str) and content:
            keep = max(64, len(content) // 4)
            last["content"] = content[-keep:]
            trimmed[-1] = last
        return trimmed

    def submit_chat_completions(self, *, messages: List[Dict[str, str]], request_id: str, info: Dict[str, Any], flag, reward_reference, total_messages):
        """Submit chat completion request without wait, completion_callback will be called when the request is done.

        Args:
            messages: List of messages.
            request_id: Request id.
            info: Any other auxiliary information pass across multi-turn.
        """
        info["__depth__"] += 1
        task = asyncio.create_task(self._submit_chat_completions_and_callback(messages, request_id, info, flag, reward_reference, total_messages))

        # “fire-and-forget” background tasks
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _submit_chat_completions_and_callback(
        self,
        messages: List[Dict[str, str]],
        request_id: str,
        info: Dict[str, Any],
        flag, reward_reference, total_messages
    ):
        """Submit chat completion request, wait request finish and do callback."""
        if request_id:
            request_id = request_id.removeprefix("chatcmpl-")
            assert request_id in self.request_id_to_address
            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address

        completions, exception = None, None
        messages = self._trim_messages_for_max_model_len(messages)
        tok_est = _estimate_chat_tokens(getattr(self.completion_callback, "tokenizer", None), messages)
        logger.info(
            "chat_request request_id=%s address=%s messages=%s token_estimate=%s",
            request_id,
            address,
            len(messages),
            tok_est,
        )
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(
                address,
                messages=messages,
                extra_body=self.completion_callback.extra_body,
                extra_headers={"x-request-id": request_id},
                **info["__sampling_params__"],
            )
        except Exception as e:
            err_msg = str(e)
            if _is_context_overflow_error(err_msg):
                logger.warning(
                    "context overflow at %s, fallback to empty completion. detail: %s",
                    address,
                    err_msg,
                )
                completions = _build_empty_completion(request_id=request_id, model_name=self.model_name)
            else:
                logger.error("chat completion request failed: %s", _short_err_msg(e))
                raise RuntimeError(
                    f"chat completion request failed at {address}, "
                    f"request_id={request_id}, messages={len(messages)}, token_estimate={tok_est}: {_short_err_msg(e)}"
                ) from e

        info["__depth__"] -= 1

        if exception is not None:
            logger.error("chat completion failed: %s", _short_err_msg(exception))
        else:
            try:
                await self.completion_callback(messages, completions, info, flag, reward_reference, total_messages) 
            except Exception as e:
                logger.error("completion callback failed: %s", _short_err_msg(e))
                raise

        # No more ongoing completion requests
        if info["__depth__"] == 0:
            info["__done__"].set()

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key="token-abc123", timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_body = chat_complete_request.pop("extra_body", {})
            chat_complete_request.update(extra_body or {})
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": "Bearer token-abc123", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    error_obj = data.get("error") if isinstance(data, dict) else None
                    error_msg = error_obj.get("message") if isinstance(error_obj, dict) else str(data)
                    raise RuntimeError(error_msg)
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        t_start = time.time()
        kwargs = dict(
            model=self.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            kwargs["top_p"] = self.config.val_kwargs.top_p 
            kwargs["top_k"] = self.config.val_kwargs.top_k # hansi added
            kwargs["temperature"] = self.config.val_kwargs.temperature

        print(f"[ChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # NOTE: For multi-turn rollout, repeat raw_prompt n times and process each prompt independently,
        # validation dataset has already been repeated in `PPOTrainer._validate`.
        n = 1 if batch.meta_info.get("validate", False) else self.config.n
        tasks = []
        batch_conversations = [None] * len(batch) * n
        batch_tools_content = [None] * len(batch) * n
        batch_flag = [None] * len(batch) * n
        batch_reward_reference = [None] * len(batch) * n
        batch_total_messages = [None] * len(batch) * n
        for batch_index, (conversation, tools_content, flag, reward_reference, total_messages) in enumerate( zip( batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch.non_tensor_batch["tools_kwargs"].repeat(n, axis=0), batch.non_tensor_batch["flag"].repeat(n, axis=0), batch.non_tensor_batch["reward_reference"].repeat(n, axis=0), batch.non_tensor_batch["total_messages"].repeat(n, axis=0)  ) ):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()
            batch_tools_content[batch_index] = tools_content
            batch_flag[batch_index] = flag
            batch_reward_reference[batch_index] = reward_reference
            batch_total_messages[batch_index] = total_messages

            tasks.append(
                asyncio.create_task(
                    self._submit_chat_completions_semaphore(
                        messages=batch_conversations[batch_index],
                        request_id=None,
                        sampling_params=kwargs,
                        flag=flag,
                        reward_reference=reward_reference,
                        total_messages=total_messages
                    )
                )
            )

        await asyncio.gather(*tasks)

        batch_reward = [None] * len(batch) * n
        for i, conversation in enumerate(batch_conversations):
            
            try:
                reward = conversation[-1]["reward"]
                batch_reward[i]=reward
                batch_conversations[i] = conversation[:-1]
            except Exception as e:
                raise AssertionError(
                    f"missing terminal reward for sample {i}, "
                    f"conversation_len={len(conversation)}, last_role={conversation[-1].get('role') if conversation else 'none'}"
                ) from e
            # Print per-trajectory reward for quick online debugging.
            logger.info("trajectory_reward sample=%s reward=%s", i, reward)

        # Print batch-level reward ratio (final-step reward > 0 means success).
        try:
            success_cnt = sum(1 for r in batch_reward if isinstance(r, (list, tuple)) and len(r) > 0 and float(r[-1]) > 0.0)
            total_cnt = len(batch_reward)
            success_ratio = (success_cnt / total_cnt) if total_cnt > 0 else 0.0
            logger.info(
                "trajectory_reward_ratio success=%s total=%s ratio=%.4f",
                success_cnt,
                total_cnt,
                success_ratio,
            )
        except Exception as ratio_err:
            logger.warning("failed to compute trajectory reward ratio: %s", ratio_err)

        def _reserve_next_dump_path(dump_dir: str, prefix: str = "batch_", suffix: str = ".json"):
            """
            返回一个未被占用的新文件路径，并以独占方式打开文件句柄（避免并发冲突）。
            规则：扫描 dump_dir 下 prefix+数字+suffix 的文件，选取最大数字+1；若并发冲突则递增重试。
            """
            os.makedirs(dump_dir, exist_ok=True)
            # 找到当前最大 k
            k = 0
            try:
                for name in os.listdir(dump_dir):
                    if name.startswith(prefix) and name.endswith(suffix):
                        num_part = name[len(prefix):-len(suffix)]
                        try:
                            k = max(k, int(num_part))
                        except Exception:
                            pass
            except FileNotFoundError:
                pass

            # 独占创建，防并发竞态
            while True:
                k += 1
                path = os.path.join(dump_dir, f"{prefix}{k}{suffix}")
                try:
                    f = open(path, "x", encoding="utf-8")  # 独占创建
                    return path, f
                except FileExistsError:
                    continue

        # === 插入点 A：保存所有 batch 元素到文件（从现有最大 k 的下一个开始） ===
        try:
            dump_dir = getattr(self.config, "dump_dir", None) or "./batch_dumps"
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # 汇总本次 batch 的所有 record
            total = len(batch) * n 
            path, fh = _reserve_next_dump_path(dump_dir, prefix="batch_", suffix=".txt")
            try:
                for i in range(total):
                    record = {
                        "index_in_batch": i,
                        "conversation": batch_conversations[i],
                        "tools_content": batch_tools_content[i],
                        "flag": batch_flag[i],
                        "total_messages": (
                            [json.loads(x) if isinstance(x, str) else x for x in batch_total_messages[i]]
                            if isinstance(batch_total_messages[i], (list, tuple))
                            else (json.loads(batch_total_messages[i]) if isinstance(batch_total_messages[i], str) else batch_total_messages[i])
                        ),
                        "reward": batch_reward[i],
                    }
                    fh.write(json.dumps(record, ensure_ascii=False))
                    fh.write("\n")
            finally:
                fh.close()
            print(f"[ChatCompletionScheduler] dumped batch to {path} (records={total})")
        except Exception as dump_err:
            print(f"[ChatCompletionScheduler] dump batch failed: {dump_err}")

        # === 保存结束 ===

        output_batch = self.completion_callback.postprocess(batch, batch_conversations, batch_tools_content, batch_flag, batch_reward_reference, batch_total_messages, batch_reward, n=n)
        output_batch.meta_info["timing"] = {"generate_sequences": time.time() - t_start}
        print("[ChatCompletionScheduler] generate_sequences done")
        return output_batch

    async def _submit_chat_completions_semaphore(
        self, messages: List[Dict[str, str]], request_id: str, sampling_params: Dict[str, Any],flag,reward_reference,total_messages
    ):
        done = asyncio.Event()

        info = {
            "__done__": done,
            "__depth__": 0,  # indicate how many ongoing completion requests
            "__sampling_params__": sampling_params,
        }

        self.submit_chat_completions(messages=messages, request_id=request_id, info=info, flag=flag,reward_reference=reward_reference,total_messages=total_messages)

        # Wait until all completion requests are done
        await done.wait()
