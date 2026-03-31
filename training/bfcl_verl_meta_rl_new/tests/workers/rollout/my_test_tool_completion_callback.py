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
import concurrent.futures
import os
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
import json
import random

import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse
import pandas as pd
from tensordict import TensorDict

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler, ToolCompletionCallback
from tests.workers.rollout.handle_message_tool_calling import extract_json_objects, convert_chat_completion_message_to_dict, pad_messages, build_tools_prompt_think, build_tools_prompt_no_think, build_tools_prompt_think_my, build_tools_prompt_no_think_my
import torch
from tests.workers.rollout.handle_message_tool_calling import generate_context1_turn1, generate_context1_turn2, generate_context2_turn1, generate_context2_turn2, generate_context2_turn3, generate_context5, generate_context4, generate_context6_turn1, generate_context6_turn2, generate_context7_turn1, generate_context7_turn2, generate_context7_turn3

torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(linewidth=200)

def _get_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]

def keep_last_one_rows(X):
    X = X.clone()
    X = np.asarray(X)
    assert np.all((X == 0) | (X == 1)), "input should be 0 or 1"
    next_col = np.concatenate([X[:, 1:], np.zeros((X.shape[0], 1), dtype=X.dtype)], axis=1)
    result = ((X == 1) & (next_col == 0)).astype(X.dtype)
    return torch.tensor(result, dtype=torch.float32)

class Qwen3CustomToolCompletionCallback(ToolCompletionCallback):
    def __init__(self, config: DictConfig, scheduler: ChatCompletionScheduler):
        super().__init__(config, scheduler)

        self.max_assistant_turns = config.actor_rollout_ref.rollout.max_assistant_turns
    

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any], flag, reward_reference, total_messages):
        message_current = completions.choices[0].message

        role, content, finish_reason = (
            completions.choices[0].message.role,
            completions.choices[0].message.content,
            completions.choices[0].finish_reason
        )

        if "</tool_call>" in message_current.content:
            last_close = message_current.content.rfind("</tool_call>")
            message_current.content = message_current.content[:last_close + len("</tool_call>")]
            message_current.content=message_current.content.replace("\n\n", "")
        else:
            message_current.content = message_current.content.strip()
            message_current.content = message_current.content.replace("\n\n", "")

        if "</tool_call>" in content:
            last_close = content.rfind("</tool_call>")
            content = content[:last_close + len("</tool_call>")]
            content = content.replace("\n\n", "")
        else:
            content = content.strip()
            content = content.replace("\n\n", "")
            
        messages.append(convert_chat_completion_message_to_dict(message_current))
        turn = len(messages)

        #print("------------------start---------------------------")
        #print(messages)
        #print("---------------------end---------------------")
        
        #print("turn:", turn)
        # STEP 0: check if we reach max turns
        if len(messages) >= self.max_assistant_turns:
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}], flag={flag}, Reach max turns, done!")
            messages.append({"reward" : [-0.0, -0.0, -0.0]})
            return

        # STEP 1: check if we reach max tokens
        #if finish_reason == "length":
        #    print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}], flag={flag}, Reach max length, done!")
        #    messages.append({"reward" : [-0.0, -0.0, -0.0]})
        #    return

        # ------------------------------ start to modify the content -------------------------------

        user_number = 0
        for entry in total_messages:
            if entry["role"] == "user":
                user_number = user_number + 1

        if flag == "multi_turn1":
            if turn == 3:
                reward_this_turn, new_query = await generate_context6_turn1(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]})
                    else:
                        messages.append({"reward" : [0.0, 0.0, 0.0]}) 
                    return
            elif turn == 5:
                reward_final_turn = await generate_context6_turn2(total_messages, reward_reference, content)
                if reward_final_turn<-0.01:
                    messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                else:
                    messages.append({"reward" : [1.0, reward_final_turn, reward_final_turn]})
                    print("hello")
                return
        elif flag == "multi_turn2":
            if turn == 3:
                reward_this_turn, new_query = await generate_context7_turn1(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]})
                    else:
                        messages.append({"reward" : [0.0, 0.0, 0.0]}) 
                    return
            elif turn == 5:
                reward_this_turn, new_query = await generate_context7_turn2(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                    else:
                        messages.append({"reward" : [1.0, 0.0, 0.0]}) 
                    return
            elif turn == 7:
                reward_final_turn = await generate_context7_turn3(total_messages, reward_reference, content)
                if reward_final_turn<-0.01:
                    messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                else:
                    messages.append({"reward" : [1.0, 1.0, reward_final_turn]})
                    print("hello")
                return
        elif user_number==2:
            if turn == 3:
                reward_this_turn, new_query = await generate_context2_turn1(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]})
                    else:
                        messages.append({"reward" : [0.0, 0.0, 0.0]}) 
                    return
            elif turn == 5:
                reward_this_turn, new_query = await generate_context2_turn2(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                    else:
                        messages.append({"reward" : [1.0, 0.0, 0.0]}) 
                    return
            elif turn == 7:
                reward_final_turn = await generate_context2_turn3(total_messages, reward_reference, content, flag)
                if reward_final_turn<-0.01:
                    messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                else:
                    messages.append({"reward" : [1.0, 1.0, reward_final_turn]})
                return
        elif user_number==1 and bool(re.match(r"^multiple_api_\d+$", flag)):
            #api_number = int(re.match(r"^multiple_api_(\d+)$", flag).group(1))
            reward_final_turn = await generate_context5(total_messages, reward_reference, content)
            if reward_final_turn<-0.01:
                messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
            else:
                messages.append({"reward" : [reward_final_turn, reward_final_turn, reward_final_turn]})
            return
        elif flag == "no_api":
            reward_final_turn = await generate_context4(total_messages, reward_reference, content)
            if reward_final_turn<-0.01:
                messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
            else:
                messages.append({"reward" : [reward_final_turn, reward_final_turn, reward_final_turn]})
            return
        else:
            if turn == 3:
                reward_this_turn, new_query = await generate_context1_turn1(total_messages, reward_reference, content)
                if not reward_this_turn > 0.01:
                    if reward_this_turn<-0.01:
                        messages.append({"reward" : [-0.0, -0.0, -0.0]})
                    else:
                        messages.append({"reward" : [0.0, 0.0, 0.0]}) 
                    return
            elif turn == 5:
                reward_final_turn = await generate_context1_turn2(total_messages, reward_reference, content, flag)
                if reward_final_turn<-0.01:
                    messages.append({"reward" : [-0.0, -0.0, -0.0]}) 
                else:
                    messages.append({"reward" : [1.0, reward_final_turn, reward_final_turn]})
                return
            else:
                messages.append({"reward" : [-0.0, -0.0, -0.0]})
                return
        
        # STEP 5: resubmit chat completions with code block output
        messages.append(new_query)
        self.scheduler.submit_chat_completions(messages=messages, request_id=completions.id, info=info, flag=flag,reward_reference=reward_reference,total_messages=total_messages)


    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], batch_tools_content, batch_flag, batch_reward_reference, batch_total_messages, batch_reward, n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # response_mask: [1,1,0,0,1,1,0,0]
        # reward_mask: [0,1,0,0,0,1,0,0]

        # prompts: [prompt] from input dataset

        prompts = []
        for prompt in batch.non_tensor_batch["raw_prompt"]:
            prompts.append(self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False))

        assert len(batch_conversations) == len(prompts) * n

        # sequences: [prompt + response]
        sequences = []
        for conversation,tools_content in zip(batch_conversations, batch_tools_content):
            sequences.append(self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False))

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

        reward_mask = keep_last_one_rows(response_mask)
        
        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        # === DEBUG: decode masked tokens for inspection ===
        valid_resp = responses["attention_mask"].bool()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        resp_keep_mask   = (response_mask > 0) & valid_resp
        reward_keep_mask = (reward_mask   > 0) & valid_resp

        resp_masked_ids = torch.where(resp_keep_mask,   responses["input_ids"], torch.full_like(responses["input_ids"], pad_id))
        rew_masked_ids  = torch.where(reward_keep_mask, responses["input_ids"], torch.full_like(responses["input_ids"], pad_id))

        response_full_text = self.tokenizer.batch_decode(responses["input_ids"].tolist(), skip_special_tokens=False)
        resp_masked_text = self.tokenizer.batch_decode(resp_masked_ids.tolist(), skip_special_tokens=False)
        rew_masked_text  = self.tokenizer.batch_decode(rew_masked_ids.tolist(),  skip_special_tokens=False)

        print("The following should be 0")
        print("the number of user in resp_masked_text", sum(['\nuser\n' in xx for xx in resp_masked_text]))
        print("the number of assistant in resp_masked_text", sum(['assistant\n' in xx for xx in resp_masked_text]))
        print("The above should be 0")
        for i, (t0, t1, t2, seq) in enumerate(zip(response_full_text, resp_masked_text, rew_masked_text,sequences)):
            if '\nuser\n' in t1:
                print(f"[SAMPLE {i}] response_full_text \n{t0}\n, response_mask kept:\n{t1}\n---\nreward_mask kept:\n{t2}\n====whole:\n{seq}")
            if i==0:
                print(f"[SAMPLE {i}] response_full_text \n{t0}\n, response_mask kept:\n{t1}\n---\nreward_mask kept:\n{t2}\n====whole:\n{seq}")

        '''
        import os
        if int(os.environ.get("LOCAL_RANK",0))==0:
            import ipdb; ipdb.set_trace()
        else:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        '''

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],  # [bsz, prompt_length]
                "responses": responses["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "reward_mask": reward_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([len(conversation) for conversation in batch_conversations], dtype=np.int32) 
        rewards = np.array([reward for reward in batch_reward])
    
        return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns, "__reward__": rewards})

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    print("CUDA visible devices:", torch.cuda.device_count())
    
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    all_actor_names = ray.util.list_named_actors()
    print("✅ 当前活跃的 Ray Actors:")
    for name in all_actor_names:
        print(name)
 
    # Load config
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    model_path = "/ebs-basemodeling/siyuanxu/model_inference/model/Qwen3-8B"
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes" ###### double check
    config.actor_rollout_ref.rollout.multi_turn.completion_callback = (
        "tests.workers.rollout.my_test_tool_completion_callback.Qwen3CustomToolCompletionCallback"
    )
    config.actor_rollout_ref.rollout.prompt_length = 32768
    config.actor_rollout_ref.rollout.response_length = 8192
    config.actor_rollout_ref.rollout.n = 8
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9

    # try:
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    # except Exception as e:
    #     print(f"Read tokenizer from model_path's huggingface subdir.")
    #     tokenizer = hf_tokenizer(os.path.join(config.actor_rollout_ref.model.path, "huggingface"))


    
    # Init sandbox and async rollout manager
    async_rollout_manager = init_async_rollout_manager(config)

    # Build dataset
    data_path = "/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/verl_toolmock/data/train.parquet"
    output_path = "/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/verl_toolmock/data/train_filter.parquet"
    
    dataset_all = pd.read_parquet(data_path).to_dict(orient='records')
    random.seed(42)
    dataset_all = dataset_all[0:45000]
    batch_size = 256
    dataset_new = []
    
    for start in range(0, len(dataset_all), batch_size):
        end = min(start + batch_size, len(dataset_all))
        dataset = dataset_all[start:end]

        controlled_generation_content = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(
                    [
                        [
                        {"role": "system", "content": build_tools_prompt_think_my(extract_json_objects(item["tools_kwargs"]))},
                        {"role": "user", "content": item["messages"][1]["content"]},
                        ]
                        for item in dataset
                    ]
                ),
                "tools_kwargs": np.array([item["tools_kwargs"] for item in dataset]),
                "flag": np.array([item["flag"] for item in dataset]),
                "reward_reference": np.array([pad_messages(item["reward_model"]) for item in dataset]),
                "total_messages": np.array([pad_messages(item["messages"]) for item in dataset])
            },
        )

        print("check first prompt:", controlled_generation_content.non_tensor_batch["raw_prompt"][0])

        result = async_rollout_manager.generate_sequences(prompts=controlled_generation_content)
        assert len(result) == len(dataset) * config.actor_rollout_ref.rollout.n

        rewards = np.asarray(result.non_tensor_batch["__reward__"][:,0]).reshape(len(dataset), config.actor_rollout_ref.rollout.n)

        avg = rewards.mean(axis=1)
        keep_mask = ~(avg > 0.58)

        for item, keep in zip(dataset, keep_mask):
            if keep:
                dataset_new.append(item)
        print("finish:",end)
        print("total:",len(dataset_all))
        pd.DataFrame(dataset_new).to_parquet(output_path, index=False)

    print(f"original {len(dataset_all)}，after filter {len(dataset_new)}；remove {(len(dataset_all)-len(dataset_new)).sum()} (average reward>0.58）")
    pd.DataFrame(dataset_new).to_parquet(output_path, index=False)
