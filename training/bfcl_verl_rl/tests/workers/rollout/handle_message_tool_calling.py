import os 
import json
import re
from copy import deepcopy
from tqdm import tqdm
import boto3
import time
import random
import string
import glob
from functools import partial
from multiprocessing import Pool
import argparse
import numpy as np
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import random 
import traceback

def check_format_tool_call(model_output_1):
    num_open = model_output_1.count("<tool_call>")
    num_close = model_output_1.count("</tool_call>")
    if not num_open == num_close:
        return False

    model_output_notoll = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_1, flags=re.DOTALL).strip()
    print("hahahahahahahahah",len(model_output_notoll))

    if len(model_output_notoll)==0:
        if "\nuser\n" in model_output_1:
            return False
        elif "\nassistant\n" in model_output_1:
            return False
        elif "\n\n\n" in model_output_1:
            return False
        else:
            return True
    else:
        return False

def check_format_response(model_output_2):
    num_open = model_output_2.count("<tool_call>")
    num_close = model_output_2.count("</tool_call>")
    if not num_open == num_close:
        return False

    matches = re.findall(r"<tool_call>(.*?)</tool_call>", model_output_2, flags=re.DOTALL)
    for match in matches:
        content = match.strip()
        try:
            json.loads(content)
        except Exception:
            return False
        
    if "<tool_call>" in model_output_2:
        model_output_2_cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_2, flags=re.DOTALL).strip()
        if len(model_output_2_cleaned)==0 and not "\n\n\n" in model_output_2:
            return True
        else:
            return False
    else:
        if "\nuser\n" in model_output_2:
            return False
        elif "\nassistant\n" in model_output_2:
            return False
        elif "\n\n\n" in model_output_2:
            return False
        else:
            return True
    
def pad_messages(msgs, target_len=15):
    padded = msgs[:target_len]
    while len(padded) < target_len:
        padded = np.append(padded, {"role": "none", "content": ""}) 
    return padded


def build_tools_prompt_no_think(tools: list[dict]) -> str:

    header = """# Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    """
    
    tool_defs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

    footer = """\n</tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>"""
    
    return header + tool_defs + footer

def build_tools_prompt_think(tools: list[dict]) -> str:

    thinking = """# Thinking

    Think step-by-step first. Write your thinking process inside <think></think> XML tag. After the closing tag, present your response.
    
    """

    header = """# Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    """

    tool_defs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

    footer = """\n</tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>{"name": <function-name>, "arguments": <args-json-object>}</tool_call>"""

    return thinking + header + tool_defs + footer

build_tools_prompt_no_think_my=build_tools_prompt_no_think
build_tools_prompt_think_my=build_tools_prompt_think

def extract_json_objects(raw_text):
    """Extract individual JSON objects from a string by tracking balanced braces."""
    json_objects = []
    brace_count = 0
    start_idx = None

    for i, char in enumerate(raw_text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = raw_text[start_idx:i+1]
                try:
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    continue

    return json_objects

def remove_think_blocks(text):
    """
    Removes <think>...</think> blocks from the input text.
    If such blocks exist, also checks if they contain <tool_call>.
    Returns a tuple: (processed_text, contains_tool_call)
    """
    contains_tool_call = False

    #if not "<think>" in text:
    #    contains_tool_call = True
    #    return "", contains_tool_call

    if re.search(r"<tool_call>.*?<think>", text, flags=re.DOTALL):
        contains_tool_call = True
        return "", contains_tool_call
    
    think_blocks = re.finditer(r"<think\b[^>]*>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    contains_tool_call = False
    for m in think_blocks:
        if re.search(r"<tool_call\b", m.group(1), flags=re.IGNORECASE):
            contains_tool_call = True
            return "", contains_tool_call

    processed = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return processed, contains_tool_call

def convert_chat_completion_message_to_dict(msg):
    tool_calls = []
    for tc in msg.tool_calls:
        tool_calls.append({
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments 
            }
        })

    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": tool_calls
    }

import aiohttp
import asyncio
from uuid import uuid4
from openai import OpenAI

async def _chat_completions_aiohttp(
    base_url,
    api_key,
    messages,
    tools,
    extra_body=None,
    extra_headers=None,
    **sampling_params
):
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "model": sampling_params.pop("model"),
            "messages": messages,
            "tools": tools,
            **sampling_params,
        }
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        if extra_headers:
            headers.update(extra_headers)

        async with session.post(
            f"{base_url}chat/completions",
            json=payload,
            headers=headers
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

async def invoke_model_with_retries(body, max_retries=20):
    client = OpenAI(
        base_url="http://localhost:8010/v1",
        api_key="token-abc123",
    )

    parsed_body = json.loads(body)
    messages = parsed_body["messages"]

    retries = 0
    model_output = None
    request_id = uuid4().hex

    while retries < max_retries:
        try:
            completion = await _chat_completions_aiohttp(
                base_url=client.base_url,
                api_key=client.api_key,
                messages=messages,
                tools=[],
                extra_body=None,
                extra_headers={"x-request-id": request_id},
                temperature=0.0,
                top_p=1.0,
                max_tokens=4096,
                model="/ebs-basemodeling/siyuanxu/model_inference/model/Qwen3-8B"
            )
            model_output = completion["choices"][0]["message"]["content"].strip()
            #print('SEIUFSKJDBFKSDJFJKBSDJK')
            break
        except Exception as e:
            print("[Retry", retries, "]")
            #traceback.print_exc()
            print("Exception repr:", repr(e))
            #print("Exception str:", str(e))
            if retries < 0.3 * max_retries:
                await asyncio.sleep(10)
            else:
                await asyncio.sleep(50)
            retries += 1

    return model_output


async def invoke_model_with_retries_1(body, model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", max_retries=20):
    
    retries = 0
    loop = asyncio.get_running_loop()

    while retries < max_retries:
        try:
            bedrock = None
            aaa= random.choice([0, 1, 2])
            if aaa==0:
                bedrock = boto3.client('bedrock-runtime',region_name="us-east-2")
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            elif aaa==1:
                bedrock = boto3.client('bedrock-runtime',region_name="us-east-1")
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            elif aaa==2:
                bedrock = boto3.client('bedrock-runtime',region_name="us-east-1")
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
            #elif aaa ==3: 
            #    bedrock = boto3.client('bedrock-runtime',region_name="us-east-2")
            #    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0"

            response = await loop.run_in_executor(
                None,
                partial(bedrock.invoke_model, body=body, modelId=model_id)
            )

            response_body = json.loads(response.get("body").read())
            model_output = response_body.get("content")[0]['text'].strip()
            return model_output

        except Exception as e:
            print(f"[Retry {retries}] Error: {e}")
            if retries < 0.3 * max_retries:
                await asyncio.sleep(5)
            else:
                await asyncio.sleep(50)
            retries += 1

    raise RuntimeError("invoke_model failed after maximum retries")


async def call_model_checker(check_prompt):

    model_output= None

    retries = 0
    body = json.dumps({
            "max_tokens": 4096,
            "messages": [{"role": "user", "content":  check_prompt}],
            "anthropic_version": "bedrock-2023-05-31"
            })
    model_output = await invoke_model_with_retries(body)
    
    results = re.findall(r'^Semantically alignment:\s*(yes|no)', model_output, flags=re.MULTILINE)

    #print(results)

    try:
        if results[0]=="yes":
            return True
        else:
            return False
    except Exception as e:
        return False

async def check_by_model_1(tool_call, simple_tool_call, simple_tool_definition):

    model_checker_prompt= '''

    You will be given two tool calling actions in JSON format. 
    The first one is the golden tool calling action, representing the correct and ideal invocation based on the user intent. 
    The second is a candidate tool calling action, which may or may not align semantically with the golden one.

    The golden tool calling action: {golden_tool_call}
    The candidate tool calling action: {model_tool_call}

    Your task is to determine whether the candidate tool-calling action is semantically aligned with the golden reference.
    Alignment is achieved when all three of the following criteria are met:
    1. The candidate tool calling conveys the same or highly similar user intent as the golden tool calling.
    2. For parameter types int, float, and bool, the parameter values in the candidate tool-calling action must be exactly the same as those in the golden.
    3. For other parameter types, the parameters in the candidate tool-calling action must contain the information present in the golden or be similar in meaning. Exact wording is not required as long as the intended meaning is preserved.

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is semantically aligned with the golden or not.
    If they are aligned, explain the reasoning and highlight any differences in parameter values that still preserve the user intent.
    If they are not aligned, explain what the semantic mismatch is and what the candidate action is actually trying to do instead.
    SECOND, on a new line, state only "yes" or "no" to indicate the evaluation metric is followed or not. Your response should use the format:
    Evaluation explanation: <explanation>
    Semantically alignment: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    valid_flag = True

    try:
        tool_call_json = json.loads(tool_call)
        simple_tool_call_json = json.loads(simple_tool_call)

        assert tool_call_json.get("name") == simple_tool_call_json.get("name"), "wrong api"

        tool_call_parameter_list = tool_call_json.get("arguments")
        simple_tool_call_parameter_list = simple_tool_call_json.get("arguments")

        tool_call_parameter_name_list = list(tool_call_parameter_list.keys())
        simple_tool_call_parameter_name_list = list(simple_tool_call_parameter_list.keys())

        assert sorted(tool_call_parameter_name_list) == sorted(simple_tool_call_parameter_name_list), "wrong parameter"

        for name in tool_call_parameter_name_list:
            type(tool_call_parameter_list[name]) == type(simple_tool_call_parameter_list[name]), "wrong parameter type"

            if type(tool_call_parameter_list[name])==int or type(tool_call_parameter_list[name])==float or type(tool_call_parameter_list[name])==bool:
                assert tool_call_parameter_list[name] == simple_tool_call_parameter_list[name], "wrong parameter content: {name}, content model: {content_model}, content_original: {content_original}".format(name=name,content_original=str(simple_tool_call_parameter_list[name]),content_model=str(tool_call_parameter_list[name]))

        current_model_checker_prompt=model_checker_prompt.format(golden_tool_call=simple_tool_call, model_tool_call=tool_call)

        assert await call_model_checker(current_model_checker_prompt), "no good content"

    except Exception as e:
        print(e)
        return False

    return valid_flag

async def model_checker_1(model_output_1,simple_tool_call,simple_tool_definition):
    try:
        tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", model_output_1, re.DOTALL)
        assert len(tool_call_blocks) == 1, "len = {aaa}".format(aaa=len(tool_call_blocks))
        tool_call=tool_call_blocks[0]
        assert await check_by_model_1(tool_call, simple_tool_call, simple_tool_definition)
        #print("verifed 1 sucessful")
        return True
    
    except Exception as e:
        print(e)
        return False


async def model_checker_2(generate_response,query,tool_response,flag):

    Prompt_normal = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.

    The response is acceptable when all the following criteria are met:
    1. The responses of the AI assistant can address the user qeury. 
    2. The responses of the AI assistant includes all the ground-truth information.
    NOTE THAT it is acceptable (this criterion can be met) when the responses of the AI assistant includes extra information not limited to the ground-truth information. As long as it does not conflict with the ground-truth, do not treat it as fabricated information. Such extra information may be obtained by the assistant from other sources and can also be beneficial to the user.
    NOTE THAT it is acceptable (this criterion can be met) when the responses of the AI assistant includes extra information of the user query other than the provided user query. Do not treat it as fabricated information.

    The user query is: {user_query}
    The ground-truth information that can solve the user query is: {ground_truth}
    The response of the AI assistant query is: {response}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how each metric is followed.
    If they are not acceptable, indicate which metric is not followed and explain why.
    As metioned in criteria 2, do not only take the reason of including fabricated information to judge the response as unacceptable.
    SECOND, on a new line, state only "yes" or "no" to indicate all the evaluation metrics are followed or there is at least one not being followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    Prompt_long = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.

    The response is acceptable when all the following criteria are met:
    1. The responses of the AI assistant can address the user qeury. 
    2. The responses of the AI assistant includes all the ground-truth information.
    NOTE THAT it is acceptable (this criterion can be met) when the responses of the AI assistant includes extra information not limited to the ground-truth information. As long as it does not conflict with the ground-truth, do not treat it as fabricated information. Such extra information may be obtained by the assistant from other sources and can also be beneficial to the user.
    NOTE THAT it is acceptable (this criterion can be met) when the responses of the AI assistant includes extra information of the user query other than the provided user query. Do not treat it as fabricated information.

    The user query is: {user_query}
    The ground-truth information that can solve the user query is: {ground_truth}
    The response of the AI assistant query is: {response}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how each metric is followed.
    If they are not acceptable, indicate which metric is not followed and explain why.
    As metioned in criteria 2, do not only take the reason of including fabricated information to judge the response as unacceptable.
    SECOND, on a new line, state only "yes" or "no" to indicate all the evaluation metrics are followed or there is at least one not being followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    Prompt_error = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.
    The response of the AI assistant is based on an API calling with errors. 
    Therefore, the AI response are expected to identify API/tool calling is falied.

    The response is acceptable when all the following criteria are met:
    1. The AI assistant should realize the last API/tool calling is falied and the user query is still unsolved (for example, apologizing for the error to the user means the AI assistant has identified the last API tool calling return is invalid and realized the user query is still unsolved).

    The user query is: {user_query}
    The ground-truth information that can solve the user query is: {ground_truth}
    The response of the AI assistant query is: {response}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how each metric is followed.
    If they are not acceptable, indicate which metric is not followed and explain why.
    SECOND, on a new line, state only "yes" or "no" to indicate all the evaluation metrics are followed or there is at least one not being followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    Prompt_noise = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.
    The response of the AI assistant is based on a probalematic API, which does not provide useful information or wrong parameter values.

    The response is acceptable when all the following criteria are met:
    1. The AI assistant must identify the last API tool calling return is invalid and realize the user query is still unsolved (for example, apologizing for the confusion to the user or retrying the API calling means the AI assistant has identified the last API tool calling return is invalid and realized the user query is still unsolved). 
    2. The responses of the AI assistant should not pretend to have solved the user’s query.

    The user query is: {user_query}
    The ground-truth information that can solve the user query is: {ground_truth}
    The response of the AI assistant query is: {response}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how each metric is followed.
    If they are not acceptable, indicate which metric is not followed and explain why.
    SECOND, on a new line, state only "yes" or "no" to indicate all the evaluation metrics are followed or there is at least one not being followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    check_prompt = None

    if flag == "original_format" or flag == "with_wrong_parameter":
        check_prompt = Prompt_normal.format(user_query=query, ground_truth=tool_response, response=generate_response)
    elif flag == "with_noise":
        check_prompt = Prompt_long.format(user_query=query, ground_truth=tool_response, response=generate_response)
    elif flag == "error":
        check_prompt = Prompt_error.format(user_query=query, ground_truth=tool_response, response=generate_response)
    elif flag == "pure_noise" or flag == "pure_wrong_parameter":
        check_prompt = Prompt_noise.format(user_query=query, ground_truth=tool_response, response=generate_response)
    else:
        print("ggggggggggggggg")
        print("flag",flag)
        print("ggggggggggggggg")

    model_output= None

    retries = 0
    body = json.dumps({
            "max_tokens": 4096,
            "messages": [{"role": "user", "content":  check_prompt}],
            "anthropic_version": "bedrock-2023-05-31"
            })
    model_output = await invoke_model_with_retries(body)
    
    results = re.findall(r'^Acceptable:\s*(yes|no)', model_output, flags=re.MULTILINE)

    try:
        if results[0]=="yes":
            return True
        else:
            return False
    except Exception as e:
        return False


async def model_checker_3(generate_response,query,tool_call):

    Prompt_normal = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.

    The user has submitted a query in which some required parameter values from the ground-truth tool call are missing.
    The assistant is expected to ask the user to provide the missing parameter values.

    A response is considered acceptable if all of the following criteria are met:
    1. The assistant correctly identifies and asks for the specific parameter(s) that are missing from the user query but are required in the ground-truth tool call.
    2. The assistant must not call the tool in the current turn, i.e., must not have <tool_call></tool_call> tag in the response.
    
    The user query is: {user_query}
    The ground-truth tool call that can solve the user query is: {tool_call}
    The response of the AI assistant is: {generate_response}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how each metric is followed.
    If they are not acceptable, indicate which metric is not followed and explain why.
    SECOND, on a new line, state only "yes" or "no" to indicate all the evaluation metrics are followed or there is at least one not being followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    #print("query:",query)
    #print("generate_response: ",generate_response)
    #print("tool_call: ",tool_call)

    check_prompt = Prompt_normal.format(user_query=query, tool_call=tool_call, generate_response=generate_response)

    model_output= None

    retries = 0
    body = json.dumps({
            "max_tokens": 4096,
            "messages": [{"role": "user", "content":  check_prompt}],
            "anthropic_version": "bedrock-2023-05-31"
            })
    model_output = await invoke_model_with_retries(body)
    
    results = re.findall(r'^Acceptable:\s*(yes|no)', model_output, flags=re.MULTILINE)

    try:
        if results[0]=="yes":
            return True
        else:
            return False
    except Exception as e:
        return False

async def model_checker_4(output,user_query):

    check_prompt_tem = ''' You are an AI assistant evaluator. Your task is to evaluate the response provided by the assistant when handling a user query.
    Now, the user query cannot be solved by the existing APIs/tools. 

    The response is acceptable when all of the following criteria are met:
    1. The AI assistant should identify that the user query cannot be directly solved by the existing APIs/tools and tell the user. (For example, asking for clarification from the user means the current user query cannot be directly solved, which is acceptable. )
    2. The AI assistant must not try to call any tool, i.e., must not have <tool_call></tool_call> tag in the response.

    The user query is: {user_query}
    The AI assistant's response: {output}

    Here is the guideline for the evaluation format:

    FIRST provide a explanation why the candidate is acceptable or not.
    If they are acceptable, explain how the metric is followed.
    If they are not acceptable, explain why.
    SECOND, on a new line, state only "yes" or "no" to indicate the evaluation metric is followed or at least one not followed. Your response should use the format:
    Evaluation explanation: <explanation>
    Acceptable: <"yes" or "no">

    Now, it is your turn. Generated evaluation:
    '''

    check_prompt = check_prompt_tem.format(user_query = user_query, output=output)

    retries = 0
    body = json.dumps({
            "max_tokens": 4096,
            "messages": [{"role": "user", "content":  check_prompt}],
            "anthropic_version": "bedrock-2023-05-31"
            })
    model_output = await invoke_model_with_retries(body)
    
    results = re.findall(r'^Acceptable:\s*(yes|no)', model_output, flags=re.MULTILINE)
    #print(model_output)

    try:
        if results[0]=="yes":
            return True
        else:
            return False
    except Exception as e:
        return False


async def generate_context1_turn1(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0, None

    if not check_format_tool_call(model_output_1):
        return -1.0, None

    verified_1 = await model_checker_1(model_output_1,simple_tool_call,simple_tool_definition) 

    if not verified_1:
        return 0.0, None
    else:
        try:
            #return 1.0, {"role": "tool", "tool_call_id": model_output_1_tool_call_id[0], "content": list_message_complex[3]}
            return 1.0, {"role": "user", "content": "<tool_response>\n"+list_message_complex[3]+"\n</tool_response>"}
        except Exception as e:
            return 0.0, None


async def generate_context1_turn2(line_complex,line_simple, content, flag):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_2 = content or "" 
    model_output_2_nothink, think_no_good = remove_think_blocks(model_output_2)

    if not check_format_response(model_output_2_nothink):
        return -1.0

    if think_no_good:
        return -1.0

    model_output_2_nothink = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_2_nothink, flags=re.DOTALL).strip()
    verified_2 = await model_checker_2(model_output_2_nothink, query_simple, tool_response_simple, flag) 

    if verified_2:
        return 1.0
    else:
        return 0.0


async def generate_context5(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]

    matches = re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call_list = []
    for match in matches:
        json_str = match.group(1)
        simple_tool_call_list.append(json_str)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0

    if not check_format_tool_call(model_output_1):
        return -1.0

    matches = re.finditer(r'<tool_call>(.*?)</tool_call>', model_output_1, re.DOTALL)
    tool_call_list = []
    for match in matches:
        json_str = match.group(0)
        tool_call_list.append(json_str)

    #print(tool_call_list)
    
    reward_1 = 0.0

    if len(tool_call_list) != len(simple_tool_call_list):
        return 0.0

    for simple_tool_call in simple_tool_call_list:
        for model_output_1 in tool_call_list[0:len(simple_tool_call_list)]:
            if await model_checker_1(model_output_1,simple_tool_call, ""):
                reward_1 = reward_1+1
                break

    if reward_1 / len(simple_tool_call_list)>0.99:
        return 1.0
    else:
        return 0.0

async def generate_context4(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0

    if not check_format_response(model_output_1):
        return -1.0
    
    if "<tool_call>" in model_output_1:
        return 0.0
    elif "</tool_call>" in model_output_1:
        return 0.0

    verified_4 = await model_checker_4(model_output_1,list_message_complex[1]) 

    if verified_4:
        return 1.0
    else:
        return 0.0
    

async def generate_context2_turn1(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    
    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0, None
    
    if not check_format_response(model_output_1):
        return -1.0, None

    model_output_1 = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_1, flags=re.DOTALL).strip()
    verified_3 = await model_checker_3(model_output_1, list_message_complex[1], simple_tool_call) 

    if not verified_3:
        return 0.0, None
    else:
        return 1.0, {"role": "user", "content": list_message_complex[3]}


async def generate_context2_turn2(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)
    if think_no_good:
        return -1.0, None
    
    if not check_format_tool_call(model_output_1):
        return -1.0, None

    verified_4 = await model_checker_1(model_output_1,simple_tool_call,simple_tool_definition)

    if not verified_4:
        return 0.0, None
    else:
        try:
            #return 1.0, {"role": "tool", "tool_call_id": model_output_1_tool_call_id[0], "content": list_message_complex[5]} 
            return 1.0, {"role": "user", "content": "<tool_response>\n"+list_message_complex[5]+"\n</tool_response>"}
        except Exception as e:
            return 0.0, None

async def generate_context2_turn3(line_complex,line_simple, content,flag):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_simple_whole, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if not check_format_response(model_output_1):
        return -1.0

    if think_no_good:
        return -1.0
    
    model_output_1 = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_1, flags=re.DOTALL).strip()
    verified_5 = await model_checker_2(model_output_1, query_simple, tool_response_simple, flag) 

    if not verified_5:
        return 0.0
    else:
        return 1.0
    

async def generate_context6_turn1(line_complex,line_simple, content):

    tool_call_first = None
    tool_call_second = None

    for entry in line_complex:
        if entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            if tool_call_first is None:
                tool_call_first = entry["content"]
            elif tool_call_second is None:
                tool_call_second = entry["content"]
            elif tool_call_first is not None and tool_call_second is not None:
                break
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_first, re.DOTALL)
    first_tool_call = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0, None

    if not check_format_tool_call(model_output_1):
        return -1.0, None

    verified_1 = await model_checker_1(model_output_1,first_tool_call,"") 

    if not verified_1:
        return 0.0, None
    else:
        try:
            #return 1.0, {"role": "tool", "tool_call_id": model_output_1_tool_call_id[0], "content": list_message_complex[3]}
            return 1.0, {"role": "user", "content": "<tool_response>\n"+list_message_complex[3]+"\n</tool_response>"}
        except Exception as e:
            return 0.0, None

async def generate_context6_turn2(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_second = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_second, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0

    if not check_format_tool_call(model_output_1):
        return -1.0

    verified_1 = await model_checker_1(model_output_1,simple_tool_call,simple_tool_definition) 

    if not verified_1:
        return 0.0
    else:
        try:
            #return 1.0, {"role": "tool", "tool_call_id": model_output_1_tool_call_id[0], "content": list_message_complex[3]}
            return 1.0, {"role": "user", "content": "<tool_response>\n"+list_message_complex[3]+"\n</tool_response>"}
        except Exception as e:
            return 0.0

async def generate_context7_turn1(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_first = entry["content"]
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_first, re.DOTALL)
    simple_tool_call = match.group(1)
    match = re.search(r'<tools>\s*(\{.*?\})\s*</tools>', system_prompt_simple, re.DOTALL)
    simple_tool_definition = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0, None

    if not check_format_tool_call(model_output_1):
        return -1.0, None

    verified_1 = await model_checker_1(model_output_1,simple_tool_call,simple_tool_definition) 

    if not verified_1:
        return 0.0, None
    else:
        try:
            #return 1.0, {"role": "tool", "tool_call_id": model_output_1_tool_call_id[0], "content": list_message_complex[3]}
            return 1.0, {"role": "user", "content": "<tool_response>\n"+list_message_complex[3]+"\n</tool_response>"}
        except Exception as e:
            return 0.0, None


async def generate_context7_turn2(line_complex,line_simple, content):

    for entry in line_simple:
        if entry["role"] == "system":
            system_prompt_simple = entry["content"]
        elif entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            tool_call_simple_whole = entry["content"]
        elif entry["role"] == "user":
            query_simple = entry["content"]
        elif entry["role"] == "tool":
            tool_response_simple = entry["content"]

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_2 = content or "" 
    model_output_2_nothink, think_no_good = remove_think_blocks(model_output_2)

    if not check_format_response(model_output_2_nothink):
        return -1.0, None

    if think_no_good:
        return -1.0, None

    model_output_2_nothink = re.sub(r"<tool_call>.*?</tool_call>", "", model_output_2_nothink, flags=re.DOTALL).strip()
    verified_2 = await model_checker_2(model_output_2_nothink, query_simple, tool_response_simple, "original_format") 

    if verified_2:
        return 1.0, {"role": "user", "content": list_message_complex[5]}
    else:
        return 0.0, None


async def generate_context7_turn3(line_complex,line_simple, content):

    tool_call_first = None
    tool_call_second = None

    for entry in line_complex:
        if entry["role"] == "assistant" and "<tool_call>" in entry["content"]:
            if tool_call_first is None:
                tool_call_first = entry["content"]
            elif tool_call_second is None:
                tool_call_second = entry["content"]
            elif tool_call_first is not None and tool_call_second is not None:
                break
    
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', tool_call_second, re.DOTALL)
    simple_tool_call = match.group(1)

    list_message_complex = []
    for entry_complex in line_complex:
        list_message_complex.append(entry_complex["content"])

    model_output_1 = content or ""
    model_output_1, think_no_good = remove_think_blocks(model_output_1)

    if think_no_good:
        return -1.0

    if not check_format_tool_call(model_output_1):
        return -1.0

    verified_1 = await model_checker_1(model_output_1,simple_tool_call,"") 

    if not verified_1:
        return 0.0
    else:
        return 1.0