import json
import os
import re
import time
from typing import Any, Optional

from openai import OpenAI


def convert_to_function_call(function_call_list):
    if isinstance(function_call_list, dict):
        function_call_list = [function_call_list]
    execution_list = []
    for function_call in function_call_list:
        for key, value in function_call.items():
            if isinstance(value, str):
                value = json.loads(value)
            execution_list.append(
                f"{key}({','.join([f'{k}={repr(v)}' for k, v in value.items()])})"
            )
    return execution_list


class QwenFCHandler:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        server_base_url: Optional[str],
        api_key: str,
        served_model: Optional[str],
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.registry_name = registry_name
        self.model_name_underline_replaced = (
            model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )

        default_base = os.getenv("INFERENCE_BASE_URL", "http://localhost:8010/v1")
        self.base_url = (server_base_url or default_base).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url + "/v1"
        self.api_key = api_key or os.getenv("INFERENCE_API_KEY", "token-abc123")
        self.served_model = served_model or os.getenv("INFERENCE_MODEL", model_name)

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.max_new_tokens = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", "4096"))

    def spin_up_local_server(
        self,  # kept for compatibility with caller; intentionally no-op
        num_gpus: int = 0,
        gpu_memory_utilization: float = 0.0,
        backend: str = "",
        skip_server_setup: bool = True,
        local_model_path: Optional[str] = None,
    ) -> None:
        return

    def close_local_server(self) -> None:
        return

    def _format_prompt(self, messages, function):
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages and messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        else:
            if messages and messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and isinstance(message["content"], str)
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]
                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + "\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"

                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (
                            tool_call == message["tool_calls"][0] and content
                        ) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"

                        if "function" in tool_call:
                            tool_call = tool_call["function"]

                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    def _query(self, messages, function):
        formatted_prompt = self._format_prompt(messages, function)

        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.served_model,
            temperature=self.temperature,
            prompt=formatted_prompt,
            max_tokens=self.max_new_tokens,
            timeout=72000,
        )
        end_time = time.time()

        return api_response, end_time - start_time, formatted_prompt

    def _parse_query_response(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if extracted_tool_calls:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }

        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": getattr(getattr(api_response, "usage", None), "prompt_tokens", 0),
            "output_token": getattr(getattr(api_response, "usage", None), "completion_tokens", 0),
        }

    @staticmethod
    def _extract_tool_calls(input_string: str):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception:
                continue
        return result

    def decode_execute(self, result: str) -> list[str]:
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if isinstance(item, str):
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)
