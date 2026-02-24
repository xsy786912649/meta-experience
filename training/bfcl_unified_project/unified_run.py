import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Optional

from config import (
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    MAXIMUM_STEP_LIMIT,
    MODEL_CONFIG,
    MULTI_TURN_CATEGORIES,
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    MULTI_TURN_FUNC_DOC_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    STATELESS_CLASSES,
    VERSION_PREFIX,
)
from io_utils import load_jsonl
from multi_turn_eval import execute_multi_turn_func_call, is_empty_execute_response, multi_turn_checker
from qwen_fc import QwenFCHandler


def _make_json_serializable(value):
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _extract_system_block(formatted_prompt: str) -> str | None:
    if not formatted_prompt:
        return None
    marker = "<|im_start|>user"
    idx = formatted_prompt.find(marker)
    if idx == -1:
        return formatted_prompt.strip() + "\n"
    return formatted_prompt[:idx].rstrip() + "\n"


def _format_messages(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + ("\n" if parts else "")


def _format_tool_responses(tool_responses: list[str]) -> str:
    parts = []
    for content in tool_responses:
        parts.append("<|im_start|>user")
        parts.append(f"<tool_response>\n{content}\n</tool_response>")
        parts.append("<|im_end|>")
    return "\n".join(parts) + ("\n" if parts else "")


def _format_state_block(state_items: list[dict]) -> str:
    if not state_items:
        return ""
    payload = json.dumps(_make_json_serializable(state_items), ensure_ascii=False)
    return f"<|im_start|>state_info\n{payload}<|im_end|>\n"


def build_trajectory_text(inference_log: list) -> str:
    initial_state = []
    turns = []
    pending_state_after_turn = None

    for item in inference_log:
        if isinstance(item, list):
            state_payload = []
            for state_item in item:
                if state_item.get("role") != "state_info":
                    continue
                state_payload.append(
                    {
                        "class_name": state_item.get("class_name"),
                        "content": state_item.get("content"),
                    }
                )
            if not turns:
                initial_state = state_payload
            else:
                pending_state_after_turn = state_payload
        elif isinstance(item, dict):
            turn_query = item.get("begin_of_turn_query", [])
            turn = {
                "begin_of_turn_query": turn_query,
                "inference_input": None,
                "steps": [],
                "state_after_turn": None,
            }
            step_keys = [k for k in item.keys() if k.startswith("step_")]
            step_keys.sort(key=lambda x: int(x.split("_")[1]))

            for step_key in step_keys:
                step_events = item.get(step_key, [])
                assistant_output = None
                tool_responses = []
                for ev in step_events:
                    role = ev.get("role")
                    if role == "inference_input" and turn["inference_input"] is None:
                        content = ev.get("content") or {}
                        turn["inference_input"] = content.get("formatted_prompt")
                    elif role == "assistant":
                        assistant_output = str(ev.get("content"))
                    elif role == "tool":
                        tool_responses.append(str(ev.get("content")))
                turn["steps"].append(
                    {
                        "assistant_output": assistant_output,
                        "tool_responses": tool_responses,
                    }
                )
            if pending_state_after_turn is not None:
                turn["state_after_turn"] = pending_state_after_turn
                pending_state_after_turn = None
            turns.append(turn)

    system_block = None
    for turn in turns:
        system_block = _extract_system_block(turn.get("inference_input"))
        if system_block:
            break

    text_parts = []
    if system_block:
        text_parts.append(system_block)
    if initial_state:
        text_parts.append(_format_state_block(initial_state))

    for turn in turns:
        if turn.get("begin_of_turn_query"):
            text_parts.append(_format_messages(turn["begin_of_turn_query"]))
        for step in turn.get("steps", []):
            assistant_output = step.get("assistant_output")
            if assistant_output:
                text_parts.append(f"<|im_start|>assistant\n{assistant_output}<|im_end|>\n")
            tool_responses = step.get("tool_responses", [])
            if tool_responses:
                text_parts.append(_format_tool_responses(tool_responses))
        state_after = turn.get("state_after_turn")
        if state_after:
            text_parts.append(_format_state_block(state_after))

    return "".join(text_parts).rstrip() + "\n"


def load_dataset_entry(test_category: str) -> list[dict]:
    file_name = f"{VERSION_PREFIX}_{test_category}.json"
    return load_jsonl(PROMPT_PATH / file_name)


def load_possible_answers(test_category: str) -> list[dict]:
    file_name = f"{VERSION_PREFIX}_{test_category}.json"
    return load_jsonl(POSSIBLE_ANSWER_PATH / file_name)


def populate_test_case_with_predefined_functions(entry: dict) -> dict:
    entry = copy.deepcopy(entry)
    involved_classes = entry["involved_classes"]
    entry["function"] = []
    for func_collection in involved_classes:
        func_doc = load_jsonl(
            MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        entry["function"].extend(func_doc)

    if "missed_function" in entry:
        for turn_index, missed_func_names in entry["missed_function"].items():
            entry["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(entry["function"]):
                    if func_doc["name"] == missed_func_name:
                        entry["missed_function"][turn_index].append(func_doc)
                        entry["function"].pop(i)
                        break
    return entry


def generate_single_entry(
    handler: QwenFCHandler,
    test_entry: dict,
    include_input_log: bool = True,
    exclude_state_log: bool = False,
) -> tuple[list[list], dict]:
    initial_config: dict = test_entry.get("initial_config", {})
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category: str = test_entry_id.rsplit("_", 1)[0]
    holdout_function: dict = test_entry.get("missed_function", {})

    all_model_response: list[list] = []
    all_inference_log: list = []
    force_quit = False

    _, involved_instances = execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        handler.model_name_underline_replaced,
        test_entry_id,
        long_context=("long_context" in test_category),
        is_eval_run=False,
    )

    if not exclude_state_log:
        state_log = []
        for class_name, class_instance in involved_instances.items():
            if class_name in STATELESS_CLASSES:
                continue
            class_instance = copy.deepcopy(class_instance)
            state_log.append(
                {
                    "role": "state_info",
                    "class_name": class_name,
                    "content": {
                        key: value
                        for key, value in vars(class_instance).items()
                        if not key.startswith("_")
                    },
                }
            )
        if state_log:
            all_inference_log.append(state_log)

    messages = []
    functions = list(test_entry["function"])
    all_multi_turn_messages: list[list[dict]] = test_entry["question"]

    for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
        if str(turn_idx) in holdout_function:
            current_turn_message = [
                {
                    "role": "user",
                    "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                        functions=holdout_function[str(turn_idx)]
                    ),
                }
            ]

        messages.extend(current_turn_message)
        current_turn_response = []
        current_turn_inference_log = {"begin_of_turn_query": current_turn_message}
        count = 0

        while True:
            current_step_inference_log: list[dict] = []
            current_turn_inference_log[f"step_{count}"] = current_step_inference_log
            try:
                api_response, _, formatted_prompt = handler._query(messages, functions)
            except Exception as e:
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": "Error during model query. Skipping this entry.",
                        "error": str(e),
                    }
                )
                force_quit = True
                break

            if include_input_log:
                current_step_inference_log.append(
                    {"role": "inference_input", "content": {"formatted_prompt": formatted_prompt}}
                )

            model_response_data = handler._parse_query_response(api_response)
            model_responses = model_response_data["model_responses"]
            messages.append(model_response_data["model_responses_message_for_chat_history"])
            current_turn_response.append(model_responses)
            current_step_inference_log.append({"role": "assistant", "content": model_responses})

            try:
                decoded_model_responses = handler.decode_execute(model_responses)
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": "Successfully decoded model response.",
                        "model_response_decoded": decoded_model_responses,
                    }
                )
                if is_empty_execute_response(decoded_model_responses):
                    current_step_inference_log.append(
                        {
                            "role": "handler_log",
                            "content": "Empty response from the model. Proceed to next turn.",
                            "model_response_decoded": decoded_model_responses,
                        }
                    )
                    break
            except Exception as e:
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": "Error decoding the model response. Proceed to next turn.",
                        "error": str(e),
                    }
                )
                break

            execution_results, involved_instances = execute_multi_turn_func_call(
                decoded_model_responses,
                initial_config,
                involved_classes,
                handler.model_name_underline_replaced,
                test_entry_id,
                long_context=("long_context" in test_category),
                is_eval_run=False,
            )

            for execution_result, decoded_call in zip(execution_results, decoded_model_responses):
                messages.append({"role": "tool", "name": decoded_call, "content": execution_result})
                current_step_inference_log.append({"role": "tool", "content": execution_result})

            count += 1
            if count > MAXIMUM_STEP_LIMIT:
                force_quit = True
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                    }
                )
                break

        all_model_response.append(current_turn_response)
        all_inference_log.append(current_turn_inference_log)

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if class_name in STATELESS_CLASSES:
                    continue
                class_instance = copy.deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if state_log:
                all_inference_log.append(state_log)

        if force_quit:
            break

    metadata = {"inference_log": all_inference_log}
    return all_model_response, metadata


def evaluate_single_entry(
    handler: QwenFCHandler,
    test_entry: dict,
    model_result_list: list,
    ground_truth_list: list[list[str]],
    model_registry: str,
) -> bool:
    if not isinstance(model_result_list, list):
        return False
    if len(model_result_list) != len(ground_truth_list):
        return False

    multi_turn_model_result_list_decoded: list[list[list[str]]] = []
    for single_turn_model_result_list in model_result_list:
        single_turn_model_result_list_decoded = []
        for model_result_item in single_turn_model_result_list:
            try:
                decoded_result = handler.decode_execute(model_result_item)
                if is_empty_execute_response(decoded_result):
                    continue
                single_turn_model_result_list_decoded.append(decoded_result)
            except Exception:
                continue
        multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

    checker = multi_turn_checker(
        multi_turn_model_result_list_decoded,
        ground_truth_list,
        copy.deepcopy(test_entry),
        test_entry["id"].rsplit("_", 1)[0],
        model_registry.replace("/", "_"),
    )
    return bool(checker.get("valid"))


def _iter_entries(args):
    if args.entry_json:
        entry = json.loads(args.entry_json)
        if "function" in entry:
            # keep as-is when user provides full entry
            pass
        else:
            entry = populate_test_case_with_predefined_functions(entry)
        if args.ground_truth_json:
            gt = json.loads(args.ground_truth_json)
        elif "ground_truth" in entry:
            gt = entry["ground_truth"]
        else:
            raise ValueError("ground truth required for --entry-json")
        yield entry, gt
        return

    categories = MULTI_TURN_CATEGORIES
    if args.categories:
        categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    for category in categories:
        entries = load_dataset_entry(category)
        answers = load_possible_answers(category)
        answer_map = {x["id"]: x["ground_truth"] for x in answers}
        for entry in entries:
            if args.entry_id and entry.get("id") != args.entry_id:
                continue
            entry = populate_test_case_with_predefined_functions(entry)
            gt = answer_map.get(entry["id"])
            if gt is None:
                continue
            yield entry, gt


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--entry-id", type=str, default=None)
    parser.add_argument("--entry-json", type=str, default=None)
    parser.add_argument("--ground-truth-json", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()
    fixed_model = "Qwen/Qwen3-8B-FC"

    output_fp = None
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fp = output_path.open("w", encoding="utf-8")

    def emit_result(obj: dict):
        payload = json.dumps(obj, ensure_ascii=False)
        if output_fp is not None:
            output_fp.write(payload + "\n")
            output_fp.flush()
        else:
            print(payload)

    if fixed_model not in MODEL_CONFIG:
        out = {"id": None, "success": False, "trajectory": ""}
        emit_result(out)
        if output_fp is not None:
            output_fp.close()
        return

    handler = QwenFCHandler(
        model_name=MODEL_CONFIG[fixed_model]["model_name"],
        temperature=args.temperature,
        registry_name=fixed_model,
        server_base_url="http://localhost:8010/v1",
        api_key="token-abc123",
        served_model="/data/zhimeng/model/Qwen3-8B",
    )

    try:
        for entry, ground_truth in _iter_entries(args):
            result_obj = {"id": entry.get("id"), "success": False, "trajectory": ""}
            model_result = None
            metadata = {"inference_log": []}

            try:
                model_result, metadata = generate_single_entry(handler, entry)
            except Exception:
                model_result = None

            if model_result is not None:
                try:
                    result_obj["trajectory"] = build_trajectory_text(metadata.get("inference_log", []))
                except Exception:
                    try:
                        result_obj["trajectory"] = json.dumps(
                            _make_json_serializable(metadata.get("inference_log", [])), ensure_ascii=False
                        )
                    except Exception:
                        result_obj["trajectory"] = ""

            if model_result is not None:
                try:
                    result_obj["success"] = evaluate_single_entry(
                        handler=handler,
                        test_entry=entry,
                        model_result_list=model_result,
                        ground_truth_list=ground_truth,
                        model_registry=fixed_model,
                    )
                except Exception:
                    result_obj["success"] = False

            emit_result(result_obj)
    finally:
        handler.close_local_server()
        if output_fp is not None:
            output_fp.close()


if __name__ == "__main__":
    main()
