import copy
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from recipe.bfcl_multiturn.bfcl_config import (
    CLASS_FILE_PATH_MAPPING,
    MAXIMUM_STEP_LIMIT,
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    MULTI_TURN_FUNC_DOC_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    STATELESS_CLASSES,
    VERSION_PREFIX,
)


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_dataset_entry(test_category: str) -> list[dict]:
    return load_jsonl(PROMPT_PATH / f"{VERSION_PREFIX}_{test_category}.json")


def load_possible_answers(test_category: str) -> list[dict]:
    return load_jsonl(POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{test_category}.json")


def populate_test_case_with_predefined_functions(entry: dict) -> dict:
    entry = copy.deepcopy(entry)
    involved_classes = entry["involved_classes"]
    entry["function"] = []

    for func_collection in involved_classes:
        func_doc = load_jsonl(MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection])
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


def process_method_calls(function_call_string: str, instance_mapping: dict[str, str]) -> str:
    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
    return re.sub(pattern, replace_function, function_call_string)


def execute_multi_turn_func_call(
    func_call_list: list[str],
    initial_config: dict,
    involved_classes: list[str],
    run_namespace: str,
    test_entry_id: str,
    long_context: bool = False,
    is_eval_run: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    if is_eval_run:
        run_namespace += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        instance_name = f"{run_namespace}_{test_entry_id}_{class_name}_instance"
        instance_name = re.sub(r"[-./]", "_", instance_name)

        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
            globals()[instance_name] = class_instance
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance
        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if not method_name.startswith("_"):
                class_method_name_mapping[method_name] = instance_name

    execution_results: list[str] = []
    for func_call in func_call_list:
        func_call = process_method_calls(func_call, class_method_name_mapping)
        try:
            func_name = func_call.split("(")[0].split(".")[-1]
            if func_name in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_name} is not allowed.")
            func_call_result = eval(func_call)
            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                try:
                    func_call_result = json.dumps(func_call_result, ensure_ascii=False)
                except Exception:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)
            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def _compare_instances(model_object, ground_truth_object):
    assert type(model_object) == type(ground_truth_object)
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_object, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)
        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}
    return valid, differences


def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
    list2_copy = list2[:]
    missing_elements = []
    for item in list1:
        try:
            list2_copy.remove(item)
        except ValueError:
            missing_elements.append(item)
    return len(missing_elements) == 0, missing_elements


def state_checker(model_instances: dict, ground_truth_instances: dict) -> dict:
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)
        if not valid:
            return {
                "valid": False,
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {"differences": differences},
            }
    return {"valid": True}


def response_checker(model_response_list: list, ground_truth_response_list: list, turn_index: int) -> dict:
    is_subsequence, missing_items = _is_subsequence_unordered(ground_truth_response_list, model_response_list)
    if not is_subsequence:
        return {
            "valid": False,
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {"missing_items": missing_items, "turn_index": turn_index},
        }
    return {"valid": True}


def multi_turn_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
    test_entry: dict,
    run_namespace: str,
) -> dict:
    initial_config: dict = test_entry["initial_config"]
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    all_turn_model_execution_results: list[str] = []

    for turn_index, single_turn_ground_truth_list in enumerate(multi_turn_ground_truth_list):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]
        single_turn_model_execution_results = []
        model_instances = {}

        for single_step_model_response in single_turn_model_response_list:
            single_step_model_execution_results, model_instances = execute_multi_turn_func_call(
                func_call_list=single_step_model_response,
                initial_config=initial_config,
                involved_classes=involved_classes,
                run_namespace=run_namespace,
                test_entry_id=test_entry_id,
                long_context=("long_context" in test_category),
                is_eval_run=True,
            )
            single_turn_model_execution_results.extend(single_step_model_execution_results)

        single_turn_ground_truth_execution_results, ground_truth_instances = execute_multi_turn_func_call(
            func_call_list=single_turn_ground_truth_list,
            initial_config=initial_config,
            involved_classes=involved_classes,
            run_namespace=run_namespace + "_ground_truth",
            test_entry_id=test_entry_id,
            long_context=("long_context" in test_category),
            is_eval_run=True,
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)

        if len(single_turn_ground_truth_list) > 0 and not single_turn_model_response_list:
            return {"valid": False, "error_type": "multi_turn:empty_turn_model_response"}

        if not single_turn_ground_truth_list:
            continue

        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            return state_check_result

        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

    return {"valid": True}


TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls_from_text(text: str) -> list[dict]:
    out = []
    for match in TOOL_CALL_PATTERN.findall(text or ""):
        try:
            payload = json.loads(match)
            if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                out.append(payload)
        except Exception:
            continue
    return out


def convert_to_function_calls(tool_calls: list[dict]) -> list[str]:
    execution_list = []
    for call in tool_calls:
        name = call["name"]
        args = call["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        execution_list.append(f"{name}({','.join([f'{k}={repr(v)}' for k, v in args.items()])})")
    return execution_list


def decode_tool_calls(text: str) -> list[str]:
    tool_calls = extract_tool_calls_from_text(text)
    if not tool_calls:
        return []
    return convert_to_function_calls(tool_calls)


def build_system_prompt_with_tools(tools: list[dict]) -> str:
    tool_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_json}\n"
        "</tools>\n\n"
        "For each function call, return a json object with function name and arguments within "
        "<tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def build_reward_vector(num_assistant_turns: int, success: bool) -> list[float]:
    if num_assistant_turns <= 0:
        return [0.0]
    rewards = [0.0] * num_assistant_turns
    rewards[-1] = 1.0 if success else -1.0
    return rewards


def bounded_should_stop(num_assistant_turns: int) -> bool:
    return num_assistant_turns >= MAXIMUM_STEP_LIMIT

