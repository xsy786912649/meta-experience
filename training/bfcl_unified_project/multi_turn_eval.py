import copy
import importlib
import inspect
import json
import re

from config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES


def execute_multi_turn_func_call(
    func_call_list: list[str],
    initial_config: dict,
    involved_classes: list,
    model_name: str,
    test_entry_id: str,
    long_context: bool = False,
    is_eval_run: bool = False,
) -> tuple[list[str], dict]:
    if is_eval_run:
        model_name += "_eval"

    class_method_name_mapping = {}
    involved_instances = {}
    for class_name in involved_classes:
        module_name = CLASS_FILE_PATH_MAPPING[class_name]
        instance_name = f"{model_name}_{test_entry_id}_{class_name}_instance"
        instance_name = re.sub(r"[-./]", "_", instance_name)
        if instance_name not in globals():
            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            class_instance = class_()
            if class_name not in STATELESS_CLASSES:
                class_initial_config = initial_config.get(class_name, {})
                class_instance._load_scenario(
                    copy.deepcopy(class_initial_config), long_context=long_context
                )
            globals()[instance_name] = class_instance
        else:
            class_instance = globals()[instance_name]

        involved_instances[class_name] = class_instance

        for method_name, method in inspect.getmembers(class_instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            class_method_name_mapping[method_name] = instance_name

    execution_results = []
    for func_call in func_call_list:
        func_call = _process_method_calls(func_call, class_method_name_mapping)

        try:
            func_call_copy = func_call
            if "(" in func_call_copy:
                func_call_copy = func_call_copy.split("(")[0]
            if "." in func_call_copy:
                func_call_copy = func_call_copy.split(".")[1]
            if func_call_copy in ["kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"]:
                raise Exception(f"Function call {func_call_copy} is not allowed.")

            func_call_result = eval(func_call)

            if isinstance(func_call_result, str):
                pass
            elif isinstance(func_call_result, dict):
                try:
                    func_call_result = json.dumps(func_call_result)
                except Exception:
                    func_call_result = str(func_call_result)
            else:
                func_call_result = str(func_call_result)

            execution_results.append(func_call_result)
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")

    return execution_results, involved_instances


def is_empty_execute_response(input_list: list) -> bool:
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def _process_method_calls(function_call_string: str, instance_mapping: dict) -> str:
    def replace_function(match):
        func_name = match.group(1)
        if func_name in instance_mapping:
            return f"{instance_mapping[func_name]}.{func_name}"
        return func_name

    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
    processed_string = re.sub(pattern, replace_function, function_call_string)

    return processed_string


def multi_turn_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
    test_entry: dict,
    test_category: str,
    model_name: str,
) -> dict:
    initial_config: dict = test_entry["initial_config"]
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    execution_results: list[dict] = []
    all_turn_model_execution_results: list[str] = []

    for turn_index, single_turn_ground_truth_list in enumerate(multi_turn_ground_truth_list):
        single_turn_model_response_list = multi_turn_model_result_list_decoded[turn_index]

        single_turn_model_execution_results = []
        single_turn_model_execution_results_uncombined = []
        model_instances = {}

        for single_step_model_response in single_turn_model_response_list:
            single_step_model_execution_results, model_instances = execute_multi_turn_func_call(
                func_call_list=single_step_model_response,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name,
                test_entry_id=test_entry_id,
                long_context=("long_context" in test_category),
                is_eval_run=True,
            )
            single_turn_model_execution_results.extend(single_step_model_execution_results)
            single_turn_model_execution_results_uncombined.append(single_step_model_execution_results)

        single_turn_ground_truth_execution_results, ground_truth_instances = execute_multi_turn_func_call(
            func_call_list=single_turn_ground_truth_list,
            initial_config=initial_config,
            involved_classes=involved_classes,
            model_name=model_name + "_ground_truth",
            test_entry_id=test_entry_id,
            long_context=("long_context" in test_category),
            is_eval_run=True,
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)
        execution_results.append(
            {
                "model": single_turn_model_execution_results_uncombined,
                "ground_truth": single_turn_ground_truth_execution_results,
            }
        )

        if len(single_turn_ground_truth_list) > 0:
            if not single_turn_model_response_list or is_empty_execute_response(
                single_turn_model_response_list
            ):
                return {
                    "valid": False,
                    "error_message": f"Model response list is empty for turn {turn_index}",
                    "error_type": "multi_turn:empty_turn_model_response",
                    "details": {"execution_result": execution_results},
                }

        if not single_turn_ground_truth_list:
            continue

        assert len(model_instances) == len(ground_truth_instances)
        assert set(model_instances.keys()) == set(ground_truth_instances.keys())

        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            state_check_result["execution_result"] = execution_results
            return state_check_result

        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

    return {"valid": True}


def state_checker(model_instances: dict, ground_truth_instances: dict):
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)

        if not valid:
            model_instance_attributes = {
                key: value
                for key, value in vars(model_instance).items()
                if not key.startswith("_")
            }
            ground_truth_instance_attributes = {
                key: value
                for key, value in vars(ground_truth_instance).items()
                if not key.startswith("_")
            }
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the state with ground truth instance.",
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {
                    "differences": differences,
                    "model_instance_state": model_instance_attributes,
                    "ground_truth_instance_state": ground_truth_instance_attributes,
                },
            }

    return {"valid": True}


def response_checker(model_response_list: list, ground_truth_response_list: list, turn_index: int):
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": (
                "Model response execution results so far does not contain all the ground truth "
                f"response execution results for turn {turn_index}."
            ),
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {
                "missing_items": missing_items,
                "model_response (including all previous turns)": model_response_list,
                "ground_truth_response (only the current turn)": ground_truth_response_list,
            },
        }

    return {"valid": True}


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
