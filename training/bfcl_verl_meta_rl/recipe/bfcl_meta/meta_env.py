import json
from copy import deepcopy
from typing import Any

from recipe.bfcl_multiturn.bfcl_env import (
    STATELESS_CLASSES,
    build_system_prompt_with_tools,
)

SUMMARY_SYSTEM_PROMPT = (
    "You write short environment experience memos for future tool-calling tasks in the same environment. "
    "Extract reusable constraints, tool behavior patterns, common failure modes, and practical tactics. "
    "Keep the memo concise, concrete, and action-oriented. Light structure is fine, but do not force a rigid schema."
)

SUMMARY_USER_PROMPT_PREFIX = (
    "You are writing an environment experience memo for future tool-calling tasks in the same environment.\n\n"
    "Write a compact, concrete, action-oriented memo. Prefer stable environment knowledge over one-off task details. "
    "Only include observations that are clearly supported by the trajectory, tool behavior, checker output, or state changes.\n\n"
    "Your job is not to answer the user or summarize the task for the user. Your job is to record reusable operational knowledge "
    "that will help another agent solve a different task in the same environment more reliably.\n\n"
    "Focus on:\n"
    "- environment-specific constraints\n"
    "- tool behavior patterns\n"
    "- hidden preconditions\n"
    "- common failure modes\n"
    "- recovery strategies and effective action orderings\n"
    "- reusable heuristics for future tasks\n\n"
    "Required style:\n"
    "- Write as an internal memo for another agent.\n"
    "- Be concise but specific.\n"
    "- Prefer reusable lessons over narrative recap.\n"
    "- Explain what to check first, what order to act in, and what mistakes to avoid.\n"
    "- Ground claims in observed behavior only.\n\n"
    "Do not:\n"
    "- write a customer-facing response\n"
    '- say "task completed", "successfully created", "let me know", "feel free to ask", or similar assistant-style closing language\n'
    "- restate the whole trajectory\n"
    "- celebrate, apologize, or add filler\n"
    "- overfit to one task when the lesson is not reusable\n"
    "- include unsupported speculation\n\n"
    "Preferred content:\n"
    "1. Environment-specific facts / constraints\n"
    "2. Tool behavior patterns\n"
    "3. Common failure modes / hidden preconditions\n"
    "4. Useful recovery strategies / action orderings\n"
    "5. Reusable heuristics\n\n"
    "Good example:\n"
    "Environment-specific facts / constraints:\n"
    "- Travel tools require a valid access token for booking, cancellation, and invoice retrieval.\n"
    "- Booking IDs are reused across follow-up tools and must be preserved exactly.\n"
    "- Travel dates must use YYYY-MM-DD format, and airport codes must be valid IATA codes.\n\n"
    "Tool behavior patterns:\n"
    "- `book_flight` depends on exact parameters such as origin, destination, date, class, and card ID.\n"
    "- `retrieve_invoice` is the reliable source of final transaction details when prices appear inconsistent.\n\n"
    "Common failure modes / hidden preconditions:\n"
    "- Missing or invalid `booking_id` causes cancellation and invoice lookup to fail.\n"
    "- Estimated cost may differ from final invoice; verify with invoice retrieval instead of trusting earlier price output.\n\n"
    "Useful recovery strategies / action orderings:\n"
    "- Authenticate first, then gather IDs, then perform booking or cancellation, then verify with invoice retrieval.\n\n"
    "Reusable heuristics:\n"
    "- Preserve every returned ID and token for later tool calls.\n"
    "- Re-check final state with a verification tool after any state-changing action.\n\n"
    "Another good example:\n"
    "Environment-specific facts / constraints:\n"
    "- Vehicle actions are state-dependent.\n"
    "- Fuel is tracked in gallons, tire pressure in psi, and climate temperature in Celsius.\n"
    "- Engine start requires all doors locked and the brake pedal pressed.\n\n"
    "Tool behavior patterns:\n"
    "- Status tools expose the current system state and should be used before critical actions.\n"
    "- Conversion tools are necessary when the user gives values in the wrong unit.\n\n"
    "Common failure modes / hidden preconditions:\n"
    "- Engine start fails if any door is unlocked.\n"
    "- Engine start fails if the brake pedal is not pressed.\n"
    "- Refueling with the wrong unit causes incorrect assumptions about capacity.\n\n"
    "Useful recovery strategies / action orderings:\n"
    "- Check status, satisfy preconditions, then perform the action, then verify state again.\n\n"
    "Reusable heuristics:\n"
    "- Before any state-changing action, identify the required preconditions explicitly.\n\n"
    "Now write the memo.\n\n"
)

SUPPORT_EXPLORATION_PREFIX = (
    "This is an early attempt in a new tool-calling environment. "
    "Try to solve the task, but also use careful exploration to learn tool behavior, constraints, and failure modes.\n\n"
)


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


def build_task_system_prompt(
    tools: list[dict],
    experience_summary: str | None = None,
    exploration_mode: bool = False,
) -> str:
    base_prompt = build_system_prompt_with_tools(tools)
    if exploration_mode:
        base_prompt = SUPPORT_EXPLORATION_PREFIX + base_prompt
    if not experience_summary:
        return base_prompt
    return (
        "You are given an environment experience memo from a previous task in the same tool-calling environment. "
        "Use it as heuristic guidance when it is relevant.\n\n"
        "# Environment Experience\n"
        f"{experience_summary.strip()}\n\n"
        f"{base_prompt}"
    )


def extract_summary_context_text(summary_output: str) -> str:
    text = (summary_output or "").strip()
    if not text:
        return ""

    if "</think>" in text:
        distilled = text.split("</think>")[-1].strip()
        if distilled:
            return distilled

    return text


def snapshot_state_items(involved_instances: dict[str, Any]) -> list[dict]:
    state_items = []
    for class_name, class_instance in involved_instances.items():
        if class_name in STATELESS_CLASSES:
            continue
        state_items.append(
            {
                "role": "state_info",
                "class_name": class_name,
                "content": _make_json_serializable(
                    {
                        key: deepcopy(value) for key, value in vars(class_instance).items() if not key.startswith("_")
                    }
                ),
            }
        )
    return state_items


def summarize_checker_result(checker: dict[str, Any]) -> str:
    if not checker:
        return "No checker output is available."
    if checker.get("valid"):
        return "The support trajectory successfully completed the task."

    error_type = checker.get("error_type", "unknown_error")
    details = checker.get("details", {}) or {}
    lines = [f"The support trajectory failed. Checker error type: {error_type}."]

    if error_type == "multi_turn:instance_state_mismatch":
        differences = details.get("differences", {}) or {}
        if differences:
            diff_items = []
            for attr_name, diff in list(differences.items())[:5]:
                diff_items.append(
                    f"{attr_name}: model={json.dumps(_make_json_serializable(diff.get('model', '')), ensure_ascii=False)} "
                    f"vs ground_truth={json.dumps(_make_json_serializable(diff.get('ground_truth', '')), ensure_ascii=False)}"
                )
            lines.append("Important state mismatches: " + "; ".join(diff_items))
    elif error_type == "multi_turn:execution_response_mismatch":
        missing_items = details.get("missing_items", []) or []
        if missing_items:
            lines.append(
                "Expected execution results that were missing: "
                + "; ".join(str(item) for item in missing_items[:5])
            )
    elif error_type == "multi_turn:empty_turn_model_response":
        lines.append("At least one turn that required tool use ended with an empty model action.")
    elif details:
        lines.append(json.dumps(_make_json_serializable(details), ensure_ascii=False)[:600])

    return "\n".join(lines)


def truncate_prefix_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text

    marker = "\n[trajectory truncated]\n"
    marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
    marker_budget = len(marker_ids)
    if max_tokens <= marker_budget + 8:
        clipped_ids = token_ids[:max_tokens]
        return tokenizer.decode(clipped_ids, skip_special_tokens=True)

    prefix_budget = max_tokens - marker_budget
    prefix_ids = token_ids[:prefix_budget]
    return tokenizer.decode(prefix_ids, skip_special_tokens=True) + marker


def build_summary_prompt_messages(
    tokenizer,
    trajectory_text: str,
    support_success: bool,
    checker: dict[str, Any],
    max_prompt_tokens: int,
) -> list[dict[str, str]]:
    checker_summary = summarize_checker_result(checker)
    support_outcome = "success" if support_success else "failure"

    user_prefix = (
        SUMMARY_USER_PROMPT_PREFIX
        + f"Support outcome: {support_outcome}\n\n"
        + "Support checker summary:\n"
        + f"{checker_summary}\n\n"
        + "Trajectory:\n"
    )

    reserved_tokens = len(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prefix},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    trajectory_budget = max(256, max_prompt_tokens - reserved_tokens - 256)
    clipped_trajectory = truncate_prefix_by_tokens(tokenizer, trajectory_text, trajectory_budget)

    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + clipped_trajectory},
    ]


def _format_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(chunks)


def _format_tool_responses(tool_responses: list[str]) -> str:
    chunks = []
    for content in tool_responses:
        chunks.append(
            "<|im_start|>user\n"
            "<tool_response>\n"
            f"{content}\n"
            "</tool_response>\n"
            "<|im_end|>\n"
        )
    return "".join(chunks)


def _format_state_items(state_items: list[dict]) -> str:
    if not state_items:
        return ""
    payload = []
    for item in state_items:
        payload.append(
            {
                "class_name": item.get("class_name"),
                "content": _make_json_serializable(item.get("content")),
            }
        )
    return f"<|im_start|>state_info\n{json.dumps(payload, ensure_ascii=False)}\n<|im_end|>\n"


def build_trajectory_text(system_prompt: str, inference_log: list[Any]) -> str:
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

    for item in inference_log:
        if isinstance(item, list):
            parts.append(_format_state_items(item))
            continue

        if not isinstance(item, dict):
            continue

        begin_of_turn_query = item.get("begin_of_turn_query", [])
        if begin_of_turn_query:
            parts.append(_format_messages(begin_of_turn_query))

        step_keys = [key for key in item if key.startswith("step_")]
        step_keys.sort(key=lambda key: int(key.split("_")[1]))
        for step_key in step_keys:
            step_events = item.get(step_key, [])
            assistant_output = None
            tool_responses = []
            for event in step_events:
                if event.get("role") == "assistant":
                    assistant_output = str(event.get("content", ""))
                elif event.get("role") == "tool":
                    tool_responses.append(str(event.get("content", "")))

            if assistant_output is not None:
                parts.append(f"<|im_start|>assistant\n{assistant_output}<|im_end|>\n")
            if tool_responses:
                parts.append(_format_tool_responses(tool_responses))

    return "".join(parts).rstrip() + "\n"
