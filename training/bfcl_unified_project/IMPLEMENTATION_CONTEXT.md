# Implementation Context (What Was Built and Why)

## 1) Original Request Context
You asked for a standalone project (outside `bfcl_multi_turn`) that reproduces the core behavior of:
- `run.py` (generate + evaluate)
- `export_trajectories.py` (pack trajectory text)

But unified into one flow:
- run inference
- evaluate success/failure
- immediately output per-sample result containing trajectory + success signal
- no extra stdout noise
- support running a single sample directly

Later, you requested further simplification:
- remove local vLLM startup/model-loading logic from code
- query an already-running external OpenAI-compatible vLLM server instead
- hardcode server/model/api settings
- support writing long output to file
- remove debug-only code after issue was resolved

---

## 2) Final Project Location
Standalone project is under:
- `/Users/xusiyuan/Desktop/meta-experience/environment/bfcl_unified_project`

This project is independent from `bfcl_multi_turn` runtime code.
It includes required dataset assets copied into this project:
- `data/`
- `data/possible_answer/`
- `data/multi_turn_func_doc/`
- `func_source_code/` (tool environment implementations)

---

## 3) Final Functional Goal Implemented
Current code provides a unified benchmark runner that does per entry:
1. load benchmark sample + tool docs + possible answers
2. run multi-turn tool-calling inference through external server
3. execute decoded tool calls in environment simulators (`func_source_code/*`)
4. evaluate success using BFCL-style state+response checking
5. build trajectory text in BFCL export style
6. emit one JSON result per sample:
   - `id`
   - `success`
   - `trajectory`

Optional: write outputs as JSONL into file instead of terminal printing.

---

## 4) Key Files and Responsibilities

### `unified_run.py`
Main entrypoint and orchestration.
- Loads entries/answers from `data/*`
- Supports full-category run or `--entry-id` single run
- Calls generation + evaluation + trajectory packing in one process
- Emits one JSON object per sample
- Supports `--output-file` for JSONL sink

### `qwen_fc.py`
Inference adapter.
- Builds prompt in the same XML tool-call style as previous pipeline
- Sends completion requests to external OpenAI-compatible vLLM server
- Parses `<tool_call>...</tool_call>` blocks
- Decodes model outputs into executable call strings

Important: local server/model startup code was removed as requested.

### `multi_turn_eval.py`
Execution and scoring logic.
- Executes decoded function calls against tool environments
- Maintains per-sample environment instances
- Performs BFCL-like validation:
  - state checker
  - response checker
- Returns final validity used as `success`

### `config.py`
Project constants.
- dataset paths
- category list
- model registry mapping
- class/module mapping
- step limit and prompt constants

### `io_utils.py`
JSONL load/write helpers.

### `func_source_code/*`
Environment/tool simulators copied into this standalone project and import-fixed to local paths.

---

## 5) Runtime Interface (Current)
The runner currently uses fixed inference backend settings in code (as requested):
- server base URL: `http://localhost:8010/v1`
- api key: `token-abc123`
- served model: `/data/zhimeng/model/Qwen3-8B`
- benchmark model registry fixed to: `Qwen/Qwen3-8B-FC`

So run command is simplified to e.g.:
- single sample: `--entry-id ...`
- or category run: `--categories ...`
- output sink: `--output-file ...`

No CLI flags are required for server/api/model selection now.

---

## 6) Output Contract
Per sample output object:
```json
{
  "id": "...",
  "success": true_or_false,
  "trajectory": "..."
}
```

When `--output-file` is given, results are written line-by-line as JSONL.

---

## 7) Important Behavioral Notes
1. Trajectory text is generated from `inference_log` in BFCL-like format:
   - optional system block extracted from first `inference_input`
   - `state_info` blocks
   - user/assistant/tool_response turns

2. To avoid empty trajectories due to serialization issues, trajectory building includes serializable fallback handling.

3. Exact string equality with historical trajectory exports is not guaranteed if live inference output differs (same environment, different model behavior/step choices).
   - In that case evaluation may still pass/fail independently of text equality.

---

## 8) Why This Was Built This Way
The implementation intentionally keeps:
- benchmark logic + stateful tool execution + scoring + trajectory packaging

And intentionally removes:
- local model loading
- local vLLM process startup/orchestration

This makes the runner lightweight and easy to embed into larger systems where inference is already served by a shared endpoint.

---

## 9) Integration-Ready Boundaries
For future integration into other codebases, the most reusable boundary is:
- `generate_single_entry(...)`
- `evaluate_single_entry(...)`
- `build_trajectory_text(...)`

These three can be imported or extracted into service-level modules while keeping external inference transport in `qwen_fc.py`.
