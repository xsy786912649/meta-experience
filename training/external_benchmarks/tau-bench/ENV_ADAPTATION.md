# Environment-Specific Adaptation Evaluation

This runner evaluates tau-bench with `0`, `1`, and `2` test-time support tasks.

- `0`: no adaptation; the query task uses the original environment wiki.
- `1`: one support task is solved first; its trajectory is summarized into an environment memo.
- `2`: two support tasks are solved first; both trajectories are summarized into one environment memo.

The memo is appended to the query agent's system/wiki prompt. Original `run.py` behavior is unchanged.

Example:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export AGENT_API_BASE="http://127.0.0.1:8000/v1"
export AGENT_API_KEY="EMPTY"

MODEL="openai/qwen3-8b" \
MODEL_PROVIDER=openai \
USER_MODEL=deepseek/deepseek-v4-flash \
USER_MODEL_PROVIDER=openrouter \
SUMMARY_MODEL="openai/qwen3-8b" \
SUMMARY_MODEL_PROVIDER=openai \
ENV_NAME=retail \
ADAPTATION_COUNTS="0 1 2" \
../run_tau_bench_env_adaptation.sh
```

The user simulator uses `USER_MODEL`. The query/support agent and memo summarizer use
`MODEL` and `SUMMARY_MODEL`; set those to your vLLM-served model when evaluating your
own model.

Do not put API keys in this file. Set `OPENROUTER_API_KEY` in the shell or job
environment on the machine that runs the benchmark.

Outputs are written under `results_env_adaptation/`, including one JSON result file per adaptation count and one summary JSON.
