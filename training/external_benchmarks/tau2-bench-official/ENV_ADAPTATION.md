# Environment-Specific Adaptation Evaluation

This runner evaluates text-mode tau2/tau3 domains with `0`, `1`, and `2` test-time support tasks.

- `0`: no adaptation; the query task uses the original domain policy.
- `1`: one support task is solved first; its trajectory is summarized into an environment memo.
- `2`: two support tasks are solved first; both trajectories are summarized into one environment memo.

The memo is appended to the query agent's domain policy. Original `tau2 run` behavior is unchanged.

Example:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export AGENT_API_BASE="http://127.0.0.1:8000/v1"
export AGENT_API_KEY="EMPTY"

AGENT_LLM="openai/qwen3-8b" \
USER_LLM=openrouter/deepseek/deepseek-v4-flash \
SUMMARY_LLM="openai/qwen3-8b" \
DOMAIN=airline \
ADAPTATION_COUNTS="0 1 2" \
NUM_TASKS=5 \
uv run ../run_tau2_bench_env_adaptation.sh
```

The user simulator uses `USER_LLM`. The query/support agent and memo summarizer use
`AGENT_LLM` and `SUMMARY_LLM`; set those to your vLLM-served model when evaluating
your own model.

Do not put API keys in this file. Set `OPENROUTER_API_KEY` in the shell or job
environment on the machine that runs the benchmark.

Outputs are written under `data/env_adaptation/`, including one JSON result file per adaptation count and one summary JSON.
