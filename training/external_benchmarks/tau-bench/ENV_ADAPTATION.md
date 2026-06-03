# Environment-Specific Adaptation Evaluation

This runner evaluates tau-bench with `0`, `1`, and `2` test-time support tasks.

- `0`: no adaptation; the query task uses the original environment wiki.
- `1`: one support task is solved first; its trajectory is summarized into an environment memo.
- `2`: two support tasks are solved first; both trajectories are summarized into one environment memo.

The memo is appended to the query agent's system/wiki prompt. Original `run.py` behavior is unchanged.

Example:

```bash
python run_env_adaptation.py \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-4o \
  --user-model-provider openai \
  --user-strategy llm \
  --adaptation-counts 0 1 2 \
  --max-concurrency 4
```

Outputs are written under `results_env_adaptation/`, including one JSON result file per adaptation count and one summary JSON.
