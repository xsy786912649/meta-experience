# Environment-Specific Adaptation Evaluation

This runner evaluates text-mode tau2/tau3 domains with `0`, `1`, and `2` test-time support tasks.

- `0`: no adaptation; the query task uses the original domain policy.
- `1`: one support task is solved first; its trajectory is summarized into an environment memo.
- `2`: two support tasks are solved first; both trajectories are summarized into one environment memo.

The memo is appended to the query agent's domain policy. Original `tau2 run` behavior is unchanged.

Example:

```bash
uv run python run_env_adaptation.py \
  --domain airline \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --adaptation-counts 0 1 2 \
  --num-tasks 5
```

Outputs are written under `data/env_adaptation/`, including one JSON result file per adaptation count and one summary JSON.
