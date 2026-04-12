from numbers import Number

import numpy as np


def _normalize_reward(extra_info):
    if extra_info is None:
        return [-0.001]

    if isinstance(extra_info, dict):
        return _normalize_reward(extra_info.get("reward", [-0.001]))

    if isinstance(extra_info, np.ndarray):
        if extra_info.ndim == 0:
            return _normalize_reward(extra_info.item())
        return [float(x) for x in extra_info.tolist()]

    if isinstance(extra_info, (list, tuple)):
        return [float(x) for x in extra_info]

    if isinstance(extra_info, Number):
        return [float(extra_info)]

    return [-0.001]


def compute_score(data_source, solution_str, ground_truth, extra_info):
    reward = _normalize_reward(extra_info)
    result = {"score": reward, "pred": ""}
    if isinstance(extra_info, dict):
        for key, value in extra_info.items():
            if key == "reward":
                continue
            result[key] = value
    return result
