def compute_score(data_source, solution_str, ground_truth, extra_info):
    if isinstance(extra_info, (list, tuple)):
        reward = [float(x) for x in extra_info]
    else:
        reward = [0.0]
    return {"score": reward, "pred": ""}

