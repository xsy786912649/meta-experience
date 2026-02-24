import json

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset

from recipe.bfcl_multiturn.bfcl_env import build_system_prompt_with_tools


class BFCLMultiTurnRLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor):
        super().__init__(data_files, tokenizer, config, processor=None)

    def __getitem__(self, item):
        row_dict = self.dataframe[item]
        payload = row_dict["total_messages"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        questions = payload["question"]
        tools = payload["function"]

        first_user = questions[0][0]["content"] if questions and questions[0] else ""
        row_dict["raw_prompt"] = [
            {"role": "system", "content": build_system_prompt_with_tools(tools)},
            {"role": "user", "content": first_user},
        ]

        reward_reference = row_dict.get("reward_model")
        if isinstance(reward_reference, str):
            reward_reference = json.loads(reward_reference)
        row_dict["reward_reference"] = reward_reference
        row_dict["total_messages"] = payload
        row_dict["flag"] = row_dict.get("id")

        raw_prompt = self.tokenizer.apply_chat_template(
            row_dict["raw_prompt"], tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["dummy_input_ids"] = input_ids[0]
        row_dict["dummy_attention_mask"] = attention_mask[0]
        return row_dict
