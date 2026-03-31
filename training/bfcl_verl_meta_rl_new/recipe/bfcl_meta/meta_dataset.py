import json
import uuid
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset


class BFCLMetaPairDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor):
        super().__init__(data_files, tokenizer, config, processor=None)

    @staticmethod
    def _decode_payload(row_dict):
        payload = row_dict["total_messages"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload

    def __getitem__(self, item):
        row_dict = dict(self.dataframe[item])
        pair_payload = dict(self._decode_payload(row_dict))
        pair_payload["meta_instance_id"] = uuid.uuid4().hex
        row_dict["raw_prompt"] = [{"role": "user", "content": "Write a compact environment experience memo."}]
        row_dict["reward_reference"] = None
        row_dict["total_messages"] = pair_payload
        row_dict["tools_kwargs"] = ""

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
