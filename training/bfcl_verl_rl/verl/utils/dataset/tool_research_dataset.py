import json
from verl.utils.dataset.rl_dataset import RLHFDataset
import verl.utils.torch_functional as verl_F
from tests.workers.rollout.handle_message_tool_calling import extract_json_objects, pad_messages, build_tools_prompt_think, build_tools_prompt_no_think, build_tools_prompt_no_think_my, build_tools_prompt_think_my
import random

class ToolResearchRLDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor):
        super().__init__(data_files, tokenizer, config, processor=None)  

    def __getitem__(self, item):
        row_dict = self.dataframe[item]
        messages = row_dict.get("messages")

        tools_content = row_dict.get("tools_kwargs")
        tools_format = extract_json_objects(tools_content)

        #system_prompt_pre = build_tools_prompt_think_my(tools_format)
        #system_prompt_pre = build_tools_prompt_no_think_my(tools_format) 

        #system_prompt_pre = build_tools_prompt_think(tools_format)
        system_prompt_pre = build_tools_prompt_no_think(tools_format) 

        #select_prompt = random.choice([build_tools_prompt_think, build_tools_prompt_no_think])
        #system_prompt_pre = select_prompt(tools_format)

        row_dict["raw_prompt"] = [
            {"role": "system", "content": system_prompt_pre},
            {"role": "user", "content": messages[1]["content"]},
        ]

        row_dict["tools_kwargs"] = tools_content
        row_dict["flag"] = row_dict.get("flag")
        row_dict["reward_reference"] = pad_messages(row_dict.get("reward_model"))
        row_dict["total_messages"] = pad_messages(row_dict.get("messages"))

        # we set a dummy_input_ids 
        raw_prompt = self.tokenizer.apply_chat_template(row_dict["raw_prompt"], tokenize=False, add_generation_prompt=True) 
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

        # add index for each prompt
        index = row_dict.get('index')
        row_dict["index"] = index
        
        return row_dict