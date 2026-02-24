# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a simple multi-turn dataset for testing
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm
import json
import re
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/ebs-basemodeling/siyuanxu/tool_synthetic/RL_training/verl_toolmock/data")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    # Create example conversations
    conversations = []

    original_data_test_list=["/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_1.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_2.jsonl",  "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_3_5.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_8.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_4.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_6.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/orignal_multi_trajectory_7.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/orignal_multi_trajectory_9.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_10.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/original_trajectory_11.jsonl"]
    complex_data_test_list= ["/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_userqueryhard_toolmock_trajectory_1.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_userqueryeasy_toolmock_trajectory_2.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_toolmock_trajectory_3_5.jsonl","/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_toolmock_trajectory_8.jsonl",  "/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_multiturn_toolmock_trajectory_4.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefiuseless_userqueryeasyhard_trajectory_6.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/multiapidefi_trajectory_7.jsonl", "/ebs-basemodeling/siyuanxu/tool_synthetic/data/multiapidefi_trajectory_9.jsonl","/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_multiturn_trajectory_10.jsonl","/ebs-basemodeling/siyuanxu/tool_synthetic/data/apidefi_multiturn_trajectory_11.jsonl"]

    simple_data_list=[]
    for original_data_test in original_data_test_list:
        with open(original_data_test, encoding="utf-8") as f:
            data = f.readlines()
            for i, line in enumerate(data):
                l_data = json.loads(json.loads(line))
                simple_data_list.append(l_data)
                if i==1000 and "3_5" in original_data_test:
                    break
                if i==2000 and "10" in original_data_test:
                    break
                if i==2000 and "11" in original_data_test:
                    break
                if i==2000 and "7" in original_data_test:
                    break
                if i==2000 and "4" in original_data_test:
                    break
                elif i==4000:
                    break


    complex_data_list=[]
    for complex_data_test in complex_data_test_list:
        with open(complex_data_test, encoding="utf-8") as f:
            data = f.readlines()
            for i, line in enumerate(data):
                l_data = json.loads(json.loads(line))
                complex_data_list.append(l_data)
                if i==1000 and "3_5" in complex_data_test:
                    break
                if i==2000 and "10" in complex_data_test:
                    break
                if i==2000 and "11" in complex_data_test:
                    break
                if i==2000 and "4" in complex_data_test:
                    break
                if i==2000 and "7" in complex_data_test:
                    break
                elif i==4000:
                    break

    for i, (data_line, data_simple) in enumerate(zip(complex_data_list,simple_data_list)):

        tool_source = data_line[0]["content"]
        tools_content = re.findall(r"<tools>(.*?)</tools>", tool_source, re.DOTALL)
        tools_content = tools_content[1].strip()

        conversations.append(
            { 
                "data_source": "mock_tool",
                "messages": data_line[0:-1],
                'reward_model': data_simple,  
                'index': i,
                'need_tools_kwargs': True,
                'flag': data_line[-1]["content"],
                'tools_kwargs': tools_content
            }
        )

    conversations = random.sample(conversations, len(conversations))

    # Create train and test datasets
    train_data = conversations[:26700]  # First 2 conversations for training
    test_data = conversations[26700:]

    print(train_data[0])

    # Create output directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Handle HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: HDFS support not available. Skipping HDFS copy.")

    # Print statistics
    print(f"Train dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    print(f"Data saved to {local_dir}")


if __name__ == "__main__":
    main()
