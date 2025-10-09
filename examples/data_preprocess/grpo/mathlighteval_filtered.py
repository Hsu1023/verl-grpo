# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/yichen/verl/data/mathlighteval", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/home/yichen/verl/data/mathlighteval", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--aimed_level_list", default="5", help="The levels to include in the filtered dataset, separated by commas."
    )

    args = parser.parse_args()
    args.aimed_level_list = [int(x) for x in args.aimed_level_list.split(',')]
    print("Aimed levels: ", args.aimed_level_list)
    local_dataset_path = args.local_dataset_path

    data_source = "DigitalLearningGmbH/MATH-lighteval"
    
    

    # dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    test_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    instruction_following = "You should first think about the reasoning process in the mind and then provide me with the answer. And the answer should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$.' (without quotes) where ANSWER is just the final number or expression that solves the problem."

    level_str = ''.join([str(x) for x in args.aimed_level_list])
    statistic = [0 for _ in range(5)]
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("solution")
            solution = extract_solution(answer_raw)

            
            data = {
                "data_source": f'grpo_mathlighteval_level{level_str}',
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
            }
            return data

        return process_fn
    
    def keep_fn(example, idx):
        match = re.search(r'Level (\d+)', example['level'])
        if match is None:
            return False
        level = int(match.group(1))
        statistic[level-1] += 1
        # print(statistic)
        return level in args.aimed_level_list

    # 先把不需要的样本删掉
    train_dataset = train_dataset.filter(keep_fn, with_indices=True)
    # print("Data statistic (level 1 to 5å): ", statistic)
    for i in range(1,5):
        statistic[i] += statistic[i-1]
    print("Data statistic (level 1 to 5): ", statistic)
    test_dataset = test_dataset.filter(keep_fn, with_indices=True)
    

    new_cols = ["data_source", "prompt", "ability", "reward_model"]
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True,remove_columns=[c for c in train_dataset.column_names if c not in new_cols])
    
    
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True,remove_columns=[c for c in test_dataset.column_names if c not in new_cols])

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    print(f"Number of training samples after filtering: {len(train_dataset)}")
    print(f"Number of testing samples after filtering: {len(test_dataset)}")
    train_dataset.to_parquet(os.path.join(local_save_dir, f"train_level{level_str}.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, f"test_level{level_str}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
