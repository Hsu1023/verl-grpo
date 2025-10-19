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

import os
CURRENT_PATH = os.path.join(os.path.dirname(__file__), '../../..')
print(CURRENT_PATH)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=f"{CURRENT_PATH}/data/mathlighteval", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default=f"{CURRENT_PATH}/data/mathlighteval", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "DigitalLearningGmbH/MATH-lighteval"

    # dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
    test_dataset = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    instruction_following = "You should first think about the reasoning process in the mind and then provide me with the answer. And the answer should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$.' (without quotes) where ANSWER is just the final number or expression that solves the problem."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("solution")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": 'grpo_mathlighteval',
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

    new_cols = ["data_source", "prompt", "ability", "reward_model"]
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True,remove_columns=[c for c in train_dataset.column_names if c not in new_cols])
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True,remove_columns=[c for c in test_dataset.column_names if c not in new_cols])

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
