# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/yichen/verl/data/olympiadbench")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_path = "math-ai/olympiadbench"
    dataset = datasets.load_dataset(data_path, "default", split="test")

    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn():
        def process_fn(example, idx):
            
            # question_raw = example.pop("prompt")
            question = example.pop("question")

            question = question + " " + instruction_following
            solution = example.pop("final_answer")[0]
            for key in ['id', 'question', 'solution', 'final_answer', 'context', 'image_1',
                'image_2', 'image_3', 'image_4', 'image_5', 'modality', 'difficulty',
                'is_multiple_answer', 'unit', 'answer_type', 'error', 'question_type',
                'subfield', 'subject', 'language']:
                if key in example:
                    example.pop(key)
            print(solution)
            data = {
                # "data_source": example['data_source'],
                
                "data_source": 'grpo_olympiadbench',
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
    dataset = dataset.map(function=make_map_fn(), with_indices=True,remove_columns=[c for c in dataset.column_names if c not in new_cols])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
