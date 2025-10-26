import json
import os
import pickle
import time
from collections import Counter
from datetime import datetime

import numpy as np
import openai
# from dynasor.core.evaluator import math_equal
from tqdm import tqdm
from datasets import Dataset, load_dataset
from verl.utils.reward_score.math_reward import *
from transformers import AutoTokenizer

from typing import List, Tuple, Sequence

def find_all_boxed_spans(s: str) -> List[Tuple[int, int]]:
    """
    返回所有 \boxed{...} 和 \fbox{...} 的字符区间列表 [(start, end), ...]，
    其中 end 为“开区间”（即切片可用 s[start:end]）。
    仅接受紧跟 '{' 的形式：\boxed{...} / \fbox{...}，支持嵌套花括号。
    """
    out: List[Tuple[int, int]] = []
    i = 0
    while i < len(s):
        j_boxed = s.find(r'\boxed', i)
        j_fbox  = s.find(r'\fbox', i)
        if j_boxed == -1 and j_fbox == -1:
            break
        if j_boxed == -1 or (j_fbox != -1 and j_fbox < j_boxed):
            idx = j_fbox
            macro_len = len(r'\fbox')
        else:
            idx = j_boxed
            macro_len = len(r'\boxed')

        arg_start = idx + macro_len
        # 不考虑空格：必须紧跟 '{'
        if arg_start >= len(s) or s[arg_start] != '{':
            i = idx + 1
            continue

        # 配对括号（支持嵌套）
        depth = 0
        k = arg_start
        while k < len(s):
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
                if depth == 0:
                    out.append((idx, k + 1))  # 区间是 [idx, k+1)
                    i = k + 1                 # 继续找下一个
                    break
            k += 1
        else:
            # 若括号不配对，直接结束（也可改成 i = idx + 1; continue）
            break
    return out


def confs_for_each_boxed(
    text: str,
    token_ids: Sequence[int],
    confs:     Sequence[float],
    tk,  # HuggingFace fast tokenizer
    *,
    require_id_match: bool = False,
) -> Tuple[List[List[float]], List[List[int]], List[Tuple[int, int]]]:
    """
    返回:
      - conf_groups:  每个 boxed 对应的 conf 列表（与 token 顺序一致）
      - idx_groups:   每个 boxed 对应的 token 索引列表
      - spans:        每个 boxed 的字符区间 [(start, end)]
    说明:
      - 需要使用 HF FastTokenizer，并开启 return_offsets_mapping=True。
      - 请确保与生成 token_ids/confs 时的设置一致（尤其 add_special_tokens）。
    """
    assert len(token_ids) == len(confs), "token_ids 与 confs 长度需一致。"

    spans = find_all_boxed_spans(text)  # 字符区间
    if not spans:
        return [], [], []

    # 用 tokenizer 取 offsets（字符区间），务必与产生 token_ids 的设置一致！
    enc = tk(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,   # 通常与你的 token_ids 对齐
    )
    offsets = enc["offset_mapping"]          # List[Tuple[int,int]]
    enc_ids  = enc["input_ids"]              # 重算的 ids

    if require_id_match:
        # 若你确信 token_ids 来自同一个 tk+设置，可开启此断言确保一一对应
        assert list(enc_ids) == list(token_ids), \
            "re-tokenize 的 input_ids 与传入的 token_ids 不一致，请确认 tokenizer 与参数（如 add_special_tokens）一致。"
    else:
        # 至少保证长度一致（有时不同 tokenizer 版本仍可长度一致）
        # print(len(enc_ids), len(token_ids))
        
        if len(enc_ids) != len(token_ids):
            print(f"re-tokenize 的长度与传入的 token_ids 不一致，请统一 tokenizer 与参数。{enc_ids[-10:]} vs {token_ids[-10:]}")
            return [], [], []

    conf_groups: List[List[float]] = []
    idx_groups:  List[List[int]]   = []

    # 对每个 boxed 的字符区间，找所有有“区间重叠”的 token
    for (s_start, s_end) in spans:
        idxs = []
        for ti, (a, b) in enumerate(offsets):
            # 跳过空区间（有些 tokenizer 可能产生 (x,x)）
            if a == b:
                continue
            # 判断 [a,b) 与 [s_start, s_end) 是否相交
            if a < s_end and b > s_start:
                idxs.append(ti)

        idx_groups.append(idxs)
        conf_groups.append([confs[i] for i in idxs])

    return conf_groups, idx_groups, spans

def get_min_confs(confs: Sequence[float]): # min mean confs for different lengths
    lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    min_confs = {}
    from itertools import accumulate
    accumulated_lengths = list(accumulate(confs))
    for length in lengths:
        min_conf = float('inf')
        if length > len(confs):
            min_confs[length] = accumulated_lengths[-1] / len(confs)
            continue
        for i in range(len(confs) - length + 1):
            if i == 0:
                total = accumulated_lengths[i + length - 1]
            else:
                total = accumulated_lengths[i + length - 1] - accumulated_lengths[i - 1]
            mean_conf = total / length
            if mean_conf < min_conf:
                min_conf = mean_conf
        min_confs[length] = min_conf
    # return [min_confs[length] for length in lengths]
    return min_confs

def get_last_confs(confs: Sequence[float]):
    
    lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    last_confs = {}
    from itertools import accumulate
    for length in lengths: # last mean confs for different lengths
        if length > len(confs):
            last_confs[length] = sum(confs)/len(confs)
        else:
            total = sum(confs[-length:])
            mean_conf = total / length
            last_confs[length] = mean_conf
    return last_confs


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="qwen3-1.7b-base", choices=['qwen3-1.7b', "qwen3-1.7b-base"], help='model path')
parser.add_argument('--data', type=str, default="math5", choices=['math5', 'math4', 'math1', 'dapo17k'],
                    help='data file path')
parser.add_argument('--begin', type=int, default=0, help='begin idx')
parser.add_argument('--end', type=int, default=100, help='end idx')
args = parser.parse_args()
if args.model == 'qwen3-1.7b':
    args.model_path = "Qwen/Qwen3-1.7B"
elif args.model == 'qwen3-1.7b-base':
    args.model_path = "Qwen/Qwen3-1.7B-Base"
MODEL_PATH = args.model_path
tk = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
MAX_TOKENS = 12800
RID = 0
QID = 0
PORT = 8000
if args.data == 'math5':
    DATASET_FILE = "/u/haoboxu/work/verl/data/mathlighteval/train_level5.parquet"
elif args.data == 'math4':
    DATASET_FILE = "/u/haoboxu/work/verl/data/mathlighteval/train_level4.parquet"
elif args.data == 'math1':
    DATASET_FILE = "/u/haoboxu/work/verl/data/mathlighteval/train_level1.parquet"
elif args.data == 'dapo17k':
    DATASET_FILE = "/u/haoboxu/work/verl/data/dapo/dapo_17k.parquet"
ds = Dataset.from_parquet(DATASET_FILE)
records = list(ds)

WARMUP_TRACES = 1
# prompt = "Re-arranging, $x^2 - 5x - 14 \le 0$. The left-hand quadratic factors as $x^2 - 5x - 14 = (x - 7)(x + 2) \le 0$. Thus, $x-7$ and $x+2$ have opposite signs, so $-2 \le x \le 7$ and $\boxed{x \in [-2,7]}$."
client = openai.OpenAI(
    api_key="None",
    base_url=f"http://localhost:{PORT}/v1",
    timeout=None
)
results = []
for idx in tqdm(range(args.begin, args.end)):
    record = records[idx]
    messages = record['prompt']
    print(record['prompt'])
    solution = record['reward_model']['ground_truth']
    

    responses = client.completions.create(
        model=MODEL_PATH,
        # messages=messages,
        prompt=messages[0]['content'],
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        top_p=1.0,
        logprobs=True,
        top_logprobs=20,
        n=WARMUP_TRACES,
        extra_body={"top_k": 0},
        seed=42,
    )
    print(responses)

    def compute_confidence(logprobs):
        """Compute confidence score from logprobs."""
        confs = []
        raw_confs = []
        for lp in logprobs:
            confs.append(round(-sum([l.logprob for l in lp]) / len(lp), 3))
            raw_confs.append([l.logprob for l in lp])
        return confs, raw_confs

    for j in range(WARMUP_TRACES):
        choice = responses.choices[j]
        text = choice.message.content
        print(text)
        tokens = [t.token for t in choice.logprobs.content]
        confs, raw_conf = compute_confidence([t.top_logprobs for t in choice.logprobs.content])
        confs = confs[:-1]
        token_ids = [tk.encode(t)[0] for t in tokens[:-1]]
        conf_groups, idx_groups, spans = confs_for_each_boxed(text, token_ids, confs, tk)
        if conf_groups == []:
            print([])
            continue
        answers = [text[s:e] for (s,e) in spans]
        correctness = [compute_score(so, solution)['acc'] for so in answers]
        min_confs = get_min_confs(confs)
        last_confs = get_last_confs(confs)
        
        results.append({
            'idx': idx,
            'prompt': messages,
            "solution": solution,
            "correctness": correctness,
            "boxed_confs": conf_groups,
            "mean_boxed_confs": [sum(cg)/len(cg) for cg in conf_groups],
            "min_confs": min_confs,
            "last_confs": last_confs,
            'length_of_confs': len(confs),
            'spans': spans,
            "text": text,
            "raw_conf": raw_conf,
        })

JSON_FILE = f"conf_results_{args.model}_{args.data}.json"
if os.path.exists(JSON_FILE):
    data = json.load(open(JSON_FILE, 'r'))
    results = data + results
with open(JSON_FILE, 'w') as f:
    json.dump(results, f)