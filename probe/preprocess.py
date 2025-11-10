import json
import numpy as np
from collections import defaultdict
from verl.utils.reward_score.math_reward import *
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

path = '/u/haoboxu/work/verl/conf_results_qwen3-4b_math5_traces10_.json' # pass @ 10: 0.769; avg @ 10: 0.753
# path = '/u/haoboxu/work/verl/conf_results_qwen3-4b_dapo17k_traces10_.json' # pass @ 10: 0.904; avg @ 10: 0.745
# path = '/u/haoboxu/work/verl/probe/sample_data.json'
dic = json.load(open(path, 'r'))

# json.dump(dic[:20], open('/u/haoboxu/work/verl/probe/sample_data.json', 'w'), indent=4)

correctness_dic = defaultdict(list)
data_set = []

def qwen3_template(messages, add_generation_prompt=False, tokenizer=None, tokenize=False, return_tensors=None):
    ret_message = ''
    for message in messages:
        if message['role'] == 'user':
            ret_message += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        elif message['role'] == 'assistant':
            ret_message += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
    if add_generation_prompt:
        ret_message += "<|im_start|>assistant\n"
    if tokenize and tokenizer is not None:
        if return_tensors == "pt":
            return tokenizer(ret_message, return_tensors="pt")["input_ids"][0]
        else:
            return tokenizer(ret_message)["input_ids"]
    return ret_message

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
# )
text_data = []
answer_start_data = []
label_data = []
index_data = []
for i in dic:
    if len(i['correctness']) == 0:
        continue
    correctness_dic[i['idx']].append(i['correctness'][-1] if len(i['correctness']) > 0 else False)
    messages = [
            i['prompt'][0],
            {"role": "assistant", "content": i['text']},
    ]
    # print(i['prompt'])
    prefix_ids = qwen3_template(
        messages[:-1],              # 只到 user；不包含最后这个 assistant message
        add_generation_prompt=True, # 让模板自动加上 assistant 的开头
        tokenizer=tokenizer,
        tokenize=True,
        return_tensors="pt",
    )
    full_ids = qwen3_template(
        messages,
        add_generation_prompt=False,
        tokenizer=tokenizer,
        tokenize=True,
        return_tensors='pt'
    )
    # 3) answer 的起始位置（在整个序列里的 token index）
    answer_start = prefix_ids.shape[0]
    answer_end   = full_ids.shape[0]
    # if answer_end <= 10000:
    #     print(i['text'])
    #     # print(answer_start)
    #     print('####')
    #     print(prefix_ids)
    #     print('####')
    #     print(full_ids)
    #     print('####')
    #     print(tokenizer.decode(prefix_ids))
    #     print('####')
    #     print(tokenizer.decode(full_ids))
    #     exit(0)

    text_data.append(qwen3_template(
        messages,
        add_generation_prompt=False
    ))
    answer_start_data.append(answer_start)
    label_data.append(i['correctness'][-1] if len(i['correctness']) > 0 else False)
    index_data.append(i['idx'])
    # data_set.append({
    #     'text': full_ids,
    #     'answer_start': answer_start,
    #     'position': i['position'],
    #     'label': i['correctness'][-1] if len(i['correctness']) > 0 else False
    # })
# text_data = torch.tensor(text_data, dtype=torch.long)
# answer_start_data = torch.tensor(answer_start_data, dtype=torch.long)
# label_data = torch.tensor(label_data, dtype=torch.float)
torch.save({
    'index': index_data,
    'text': text_data,
    'answer_start': answer_start_data,
    'label': label_data,
}, './probe_data_qwen3-4b_math5_traces10.pt')