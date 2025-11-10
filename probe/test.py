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

correctness_dic = defaultdict(list)
data_set = []

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
text_data = []
answer_start_data = []
position_data = []
label_data = []
# correctness_dic[i['idx']].append(i['correctness'][-1] if len(i['correctness']) > 0 else False)
messages = [
        {"role": "user", "content": 'Hello, how are you?'},
        {"role": "assistant", "content": 'I am fine, thank you!'},
]
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


print(qwen3_template(messages, add_generation_prompt=False, tokenizer=tokenizer, tokenize=True))
print(qwen3_template(messages, add_generation_prompt=False, tokenizer=tokenizer, tokenize=True, return_tensors="pt"))


