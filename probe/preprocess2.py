import json
import numpy as np
from collections import defaultdict
from verl.utils.reward_score.math_reward import *
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm, trange
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

PROBE_STOP_TOKEN_IDS = [0, 3, 13, 30, 197, 198, 201, 271, 319, 568, 624, 936, 1773, 4292, 4894, 5267, 6313, 7810, 9338, 11319, 11843, 12947, 14085, 15087, 23586, 25046, 26126, 27275, 31716, 41295, 49964, 52402, 55807, 73594, 89478, 94280]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dapo17k', choices=['math5', 'dapo17k'])
args = parser.parse_args()
path = './probe_data_qwen3-4b_math5_traces10.pt' # pass @ 10: 0.769; avg @ 10: 0.753
path = './probe_data_qwen3-4b_dapo17k_traces10.pt' # pass @ 10: 0.904; avg @ 10: 0.745
path = './probe_data_qwen3-4b_' + args.data + '_traces10.pt'
dic = torch.load(path)
text_data = dic['text']
answer_start_data = dic['answer_start']
label_data = dic['label']
index_data = dic['index']

correctness_dic = defaultdict(list)
data_set = []

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# print(tokenizer.encode("Hello, how are you?"))
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

for (text, answer_start, label, idx) in zip(text_data, answer_start_data, label_data, index_data):
    data_set.append({
        'idx': idx,
        'text': text,
        'answer_start': answer_start,
        'label': label,
    })

# batch_size = 2
# for data in tqdm(data_set):
# data_set = data_set[:10]  # for testing
with torch.no_grad():
    # for i in trange(0, len(data_set), batch_size):
    for data in tqdm(data_set):
        # print(data['text'])
        inputs = tokenizer(data['text'], return_tensors="pt").to(model.device)
        # print([tokenizer.decode(inputs['input_ids'][0][-1:])])
        # print([tokenizer.decode(inputs['input_ids'][0][-2:-1])])
        # exit(0)
        total_length = inputs['input_ids'].shape[1]
        data['answer_end'] = total_length
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        last_layer_hidden_states = outputs.hidden_states[-1]  
        last_token_hidden_states = last_layer_hidden_states[0, :, :].float().cpu()
        # print(last_token_hidden_states.shape, total_length)  # (1, seq_len, hidden_size)
        # data['last_hidden_state'] = last_token_hidden_states[-1]
        
        # hidden_states_pos = torch.tensor([-2, -1, data['answer_start'], (total_length - data['answer_start'])//4, (total_length - data['answer_start'])//2, (total_length - data['answer_start']) * 3 //4, min(64, total_length-2), min(128, total_length-2), min(256, total_length-2), min(512, total_length-2), min(1024, total_length-2), min(2048, total_length-2), min(4096, total_length-2), min(8192, total_length-2)] + random.sample(list(range((data['answer_start']), total_length)), 5))
        hidden_states_pos = torch.tensor([total_length-2, total_length-1, min(data['answer_start']+32, total_length-2), min(data['answer_start']+64, total_length-2), min(data['answer_start']+128, total_length-2), min(data['answer_start']+256, total_length-2), min(data['answer_start']+512, total_length-2), min(data['answer_start']+1024, total_length-2), min(data['answer_start']+2048, total_length-2), min(data['answer_start']+4096, total_length-2)])
        hidden_states_list = last_token_hidden_states[hidden_states_pos] # (10, hidden_size)
        data['hidden_state'] = hidden_states_list  # (10, hidden_size)
        data['hidden_state_pos'] = hidden_states_pos

        # 在每个采样位置（>=0）向左找到最近的 stop token，并提取其 hidden state
        stop_token_ids = torch.tensor(PROBE_STOP_TOKEN_IDS, device=model.device)
        stop_mask = torch.isin(inputs['input_ids'][0], stop_token_ids)
        stop_positions = torch.nonzero(stop_mask, as_tuple=False).squeeze(-1).cpu()
        if stop_positions.dim() == 0:
            stop_positions = stop_positions.view(1)
        stop_positions = stop_positions.clamp_max(total_length - 1)

        # 使用全部采样点（含 -2/-1），保证 stop_hidden_state 长度与 hidden_states_pos 对齐
        target_pos = hidden_states_pos
        stop_pos_for_targets = []
        stop_hidden_for_targets = []
        hidden_size = last_token_hidden_states.shape[-1]
        for tgt in target_pos.tolist():
            candidates = stop_positions[stop_positions <= tgt]
            if candidates.numel() > 0:
                pos = candidates.max().item()
                stop_pos_for_targets.append(pos)
                stop_hidden_for_targets.append(last_token_hidden_states[pos])
            else:
                # 若左侧没有 stop token，则返回该采样点自身的 hidden state
                stop_pos_for_targets.append(tgt)
                stop_hidden_for_targets.append(last_token_hidden_states[tgt])

        data['stop_hidden_state'] = torch.stack(stop_hidden_for_targets, dim=0)  # (len(target_pos), hidden_size)
        data['stop_hidden_state_pos'] = torch.tensor(stop_pos_for_targets, dtype=torch.long)  # (len(target_pos),)
        # assert 0, (data['answer_start'], data['stop_hidden_state_pos'],inputs['input_ids'][0].tolist())
        # print(data)
        # exit(0)
to_be_saved = {
    'text': [data['text'] for data in data_set],
    'answer_start': torch.tensor([data['answer_start'] for data in data_set], dtype=torch.long),
    'answer_end': torch.tensor([data['answer_end'] for data in data_set], dtype=torch.long),
    'label': torch.tensor([data['label'] for data in data_set], dtype=torch.bool),
    'idx': torch.tensor([data['idx'] for data in data_set], dtype=torch.long),
    'hidden_state': torch.stack([data['hidden_state'] for data in data_set], dim=0),
    'hidden_state_pos': torch.stack([data['hidden_state_pos'] for data in data_set], dim=0),
    'stop_hidden_state': torch.stack([data['stop_hidden_state'] for data in data_set], dim=0),
    'stop_hidden_state_pos': torch.stack([data['stop_hidden_state_pos'] for data in data_set], dim=0),
}
torch.save(to_be_saved, f'./probe_data_qwen3-4b_{args.data}_traces10_with_hidden.pt')
print(f'Saved to ./probe_data_qwen3-4b_{args.data}_traces10_with_hidden.pt')
