import json
from tqdm import trange
import random
import pickle

import random

def get_conf(confs):
    return [-sum(d)/len(d) for d in confs]

from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base", trust_remote_code=True, use_fast=True)
# data = json.load(open('/u/haoboxu/work/verl/conf_results_qwen3-4b_dapo17k_traces10_.json', 'r'))

# data = [d for d in data if len(d['correctness']) > 0]
# cor = [d['correctness'][-1] for d in data]
# err_case = [d for d in data if int(d['correctness'][-1])== 0 ]
# random.shuffle(err_case)
# pickle.dump(err_case[:20], open('dapo17k_error_cases.pkl', 'wb'))
data = pickle.load(open('dapo17k_error_cases.pkl', 'rb'))
for d in data:
    print(d['correctness'])
    # print(tk.tokenize(d['text']))
    print(d['text'])
    exit(0)
# pickle.dump(data, open('dapo17k_error_cases.pkl', 'wb'))
            # "text": text,
            # "raw_conf": raw_conf,