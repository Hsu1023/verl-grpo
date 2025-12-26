import pickle as pkl

from datasets import Dataset
import numpy as np
from verl.utils.reward_score.math_reward import compute_score
from utils import *

# result_path = '/u/haoboxu/work/verl/pass/_u_haoboxu_work_verl_checkpoints_merged_checkpoints_qwen3_4b_0.5_global_step_100_amc23.pkl'
# result_path = '/u/haoboxu/work/verl/pass/_u_haoboxu_work_verl_checkpoints_merged_checkpoints_qwen3_4b_baseline_global_step_100_amc23.pkl'
result_path = '/u/haoboxu/work/verl/pass/_u_haoboxu_work_verl_checkpoints_merged_checkpoints_qwen3_1.7b_0.5_p0.4_global_step_300_aime2025.pkl'
# result_path = '/u/haoboxu/work/verl/pass/_u_haoboxu_work_verl_checkpoints_merged_checkpoints_qwen3_1.7b_baseline_global_step_150_amc23.pkl'
data = pkl.load(open(result_path, 'rb'))
# assert 0, len(data)
def add_score(dataset_name):
    def get_dataset():
        ret = []
        for dataset in [f'/u/haoboxu/work/verl/data/{dataset_name}/test.parquet']:
            data = list(Dataset.from_parquet(dataset))
            ret.append({
                'name': dataset.split('/')[-2],
                'data': data,
            })
        return ret

    dataset = get_dataset()[0]['data']

    prompt = [ex['prompt'][0]['content'] for ex in dataset]
    gt = [ex['reward_model']['ground_truth'] for ex in dataset]
    if all([1 if p == p2['prompt'] else 0 for p, p2 in zip(prompt, data)]):
        for d, g in zip(data, gt):
            d['score'] = [compute_score(gen, g, False)['acc'] for gen in d['text']]
            d['answer'] = g
    else:
        raise ValueError('mismatch')
    print(data[0]['score'])
    
    pkl.dump(data, open(result_path, 'wb'))
    exit(0)
            
# add_score('amc23')

def pass_k(k=16):
    # assert 0, len(data[0]['text'])
    pass_k = [any(d['score'][:k]) for d in data]
    print('pass_k', sum(pass_k)/len(pass_k))
# 

def weighted_majority_vote(answers, weights):
    """Perform weighted majority voting"""
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return voted_answer

def extract_answer(text):
            """Extract boxed answer from text"""
            if "boxed" in text:
                ans = text.split("boxed")[-1]
                if len(ans) == 0:
                    return ""
                elif ans[0] == "{":
                    stack = 1
                    a = ""
                    for c in ans[1:]:
                        if c == "{":
                            stack += 1
                            a += c
                        elif c == "}":
                            stack -= 1
                            if stack == 0:
                                break
                            a += c
                        else:
                            a += c
                else:
                    a = ans.split("$")[0].strip()
                return a.strip()
            return None
        
def deepconf():
    def get_min(ll):
        def gget_min(l):
            WINDOW_SIZE=1024
            return min([sum(l[i-WINDOW_SIZE+1:i+1])/WINDOW_SIZE for i in range(WINDOW_SIZE-1, len(l))]) if len(l) > WINDOW_SIZE else sum(l)/len(l)
        return [gget_min(l) for l in ll]
    min_confs = []
    # for i in [get_min(l) for l in [d['logprobs'] for d in data]]:
    #     min_confs += i
    for d in data:
        min_conf = get_min(d['logprobs'])
        d['min_conf'] = min_conf
        min_confs += min_conf
    p = float(np.percentile(min_confs, 10))
    
    def cal_single(d):
        answers = []
        weights = []
        texts, ans = d['text'], d['answer']
        confs = d['min_conf']
        for text, conf in zip(texts, confs):
            if conf > p:
                answers.append(extract_answer(text))
                weights.append(conf)
        pred = weighted_majority_vote(answers, weights)
        def math_equal(a, b):
            from verl.utils.reward_score.math_reward import is_equiv
            return is_equiv(a, b)
        acc = math_equal(pred, ans)
        return acc
    score = [cal_single(d) for d in data]
    
    print('deepconf', sum(score)/len(score))
    
def majority():
    def cal_single(d):
        
        answers = [extract_answer(text) for text in d['text']]
        from collections import Counter
        answer_counts = Counter(answers)
        voted_answer, _ = answer_counts.most_common(1)[0]
        def math_equal(a, b):
            from verl.utils.reward_score.math_reward import is_equiv
            return is_equiv(a, b)
        acc = math_equal(voted_answer, d['answer'])
        return acc
    score = [cal_single(d) for d in data]
    
    print('majority', sum(score)/len(score))

def ours():
    def cal_single(d):
        # answers = [extract_answer(text) for text in d['text']]
        # weights = [_[0] for _ in d['probe_logits']]
        answers = []
        weights = []
        # print( d['probe_logits'])
        for text, probe_logit in zip(d['text'], d['probe_logits']):
            ans = extract_answer(text)
            # if probe_logit[0] > 1e-3:
            if True:
                answers.append(ans)
                weights.append(probe_logit[0])
        weights = np.array(weights)
        # answers, weights = weights_power_smooth(answers, weights)
        # answers, weights = weights_prob_flatten(answers, weights)
        # answers, weights = weights_winsorize(answers, weights)
        answers, weights = weights_rank_map(answers, weights, a=0.2, b=0.8)
        # answers, weights = weights_flatten_to_target_ess(answers, weights)
        
        
        
        pred = weighted_majority_vote(answers, weights)
        def math_equal(a, b):
            from verl.utils.reward_score.math_reward import is_equiv
            return is_equiv(a, b)
        acc = math_equal(pred, d['answer'])
        return acc
    score = [cal_single(d) for d in data]
    print('ours', sum(score)/len(score))

pass_k()
majority()
deepconf()
ours()