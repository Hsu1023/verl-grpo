

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # 双保险（可选）
from scripts.legacy_model_merger import FSDPModelMerger, MegatronModelMerger, ModelMergerConfig
import argparse
import uuid
import shutil
from datasets import Dataset
import json
from vllm import LLM, SamplingParams
from verl.utils.reward_score.math_reward import compute_score
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import torch
import gc
from tqdm import tqdm, trange
import os
os.environ["VLLM_LOG_LEVEL"] = "ERROR"  # 新进程生效更稳


def get_llm(ckpt, args):
    return LLM(
        model=ckpt,
        # tensor_parallel_size=args.tp_size if hasattr(args, 'tp_size') else 1,
        # distributed_executor_backend="external_launcher",
        gpu_memory_utilization=0.85,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_num_batched_tokens=args.max_length,
        trust_remote_code=True,
        seed=42,
        # verbose=False,
    )


def get_dataset(args):
    # print('val_dataset_str:', args.val_dataset)
    # exit(0)
    # val_datasets = json.loads(args.val_dataset)
    # print('val_dataset_str:', val_datasets)
    val_datasets = ['/u/haoboxu/work/verl/data/aime2024/test.parquet']
    ret = []
    for dataset in val_datasets:
        data = list(Dataset.from_parquet(dataset))
        if args.limit:
            data = data[:args.limit]
        ret.append({
            'name': dataset.split('/')[-2],
            'data': data,
        })
    return ret


def eval(llm, datasets, args):
    ret_results = {}
    for dataset in datasets:
        n = args.pass_k
        sampling_params = SamplingParams(
            n=n,
            temperature=0.6,
            top_p=0.95,
            top_k=0,
            max_tokens=args.max_length,
            logprobs=20
        )
        # kwargs=dict(
        #     logprobs=True,
        #     top_logprobs=20)
        # kwargs=dict(logprobs=20, )
        data = dataset['data']
        total = len(data)
        batch_size = min(args.batch_size, total)
        avg_scores = []
        pass_scores = []
        lengths = []
        cnt = 0
        cur_dataset = []
        import time
        t1 = time.time()
        for i in trange(0, total, batch_size):
            batch = data[i:i + batch_size]
            prompts = [ex['prompt'] for ex in batch]
            outputs = llm.chat(prompts, sampling_params=sampling_params)
            continue
        t2 = time.time()
        assert 0, t2-t1
        # ret_results[dataset]=cur_dataset
        ret_results[dataset['name']] = cur_dataset
                
        #         pass_scores.append(max(score))
        #         avg_scores += score
        #         lengths += gen_length
        # avg_score = sum(avg_scores) / len(avg_scores)
        # pass_score = sum(pass_scores) / len(pass_scores)
        # avg_length = sum(lengths) / len(lengths)
        # ret_results[dataset['name']] = {
        #     'avg_score': avg_score,
        #     'pass_score': pass_score,
        #     'avg_length': avg_length,
        # }
        # print(f"Model: {args.model_dir.split('/')[-1]}, Dataset: {dataset['name']}, Avg Score: {avg_score}, Pass Score: {pass_score}, Avg Length: {avg_length}")
        print(f"Model: {args.model_dir.split('/')[-1]}, Dataset: {dataset['name']}")
        
        # import pickle as pkl
        # with open(f"{args.model_dir.replace('/', '_')}_{dataset['name']}.pkl", "wb") as f:
        #     pkl.dump(cur_dataset, f)
    return ret_results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy Model Merger")

    parser.add_argument("--model_dir", type=str, default='Qwen/Qwen3-4B', help="Directory containing local checkpoints")

    # parser.add_argument("--val_dataset", type=str, default="/u/haoboxu/work/verl/data/aime2024/test.parquet", help="Path to validation dataset parquet file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_length", type=int, default=32000, help="Maximum sequence length")
    parser.add_argument("--pass_k", type=int, default=4, help="Number of samples to pass for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset")
    args = parser.parse_args()
    
    datasets = get_dataset(args)
    llm = get_llm(args.model_dir, args)
    eval_results = eval(llm, datasets, args)