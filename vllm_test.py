# import multiprocessing as mp
# if mp.get_start_method(allow_none=True) != "spawn":
#     mp.set_start_method("spawn", force=True)
    
    
from vllm import LLM, SamplingParams
import json
import os
from tqdm import tqdm
from datasets import Dataset, load_dataset
from verl.utils.reward_score.math_reward import *
from transformers import AutoTokenizer
from typing import List, Tuple, Sequence
import openai
import os
# import multiprocessing as mp
# if mp.get_start_method(allow_none=True) != "spawn":
#     mp.set_start_method("spawn", force=True)
# # 
# # Set the environment variable export VLLM_LOGGING_LEVEL=DEBUG to turn on more logging.
# # Set the environment variable export CUDA_LAUNCH_BLOCKING=1 to know exactly which CUDA kernel is causing the trouble.
# # Set the environment variable export NCCL_DEBUG=TRACE to turn on more logging for NCCL.
# # Set the environment variable export VLLM_TRACE_FUNCTION=1. All the function calls in vLLM will be recorded. Inspect these log files, and tell which function crashes or hangs.
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["VLLM_TRACE_FUNCTION"] = "1"


if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="qwen3-1.7b", choices=['qwen3-1.7b', "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-4b"], help='model path')
    parser.add_argument('--data', type=str, default="math1", choices=['math5', 'math4', 'math1', 'dapo17k'],
                        help='data file path')
    parser.add_argument('--begin', type=int, default=0, help='begin idx')
    parser.add_argument('--end', type=int, default=100, help='end idx')
    parser.add_argument('--traces', type=int, default=10, help='number of traces to generate for each prompt')
    parser.add_argument('--port', type=int, default=8000, help='max tokens to generate')
    args = parser.parse_args()
    MODEL_PATH = '/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base'
    MAX_TOKENS = 1024  #16384 * 2
    RID = 0
    QID = 0
    PORT = args.port

    prompt = "Re-arranging, $x^2 - 5x - 14 \le 0$. The left-hand quadratic factors as $x^2 - 5x - 14 = (x - 7)(x + 2) \le 0$. Thus, $x-7$ and $x+2$ have opposite signs, so $-2 \le x \le 7$ and $\boxed{x \in [-2,7]}$."
    llm = LLM(model=MODEL_PATH, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
        n=1,
        seed=42,
        stop_at_boxed=True,
    )

    tk = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    request_outputs = llm.generate([prompt], sampling_params=sampling_params)
    print(request_outputs[0])