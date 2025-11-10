import argparse
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
from tqdm import tqdm

def get_checkpoints(local_dir, add_zero=False, add_final=False):
    import os
    checkpoints = []
    for item in os.listdir(local_dir):
        item_path = os.path.join(local_dir, item)
        if os.path.isdir(item_path) and item.startswith("global_step_"):
            checkpoints.append(int(item.split('_')[-1]))
    if add_zero:
        checkpoints.append(0)
    checkpoints.sort()
    return checkpoints

def merge(args, checkpoint_id):
    # uuid_str = str(uuid.uuid4())
    local_dir = args.local_dir + f"/global_step_{checkpoint_id}/actor"
    target_dir = args.target_dir + f"/global_step_{checkpoint_id}"
    args.hf_model_config_path = args.hf_model_path
    
    config = ModelMergerConfig(
        operation=args.operation,
        backend=args.backend,
        local_dir=local_dir,
        hf_model_config_path=args.hf_model_config_path,
        target_dir=target_dir,
        hf_upload_path=args.hf_upload_path,
        private=args.private,
        test_hf_dir=args.test_hf_dir,
        tie_word_embedding=args.tie_word_embedding,
        is_value_model=args.is_value_model,
        hf_model_path=args.hf_model_path,
        # hf_upload=False,
    )
    merger = FSDPModelMerger(config)
    if config.backend == "fsdp":
        merger = FSDPModelMerger(config)
    elif config.backend == "megatron":
        merger = MegatronModelMerger(config)
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")
    
    merger.merge_and_save()
    print(f"Merged checkpoint {checkpoint_id} to {target_dir} successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legacy Model Merger")
    parser.add_argument("--operation", type=str, default="merge", choices=["merge"], help="Operation to perform")
    parser.add_argument("--backend", type=str, default="fsdp", choices=["fsdp", "megatron"], help="Backend type")
    parser.add_argument("--local_dir", type=str, required=True, help="Directory containing local checkpoints")
    parser.add_argument("--target_dir", type=str, default=None, help="Directory to save merged checkpoints")   
    parser.add_argument("--hf_model_path", type=str, default=None, help="Hugging Face model path")
    parser.add_argument("--hf_upload_path", type=str, default=None, help="Number of shards for Megatron backend")
    parser.add_argument("--private", type=bool, default=False, help="Number of shards for Megatron backend")
    parser.add_argument("--test_hf_dir", type=str, default=None, help="Directory to test the merged Hugging Face model")
    parser.add_argument("--tie_word_embedding", type=bool, default=False, help="Whether to tie word embeddings")
    parser.add_argument("--is_value_model", type=bool, default=False, help="Whether the model is a value model")
    parser.add_argument("--val_dataset", type=str, default="\{\}", help="Path to validation dataset parquet file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_length", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--pass_k", type=int, default=32, help="Number of samples to pass for evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per dataset")
    args = parser.parse_args()
    checkpoints = get_checkpoints(args.local_dir)
    for ckpt in checkpoints:
        merge(args, ckpt)