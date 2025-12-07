# export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server --port 8000 --model /u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base
# python -m vllm.entrypoints.openai.api_server --port 8000 --model Qwen/Qwen3-1.7B-Base