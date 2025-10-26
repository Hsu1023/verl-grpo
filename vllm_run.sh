# export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server --port 8000 --model Qwen/Qwen3-1.7B