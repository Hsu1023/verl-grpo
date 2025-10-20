export CUDA_VISIBLE_DEVICES=0,1
python -m vllm.entrypoints.openai.api_server --port 8000 --model /data/public_models/qwen2.5-1.5b-instruct