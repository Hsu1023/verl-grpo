export CUDA_VISIBLE_DEVICES=2,3
python -m vllm.entrypoints.openai.api_server --port 8000 --model /data/yichen/wyc/qwen2.5-1.5b-instruct