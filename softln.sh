# for i in 'model-00001-of-00005.safetensors' 'model-00003-of-00005.safetensors' 'model-00005-of-00005.safetensors' 'tokenizer_config.json' 'vocab.json' 'model-00002-of-00005.safetensors' 'model-00004-of-00005.safetensors' 'model.safetensors.index.json' 'tokenizer.json' 'generation_config.json'; do
#     ln -s /u/haoboxu/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4/${i} /u/haoboxu/work/verl/qwen_probe/Qwen3-8B-Base/${i}
# done

# for i in 'model.safetensors' 'tokenizer_config.json' 'vocab.json' 'model.safetensors.index.json' 'tokenizer.json' 'generation_config.json'; do
#     ln -s /u/haoboxu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1/${i} /u/haoboxu/work/verl/qwen_probe/Qwen3-1.7B-Base/${i}
# done

for i in 'model.safetensors' 'tokenizer_config.json' 'vocab.json' 'model.safetensors.index.json' 'tokenizer.json' 'generation_config.json'; do
    ln -s /u/haoboxu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/${i} /u/haoboxu/work/verl/qwen_probe/llama-3.2-1b/${i}
done