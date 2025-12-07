
# for step in 50 100 150; do
for step in 50 100; do
    # python scripts/legacy_model_merger.py merge  --backend fsdp  --local_dir  /u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-4b_base_grpo_1e-6_math4_16k_dapo/global_step_$step/actor --target_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/baseline/global_step_$step --hf_model_path Qwen/Qwen3-4B-Base
    python scripts/legacy_model_merger.py merge  --backend fsdp  --local_dir  /u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-4b_base_grpo_1e-6_math4_16k_dapo/global_step_$step/actor --target_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/baseline/global_step_$step --hf_model_path Qwen/Qwen3-4B-Base
done