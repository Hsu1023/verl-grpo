# # clip
# for step in 200; do
#     python scripts/legacy_model_merger.py merge  --backend fsdp  --local_dir  /u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-1.7b_base_grpo_16_dapo/global_step_$step/actor --target_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/qwen3_1.7b_baseline_dapo/global_step_$step --hf_model_path /u/haoboxu/work/verl/qwen_probe/Qwen3-1.7B-Base
# done


# dapo
# for step in 550 600; do
#     python scripts/legacy_model_merger.py merge  --backend fsdp  --local_dir  /u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-4b_base_grpo_1e-6_math4_16k_dapo/global_step_$step/actor --target_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/baseline/global_step_$step --hf_model_path /u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base
# done

# clip_0.5
for step in 200; do
    python scripts/legacy_model_merger.py merge  --backend fsdp  --local_dir  /u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-1.7b_base_grpo_1e-6_math4_16k_clip_0.5_hist_0.2_dapo/global_step_$step/actor --target_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/qwen3_1.7b_0.5_p0.2_dapo/global_step_$step --hf_model_path /u/haoboxu/work/verl/qwen_probe/Qwen3-1.7B-Base
done