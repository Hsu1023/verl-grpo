#!/bin/bash -l
#SBATCH -A bfne-dtai-gh
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -J run_print_conf
#SBATCH -o outputs/%x.%j.out
#SBATCH -e outputs/%x.%j.err

conda activate verl

BASE_PATH=/u/haoboxu/work/verl
aime2024_path=$BASE_PATH/data/aime2024/test.parquet
aime2025_path=$BASE_PATH/data/aime2025/test.parquet
amc23_path=$BASE_PATH/data/amc23/test.parquet
dapo17k_path=$BASE_PATH/data/amc23/test.parquet
olympiadbench_path=$BASE_PATH/data/olympiadbench/test.parquet
omnimath_path=$BASE_PATH/data/omnimath/test.parquet
math500_path=$BASE_PATH/data/math500/test.parquet
minerva_path=$BASE_PATH/data/minervamath/test.parquet
gsm8k_path=$BASE_PATH/data/gsm8k/test.parquet

VAL_FILES="[\"$aime2024_path\",\"$aime2025_path\",\"$amc23_path\"]"
# VAL_FILES="[\"$aime2024_path\"]"

# LOCAL_DIR=/u/haoboxu/work/verl/checkpoints/verl_examples/qwen3-4b_base_grpo_1e-6_math4_16k_dapo
LOCAL_DIR=/projects/bfne/qwen3-4b_base_grpo_1e-6_math4_16k_dapo
TARGET_DIR=/u/haoboxu/work/verl/checkpoints/merged_checkpoints/qwen3_4b_baseline
BASE_MODEL_PATH=/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base
EVALUATE_ARGS="--max_length 12800 --pass_k 16 --batch_size 1 --val_dataset $VAL_FILES"

# torchrun --standalone --nproc_per_node=1 merge.py --local_dir $LOCAL_DIR  --target_dir $TARGET_DIR --hf_model_path $BASE_MODEL_PATH

# torchrun evaluate.py --model_dir $BASE_MODEL_PATH ${EVALUATE_ARGS}

# ckpts=("200" "300" "400" "500" "600")
# ckpts=("600")
ckpts=("100")
for ckpt in "${ckpts[@]}"; do
    torchrun evaluate_test.py --model_dir $TARGET_DIR/global_step_$ckpt ${EVALUATE_ARGS}
done