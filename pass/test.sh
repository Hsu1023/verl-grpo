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

LOCAL_DIR=/projects/bfne/qwen3-4b_base_grpo_1e-6_math4_16k_dapo
TARGET_DIR=/u/haoboxu/work/verl/merged_checkpoints/qwen3-4b_base_grpo_1e-6_math4_16k_dapo
BASE_MODEL_PATH=/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base
EVALUATE_ARGS="--max_length 16384 --pass_k 16 --batch_size 1 --val_dataset $VAL_FILES"

torchrun evaluate.py --model_dir /u/haoboxu/work/verl/checkpoints/merged_checkpoints/clip/global_step_300  ${EVALUATE_ARGS}