#!/bin/bash -l
#SBATCH -A bfne-dtai-gh
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH -J qwen3-4b_base_grpo_1e-6_mcs
#SBATCH -o outputs/%x.%j.out
#SBATCH -e outputs/%x.%j.err

set -euo pipefail
set -x

export RAY_BACKEND_LOG_LEVEL=FATAL
export BASE_PATH=$HOME/work/verl
export TMPDIR=$BASE_PATH/tmp
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
save_path=$BASE_PATH/output
exp_name=qwen3-4b_base_grpo_1e-6_mcs
project_name='verl_grpo_example_gsm8k'


TRAIN_FILES=$BASE_PATH/data/mcs.parquet
VAL_FILES=$BASE_PATH/data/mcs_example.parquet

# start time
time1=$(date +%s)

mkdir -p $BASE_PATH/checkpoints/$exp_name

conda activate verl

python3 -m verl.trainer.main_ppo \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4  \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=7168 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.experiment_name=$exp_name \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=50 \
    trainer.total_epochs=2 "$@" 2>&1 | tee -a $BASE_PATH/checkpoints/$exp_name/train.log
    #  \
time2=$(date +%s)
time_diff=$((time2 - time1))
echo "Training time: $time_diff seconds"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
#     trainer.n_gpus_per_node=4 \
#     trainer.val_before_train=True \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$TRAIN_FILES \
#     "data.val_files=$VAL_FILES" \
#     data.train_batch_size=256 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     data.max_prompt_length=512 \
#     data.max_response_length=2048 \
#     actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/home/yichen/open-r1/qwen2.5-3b \
#     actor_rollout_ref.actor.optim.lr=5e-5 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.model.lora_rank=64 \
#     actor_rollout_ref.model.lora_alpha=32 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=5e-4 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=5e-4 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console"]' \
#     trainer.project_name=$project_name \
#     trainer.experiment_name=$exp_name \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.resume_mode=auto \
#     trainer.max_actor_ckpt_to_keep=1 \
#     trainer.max_critic_ckpt_to_keep=1 \
#     trainer.test_freq=20 \
#     trainer.total_epochs=3 "$@" 2>&1 | tee -a $BASE_PATH/checkpoints/$exp_name/train.log
