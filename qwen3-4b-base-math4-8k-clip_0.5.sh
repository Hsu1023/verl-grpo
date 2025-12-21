#!/bin/bash -l
#SBATCH -A bfne-dtai-gh
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=350G
#SBATCH --time=48:00:00
#SBATCH -J qwen3-4b_base_grpo_1e-6_math4_16k_clip_0.5
#SBATCH -o outputs/%x.%j.out
#SBATCH -e outputs/%x.%j.err

# set -euo pipefail
set -x
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO

export RAY_BACKEND_LOG_LEVEL=FATAL
export BASE_PATH=$HOME/work/verl
export TMPDIR=$BASE_PATH/tmp
export HYDRA_FULL_ERROR=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
save_path=$BASE_PATH/output
exp_name=qwen3-4b_base_grpo_1e-6_math4_16k_clip_0.5
project_name='verl_grpo_example_gsm8k'

aime2024_path=$BASE_PATH/data/aime2024/test.parquet
aime2025_path=$BASE_PATH/data/aime2025/test.parquet
amc23_path=$BASE_PATH/data/amc23/test.parquet
dapo17k_path=$BASE_PATH/data/amc23/test.parquet
olympiadbench_path=$BASE_PATH/data/olympiadbench/test.parquet
omnimath_path=$BASE_PATH/data/omnimath/test.parquet
math500_path=$BASE_PATH/data/math500/test.parquet
minerva_path=$BASE_PATH/data/minervamath/test.parquet
gsm8k_path=$BASE_PATH/data/gsm8k/test.parquet

TRAIN_FILES=$BASE_PATH/data/dapo17k/train.parquet

# start time
time1=$(date +%s)
VAL_FILES="['$aime2024_path', '$aime2025_path', '$amc23_path', '$olympiadbench_path', '$math500_path', '$minerva_path']"
# VAL_FILES="['$aime2024_path', '$aime2025_path', '$amc23_path', '$olympiadbench_path', '$minerva_path']"
# VAL_FILES="['$aime2024_path', '$aime2025_path']"

mkdir -p $BASE_PATH/checkpoints/$exp_name

conda activate verl

export VERL_AUTO_PADDING=1


    # +algorithm.cutoff=True \

python3 -m verl.trainer.main_ppo \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    "data.val_files=$VAL_FILES" \
    data.train_batch_size=16 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2  \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    +trainer.probe_m=0.5 \
    data.max_prompt_length=512 \
    data.max_response_length=7680 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +trainer.probe_max_init_value=1.0 \
    +trainer.probe_min_init_value=0.0 \
    "+algorithm.early_exit_grad=False" \
    actor_rollout_ref.model.path=/u/haoboxu/work/verl/qwen_probe/Qwen3-4B-Base \
    +reward_model.use_format_reward=False \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=True \
    +trainer.probe_warmup_steps=25 \
    +trainer.probe_stop_token_num=512 \
    actor_rollout_ref.actor.optim.probe_lr=0.01 \
    actor_rollout_ref.actor.probe_loss_coef=1.0 \
    +trainer.probe_momentum=0.95 \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.use_probe=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-3 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.experiment_name=$exp_name \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.resume_mode=True \
    trainer.test_freq=50 \
    "+trainer.pos_sample_list=[0.416015625, 0.8046875, 1.0, 1.0, 9.5367431640625e-06, 0.0023956298828125, 1.2759119272232056e-07, 0.51171875, 0.99609375, 0.6953125, 0.07275390625, 0.0002307891845703125, 1.2218952178955078e-05, 1.2218952178955078e-05, 1.2218952178955078e-05, 1.0, 1.0, 1.0, 1.0, 0.6953125, 4.731118679046631e-07, 0.0693359375, 0.0693359375, 0.0038299560546875, 0.0038299560546875, 0.0038299560546875, 0.99609375, 0.99609375, 0.86328125, 4.798173904418945e-06, 1.3709068298339844e-06, 1.2514647096395493e-09, 1.2759119272232056e-07, 3.774403012357652e-11, 1.955777406692505e-08, 1.955777406692505e-08, 1.0, 3.9637088775634766e-06, 1.0, 1.0, 1.3709068298339844e-06, 1.3709068298339844e-06, 1.2514647096395493e-09, 1.2514647096395493e-09, 1.2514647096395493e-09, 1.2514647096395493e-09, 1.2514647096395493e-09, 1.2514647096395493e-09, 3.774403012357652e-11, 0.5078125, 1.955777406692505e-08, 1.2514647096395493e-09, 1.955777406692505e-08, 6.007030606269836e-08, 3.528594970703125e-05, 0.1767578125, 1.0, 1.0, 0.55859375, 0.55859375, 1.318767317570746e-10, 2.1736923372372985e-10, 8.754432201385498e-08, 0.000732421875, 5.0961971282958984e-06, 6.628036499023438e-05, 0.86328125, 0.62890625, 0.62890625, 0.62890625, 7.963180541992188e-05, 1.6540288925170898e-06, 4.94765117764473e-09, 4.94765117764473e-09, 1.318767317570746e-10, 1.8533319234848022e-07, 8.754432201385498e-08, 2.9335764912906377e-30, 2.9335764912906377e-30, 5.0961971282958984e-06, 5.0961971282958984e-06, 0.000148773193359375, 0.000148773193359375, 0.000148773193359375, 0.98828125, 0.98828125, 1.0, 1.0, 1.0, 0.953125, 0.482421875, 6.845220923423767e-08, 6.845220923423767e-08, 1.7229467630386353e-08, 0.000148773193359375, 0.000148773193359375, 1.0, 0.41796875, 0.154296875, 1.0, 0.953125, 0.61328125, 6.628036499023438e-05, 1.7229467630386353e-08, 1.955777406692505e-08, 1.955777406692505e-08, 5.781650543212891e-06, 5.781650543212891e-06, 5.781650543212891e-06, 5.781650543212891e-06, 0.419921875, 0.419921875, 1.0, 1.0, 1.0, 1.0, 0.99609375, 0.99609375, 0.95703125, 0.95703125, 0.95703125, 1.0, 0.984375, 7.963180541992188e-05, 9.255018085241318e-09, 9.255018085241318e-09, 0.0002460479736328125, 6.5267086029052734e-06, 0.5546875, 0.5546875, 1.0, 1.0, 1.0, 1.0, 0.25390625, 7.963180541992188e-05, 0.0001583099365234375, 5.14984130859375e-05, 5.14984130859375e-05, 0.0004177093505859375, 2.3990869522094727e-06, 2.3990869522094727e-06, 0.0419921875, 0.0419921875, 0.0419921875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0001583099365234375, 5.14984130859375e-05, 5.14984130859375e-05, 5.14984130859375e-05, 0.0008544921875, 0.0004177093505859375, 3.268496584496461e-13, 5.4836273193359375e-05, 5.4836273193359375e-05, 5.4836273193359375e-05, 0.11279296875, 0.0419921875, 0.0419921875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.5299530029296875e-05, 5.3085386753082275e-08, 1.0477378964424133e-08, 1.0477378964424133e-08, 7.048583938740194e-11, 7.048583938740194e-11, 6.007030606269836e-08, 9.049472282640636e-11, 9.049472282640636e-11, 0.0005340576171875, 0.000179290771484375, 0.9765625, 0.9765625, 0.9765625, 4.5299530029296875e-05, 4.5299530029296875e-05, 7.566995918750763e-10, 2.2118911147117615e-08, 2.2118911147117615e-08, 1.996755599975586e-06, 1.1874362826347351e-08, 1.1874362826347351e-08, 0.02978515625, 8.307397365570068e-07, 0.828125, 0.828125]" \
    "+trainer.neg_sample_list=[0.25390625, 0.00016880035400390625, 0.00016880035400390625, 0.00016880035400390625, 0.00016880035400390625, 0.00016880035400390625, 0.00016880035400390625, 0.00016880035400390625, 0.0001583099365234375, 0.0001583099365234375, 1.895427703857422e-05, 0.0001583099365234375, 5.14984130859375e-05, 7.963180541992188e-05, 0.0004177093505859375, 0.0004177093505859375, 6.139278411865234e-06, 6.139278411865234e-06, 6.139278411865234e-06, 2.0236257114447653e-11, 2.3990869522094727e-06, 2.868473529815674e-07, 5.4836273193359375e-05, 5.4836273193359375e-05, 8.307397365570068e-07, 8.307397365570068e-07, 0.00070953369140625, 0.00070953369140625, 0.00070953369140625, 0.00070953369140625, 0.0419921875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.08056640625, 1.0, 1.0, 1.0, 1.0, 1.0, 0.98828125, 0.98828125, 0.99609375, 0.99609375, 0.99609375, 1.0, 1.0, 0.0001583099365234375, 0.0001583099365234375, 5.14984130859375e-05, 0.0004177093505859375, 0.0004177093505859375, 0.0004177093505859375, 6.139278411865234e-06, 1.3828277587890625e-05, 7.009506225585938e-05, 0.11572265625, 6.198883056640625e-05, 2.3990869522094727e-06, 4.839897155761719e-05, 5.4836273193359375e-05, 0.11279296875, 0.00070953369140625, 0.00070953369140625, 0.0419921875, 0.099609375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1552734375, 0.1552734375, 0.1552734375, 0.00186920166015625, 0.1875, 0.1875, 0.1875, 0.1875, 7.867813110351562e-06, 6.5267086029052734e-06, 6.5267086029052734e-06, 6.5267086029052734e-06, 6.5267086029052734e-06, 6.5267086029052734e-06, 4.5299530029296875e-05, 5.3085386753082275e-08, 1.0477378964424133e-08, 1.0477378964424133e-08, 1.0477378964424133e-08, 1.0477378964424133e-08, 7.048583938740194e-11, 7.048583938740194e-11, 2.514570951461792e-08, 0.048828125, 7.188646122813225e-09, 9.255018085241318e-09, 3.7670135498046875e-05, 9.049472282640636e-11, 9.049472282640636e-11, 9.049472282640636e-11, 9.049472282640636e-11, 4.4517219066619873e-07, 4.4517219066619873e-07, 4.4517219066619873e-07, 0.000606536865234375, 0.000606536865234375, 0.000606536865234375, 0.000606536865234375, 8.307397365570068e-07, 0.0005340576171875, 0.0003681182861328125, 0.0003681182861328125, 0.0004444122314453125, 0.0004444122314453125, 0.000179290771484375, 0.9765625, 0.9765625, 0.9765625, 0.9765625, 0.9765625, 0.1552734375, 0.1552734375, 0.1552734375, 5.699694156646729e-07, 0.1875, 0.1875, 0.1875, 0.359375, 6.5267086029052734e-06, 6.5267086029052734e-06, 1.318767317570746e-10, 4.5299530029296875e-05, 6.5267086029052734e-06, 6.5267086029052734e-06, 7.566995918750763e-10, 7.566995918750763e-10, 7.566995918750763e-10, 2.2118911147117615e-08, 2.2118911147117615e-08, 7.188646122813225e-09, 6.07222318649292e-07, 3.342393029015511e-11, 3.342393029015511e-11, 4.94765117764473e-09, 4.94765117764473e-09, 4.94765117764473e-09, 4.94765117764473e-09, 4.94765117764473e-09, 1.55717134475708e-06, 9.918585419654846e-08, 1.55717134475708e-06, 0.000606536865234375, 8.307397365570068e-07, 8.307397365570068e-07, 8.307397365570068e-07, 8.307397365570068e-07, 1.0058283805847168e-06, 1.0058283805847168e-06, 0.0003681182861328125, 0.0003681182861328125, 0.0003681182861328125, 0.0003681182861328125, 0.0004444122314453125, 0.0004444122314453125, 0.0004444122314453125, 0.0004444122314453125, 0.828125, 0.828125, 0.828125, 0.828125, 0.828125, 0.828125]" \
    trainer.total_epochs=1 "$@" 2>&1 | tee -a $BASE_PATH/checkpoints/$exp_name/train.log
    #  \
time2=$(date +%s)
time_diff=$((time2 - time1))

echo "Training time: $time_diff seconds"


echo "==== OOM check (dmesg) ===="
dmesg | grep -i -E "out of memory|killed process" | tail -n 50 || echo "no OOM lines or no permission"
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
