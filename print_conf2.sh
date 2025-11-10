#!/bin/bash -l
#SBATCH -A bfne-dtai-gh
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH -J run_print_conf
#SBATCH -o outputs/%x.%j.out
#SBATCH -e outputs/%x.%j.err

set -euo pipefail
mkdir -p outputs

# ==== 可调参数 ====
PORT=8000                             # vLLM API 端口
WAIT_SECONDS="${WAIT_SECONDS:-180}"                 # 等待 60s 后并行跑 1.sh / 2.sh
VLLM_CMD="python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --port $PORT "   # vLLM 启动脚本
# SCRIPT1="python print_conf.py --port $PORT --model qwen3-4b --traces 10 --data math4 --begin 250 --end 500"
# SCRIPT2="python print_conf.py --port $PORT --model qwen3-4b --traces 10 --data math1 --begin 0 --end 500"
SCRIPT1="python print_conf.py --port $PORT --model qwen3-4b --traces 10 --data math5 --begin 20 --end 130"
SCRIPT2="python print_conf.py --port $PORT --model qwen3-4b --traces 10 --data math5 --begin 130 --end 240"
SCRIPT3="python print_conf.py --port $PORT --model qwen3-4b --traces 10 --data math5 --begin 240 --end 350"
VLLM_VISIBLE="0"
# ===============

echo "[`date '+%F %T'`] activate conda"
conda activate verl

# 1) 启动 vLLM（后台），不使用 srun
#    如需限定用哪些卡，给 CUDA_VISIBLE_DEVICES 赋值；否则继承 sbatch 分到的设备
if [ -n "${VLLM_VISIBLE}" ]; then
  echo "[`date '+%F %T'`] Launch vLLM with CUDA_VISIBLE_DEVICES=${VLLM_VISIBLE}"
  ( export CUDA_VISIBLE_DEVICES="${VLLM_VISIBLE}"; conda activate verl;  ${VLLM_CMD} ) \
    > outputs/vllm_server_2.log 2>&1 &
else
  echo "[`date '+%F %T'`] Launch vLLM with inherited CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
  bash -lc "${VLLM_CMD}" > outputs/vllm_server_2.log 2>&1 &
fi
VLLM_PID=$!
echo "[`date '+%F %T'`] vLLM PID=${VLLM_PID} (logs: outputs/vllm_server_2.log)"

cleanup() {
  echo "[`date '+%F %T'`] Cleaning up..."
  if ps -p "$VLLM_PID" >/dev/null 2>&1; then
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    sleep 5
    ps -p "$VLLM_PID" >/dev/null 2>&1 && kill -9 "$VLLM_PID" || true
  fi
}
trap cleanup EXIT

# 2) 固定等待（例如 60s 让服务完成加载）
echo "[`date '+%F %T'`] Sleeping ${WAIT_SECONDS}s for warmup..."
sleep "${WAIT_SECONDS}"

# 3) 并行运行 1.sh / 2.sh（非阻塞并行），并且明确不占 GPU
( conda activate verl; $SCRIPT1 > outputs/2_1.log 2>&1 ) & PID1=$!
( conda activate verl; $SCRIPT2 > outputs/2_2.log 2>&1 ) & PID2=$!
( conda activate verl; $SCRIPT3 > outputs/2_3.log 2>&1 ) & PID3=$!

# 4) 等待两脚本结束
FAIL=0
wait $PID1 || { echo "[`date '+%F %T'`] $SCRIPT1 failed, see outputs/2_1.log"; FAIL=1; }
wait $PID2 || { echo "[`date '+%F %T'`] $SCRIPT2 failed, see outputs/2_2.log"; FAIL=1; }
wait $PID3 || { echo "[`date '+%F %T'`] $SCRIPT3 failed, see outputs/2_3.log"; FAIL=1; }

[ $FAIL -eq 0 ] && echo "[`date '+%F %T'`] Both scripts finished OK." || exit 1
