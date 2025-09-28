#!/usr/bin/env bash
set -euo pipefail

# 要执行的脚本
TARGET="/home/yichen/verl/qwen2.5-3b.sh"

while true; do
  rm -rf /home/yichen/verl/tmp/*
  ts=$(date '+%F %T')
  {
    echo "[$ts] === START ==="
    bash "$TARGET"
    rc=$?
    echo "[$(date '+%F %T')] === END (exit $rc) ==="
  } >> "run_hourly.log" 2>&1

  # 等 1 小时
  sleep 3600
done
