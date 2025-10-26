srun --account=bfne-dtai-gh --partition=ghx4 \
  --nodes=1 --gpus-per-node=1 --tasks=1 \
  --tasks-per-node=1 --cpus-per-task=8 --mem=200G --time=04:00:00 \
  --pty bash 
