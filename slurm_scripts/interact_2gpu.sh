srun --account=bfne-dtai-gh --partition=ghx4-interactive \
  --nodes=1 --gpus-per-node=2 --tasks=1 \
  --tasks-per-node=1 --cpus-per-task=8 --mem=250G \
  --pty bash
