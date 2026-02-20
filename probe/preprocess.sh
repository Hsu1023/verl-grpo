#!/bin/bash -l
#SBATCH -A bfne-dtai-gh
#SBATCH -p ghx4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -o %x.out

conda activate verl
python preprocess2.py --data dapo17k
python preprocess2.py --data math5