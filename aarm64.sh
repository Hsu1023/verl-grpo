
## 1) Install vllm and PyTorch with CUDA 12.9 support
module load cuda-compat/12.9

export LD_LIBRARY_PATH=/sw/user/cudatoolkits/cuda-12.9-compat/lib64:$LD_LIBRARY_PATH
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install vllm==0.10.2

### 2) Install flash-attn

### 2.1) before building, make sure you first clone the repo and switch to the right commit
cd your/flash-attention/dir
git fetch --tags

git checkout v2.8.0.post2
git describe --tags --exact-match      # should print v2.8.0.post2
git submodule sync --recursive
git submodule update --init --recursive

module load gcc/11.4
conda install -y -c nvidia/label/cuda-12.9.0 cuda-nvcc=12.9.*
# export CONDA_PREFIX="/projects/bfmw/schen33/conda-envs/demo2"
# export PATH="$CONDA_PREFIX/bin:$PATH"
# export CUDA_HOME=$CONDA_PREFIX
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST="90a"
export CMAKE_CUDA_ARCHITECTURES="90"
export CC=gcc CXX=g++
export MAX_JOBS=8   # adjust to your memory capacity applied
export FLASH_ATTENTION_FORCE_BUILD=1   # some versions honor this

# Clean any leftovers
rm -rf build/ dist/ *.egg-info ~/.cache/torch_extensions/*

python -m pip install -U pip wheel setuptools ninja packaging
pip install --no-build-isolation --no-cache-dir -v .

## 2.2) Make sure PyTorch libs are found first at runtime (otherwise FA may fail to load)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# This creates a file and adds the export command to it. You need reactivate the env after this.
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh


### 3) Verify
source ~/.bashrc
conda activate your-env-name
python - <<'PY'
import torch, flash_attn, vllm
print("FA OK;", torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY