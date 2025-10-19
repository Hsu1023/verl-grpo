export TORCH_CUDA_ARCH_LIST="90a"
export CMAKE_CUDA_ARCHITECTURES="90"
export CC=gcc CXX=g++
export SETUPTOOLS_SCM_PRETEND_VERSION=0.10.2

# export PIP_NO_BUILD_ISOLATION=1

rm -rf build/ dist/ *.egg-info ~/.cache/torch_extensions/*
VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -v -e .