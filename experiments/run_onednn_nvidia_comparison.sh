#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/../results/onednn_nvidia_comparison}
RUNS=${RUNS:-5}
GPU_ID=${GPU_ID:-0}

ONEDNN_BINARY=${ONEDNN_BINARY:-${SCRIPT_DIR}/onednn_cuda_vendor_baseline}
TVM_PROBE=${TVM_PROBE:-${SCRIPT_DIR}/tvm_sycl_extended_probe.py}
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_ROOT=${DPCPP_ROOT:-/mnt/data/wlx/sycl_workspace/DPC++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
CUDNN_ROOT=${CUDNN_ROOT:-/mnt/data/wlx/anaconda3/envs/looptune2/lib/python3.9/site-packages/nvidia/cudnn}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
GCC_VERSION=${GCC_VERSION:-11}

export PATH="${DPCPP_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
export PYTHONPATH="${PYTHON_PACKAGES}:${TVM_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_LIBRARY_PATH="${TVM_BUILD}"
export LD_LIBRARY_PATH="${TVM_BUILD}:${DPCPP_ROOT}/lib:${DPCPP_ROOT}/lib64:${DPCPP_ROOT}/install/lib:${CUDNN_ROOT}/lib:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/${GCC_VERSION}:/usr/include/x86_64-linux-gnu/c++/${GCC_VERSION}:/usr/include/c++/${GCC_VERSION}/backward${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/${GCC_VERSION}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export ONEAPI_DEVICE_SELECTOR=cuda:gpu

mkdir -p "${OUT_DIR}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
  --format=csv,noheader >"${OUT_DIR}/environment.txt"
"${CXX:-${DPCPP_ROOT}/bin/clang++}" --version | head -n 1 >>"${OUT_DIR}/environment.txt"

ops=(matmul row_sum softmax conv2d_nhwc)
shapes=(256x256x256 1024x1024 512x512 8x32x32x16x32x3x3)

for run in $(seq 1 "${RUNS}"); do
  DNNL_VERBOSE=0 "${ONEDNN_BINARY}" \
    "${OUT_DIR}/onednn_cuda_vendor_run${run}.csv"

  for idx in "${!ops[@]}"; do
    "${PYTHON_BIN}" "${TVM_PROBE}" --worker \
      --op "${ops[$idx]}" --shape "${shapes[$idx]}" --wg-size 256 \
      --warmup 10 --measure 30 \
      >"${OUT_DIR}/tvm_sycl_run${run}_${ops[$idx]}.json"
  done
done

DNNL_VERBOSE=1 "${ONEDNN_BINARY}" \
  "${OUT_DIR}/onednn_cuda_vendor_verbose.csv" \
  >"${OUT_DIR}/onednn_cuda_vendor_verbose.log" 2>&1

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_onednn_nvidia_comparison.py" \
  --input-dir "${OUT_DIR}" --output-stem "${OUT_DIR}/summary"
