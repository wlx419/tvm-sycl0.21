#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ONEDNN_SRC=${ONEDNN_SRC:-/mnt/data/wlx/workspace/oneDNN}
DPCPP_ROOT=${DPCPP_ROOT:-/mnt/data/wlx/sycl_workspace/DPC++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
CUDNN_ROOT=${CUDNN_ROOT:-/mnt/data/wlx/anaconda3/envs/looptune2/lib/python3.9/site-packages/nvidia/cudnn}
BUILD_DIR=${BUILD_DIR:-${ONEDNN_SRC}/build-codex-sycl-nvidia}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}}
CUDA_ARCH=${CUDA_ARCH:-sm_89}
GCC_VERSION=${GCC_VERSION:-11}
JOBS=${JOBS:-8}

C_COMPILER=${C_COMPILER:-${DPCPP_ROOT}/bin/clang}
CXX_COMPILER=${CXX_COMPILER:-${DPCPP_ROOT}/bin/clang++}
CUDNN_INCLUDE_DIR=${CUDNN_INCLUDE_DIR:-${CUDNN_ROOT}/include}
CUDNN_LIBRARY=${CUDNN_LIBRARY:-${CUDNN_ROOT}/lib/libcudnn.so.8}
BENCHMARK_SOURCE=${BENCHMARK_SOURCE:-${SCRIPT_DIR}/onednn_sycl_baseline.cpp}
BENCHMARK_BINARY=${BENCHMARK_BINARY:-${OUTPUT_DIR}/onednn_cuda_vendor_baseline}

host_include_flags=(
  -isystem "/usr/include/c++/${GCC_VERSION}"
  -isystem "/usr/include/x86_64-linux-gnu/c++/${GCC_VERSION}"
  -isystem "/usr/include/c++/${GCC_VERSION}/backward"
)
host_library_flags=(
  -L"/usr/lib/gcc/x86_64-linux-gnu/${GCC_VERSION}"
)

cmake -S "${ONEDNN_SRC}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_C_FLAGS="${host_library_flags[*]}" \
  -DCMAKE_CXX_FLAGS="${host_include_flags[*]} ${host_library_flags[*]}" \
  -DDNNL_CPU_RUNTIME=NONE \
  -DDNNL_GPU_RUNTIME=SYCL \
  -DDNNL_GPU_VENDOR=NVIDIA \
  -DDNNL_BUILD_TESTS=OFF \
  -DDNNL_BUILD_EXAMPLES=OFF \
  -DDNNL_BUILD_DOC=OFF \
  -DONEDNN_BUILD_GRAPH=OFF \
  '-DDNNL_ENABLE_PRIMITIVE=MATMUL;CONVOLUTION;SOFTMAX;REDUCTION' \
  -DDNNL_VERBOSE=ON \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_ROOT}" \
  -DCUDNN_INCLUDE_DIR="${CUDNN_INCLUDE_DIR}" \
  -DCUDNN_LIBRARY="${CUDNN_LIBRARY}"

cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
mkdir -p "${OUTPUT_DIR}"

"${CXX_COMPILER}" -O3 -std=c++17 \
  -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
  -Xsycl-target-backend "--offload-arch=${CUDA_ARCH}" \
  "${host_include_flags[@]}" "${host_library_flags[@]}" \
  -I"${ONEDNN_SRC}/include" -I"${ONEDNN_SRC}/src" -I"${BUILD_DIR}/include" \
  "${BENCHMARK_SOURCE}" -L"${BUILD_DIR}/src" -ldnnl \
  -Wl,-rpath,"${BUILD_DIR}/src" -o "${BENCHMARK_BINARY}"

printf '%s\n' "${BENCHMARK_BINARY}"
