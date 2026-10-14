#!/usr/bin/env bash
# Build the r2 sources a second time against DPC++ v7.1.1 (side-by-side toolchain evaluation).
# Same tree and configuration as build-4090-r2; only the SYCL compiler (runtime headers, libsycl,
# JIT device compiler) and the JIT scratch folder differ. The 646ab08 build is untouched.
set -euo pipefail
T2=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2
D=/mnt/data/wlx/sycl_workspace/DPC++-v7.1.1
B=$T2/build-4090-v711
mkdir -p $B /tmp/tvm_sycl_v711
sed -e "s#/mnt/data/wlx/sycl_workspace/DPC++/bin/tvm-sycl-clang++#$D/bin/tvm-sycl-clang++#" \
    -e 's#set(SYCL_TEMP_FOLDER "/tmp/tvm_sycl")#set(SYCL_TEMP_FOLDER "/tmp/tvm_sycl_v711")#' \
    $T2/build-4090-r2/config.cmake > $B/config.cmake
grep -n "USE_SYCL \|SYCL_TEMP_FOLDER" $B/config.cmake
cd $B
export PATH=/usr/local/cuda-12.2/bin:$PATH
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 -DCMAKE_CUDA_ARCHITECTURES=52 > cmake.log 2>&1
grep "Build with SYCL" cmake.log
nice -n 5 ninja -j 32 > build.log 2>&1 || { grep -m20 -B2 -A12 "error" build.log; exit 1; }
ldd libtvm_runtime.so | grep -i "sycl\|ur_"
ls -la --time-style=long-iso libtvm.so libtvm_runtime.so
echo V711_TVM_BUILD_DONE
