#!/usr/bin/env bash
# Build overlay_r2 on the RTX 4090 host into a separate tree; the published build is untouched.
set -euo pipefail
T=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809
T2=${T}-r2
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo
O=$R/revision_round2/code/overlay_r2
# Only the top-level build directories are excluded (a bare "build*" pattern also drops
# source files such as src/target/build_common.h).
rm -rf "$T2"
rsync -a --exclude "/build" --exclude "/build-*" "$T/" "$T2/"
for f in src/target/source/codegen_sycl.cc src/target/source/codegen_sycl.h src/runtime/sycl/sycl_module.cc \
         src/runtime/sycl/sycl_module.h src/runtime/sycl/sycl_device_api.cc src/runtime/sycl/sycl_common.h \
         src/target/target_kind.cc src/meta_schedule/utils.h python/tvm/relax/pipeline.py \
         src/tir/transforms/lower_thread_allreduce.cc python/tvm/ffi/cython/device.pxi ffi/cmake/Utils/CxxWarning.cmake; do
  cp "$O/$f" "$T2/$f"
done
mkdir -p "$T2/build-4090-r2"
cp "$T/build-4090/config.cmake" "$T2/build-4090-r2/config.cmake"
echo 'set(USE_OPENCL ON)' >> "$T2/build-4090-r2/config.cmake"
cd "$T2/build-4090-r2"
( for f in src/target/source/codegen_sycl.cc src/runtime/sycl/sycl_module.cc src/runtime/sycl/sycl_device_api.cc src/tir/transforms/lower_thread_allreduce.cc; do
    echo "$(md5sum < "$T2/$f" | cut -c1-12) $f"; done ) > source_md5.txt
export PATH=/usr/local/cuda-12.2/bin:$PATH
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 -DCMAKE_CUDA_ARCHITECTURES=52 > cmake.log 2>&1
ninja -j 32 > build.log 2>&1
ls -la --time-style=long-iso libtvm.so libtvm_runtime.so
echo BUILD_DONE
