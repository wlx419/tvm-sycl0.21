#!/usr/bin/env bash
# Incremental rebuild of the r2 tree after overlay changes (sources and Python only; the
# server's own CMake modules are kept).
set -euo pipefail
T2=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2
O=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/code/overlay_r2
rsync -a --checksum --itemize-changes --exclude "/cmake/" --exclude "/README.md" --exclude "/CMakeLists.txt.patch" "$O/" "$T2/" | grep '^>f' || true
cd "$T2/build-4090-r2"
export PATH=/usr/local/cuda-12.2/bin:$PATH
cmake . > cmake_incr.log 2>&1
ninja -j 32 > build_incr.log 2>&1 || { grep -m20 -B2 -A12 "error" build_incr.log; exit 1; }
( cd "$T2" && find src/runtime/sycl src/target/source/codegen_sycl.cc src/tir/transforms/lower_thread_allreduce.cc python/tvm/relax/backend/sycl python/tvm/relax/pipeline.py -type f | sort | xargs md5sum ) > source_md5.txt
ls -la --time-style=long-iso libtvm.so libtvm_runtime.so; echo INCR_BUILD_DONE
