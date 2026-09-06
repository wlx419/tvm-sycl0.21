#!/usr/bin/env bash
set -uo pipefail
SRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
export TVM_LIBRARY_PATH=/root/tvm_rocm_build
export PYTHONPATH="${SRC}/python"

TGT="${1:-rocm}"
MODEL="${2:-transformer}"
mkdir -p "/root/rocm_baseline/bisect_${TGT}"

echo "################ ${MODEL} on ${TGT} ################"
# The SYCL path emits a linker warning per kernel; keep only the report lines.
python3 /home/wlx/dcu_bisect_nonfinite.py --target "${TGT}" --model "${MODEL}" --batch 1 2>&1 \
  | grep -vE "Tuning record is not found|warning:|Wlinker-warnings|warning generated|^ \[|amdgcn|libspirv|^$" \
  | head -60
