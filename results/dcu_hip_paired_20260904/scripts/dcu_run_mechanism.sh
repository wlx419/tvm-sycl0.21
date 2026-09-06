#!/usr/bin/env bash
set -uo pipefail
SRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
export TVM_LIBRARY_PATH=/root/tvm_rocm_build
export PYTHONPATH="${SRC}/python"
mkdir -p /root/rocm_baseline/bisect_mech

python3 /home/wlx/dcu_layernorm_mechanism.py --model "${1:-transformer}" 2>&1 \
  | grep -vE "Tuning record is not found|JSONDatabase" | head -60
