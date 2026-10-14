#!/usr/bin/env bash
# E10 on DCU: operator correctness with float64 references, masked relative error, ULP, raw samples.
# Runs inside the tvm_sycl_test container. Uses the same libtvm.so as the published DCU results.
set -u
TVM_HOME=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
export PYTHONPATH=$TVM_HOME/python
export TVM_LIBRARY_PATH=$TVM_HOME/build
export LD_LIBRARY_PATH=$TVM_HOME/build:/root/sycl_workspace/llvm/build/install/lib:/root/sycl_workspace/llvm/build/lib:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}
cd /home/wlx/round2_20261012/experiments
for dev in 0 1; do
  HIP_VISIBLE_DEVICES=$dev SYCL_DEVICE_FILTER=hip:gpu:0 python3 tvm_sycl_extended_probe_r2.py \
    --suite correctness --out-dir ../results --stem e10_operators_dcu_card${dev}_20261012 \
    > ../results/e10_operators_dcu_card${dev}_20261012.log 2>&1
  echo "card $dev rc=$?"
done
