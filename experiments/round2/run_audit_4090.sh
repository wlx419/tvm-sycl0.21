#!/usr/bin/env bash
# Warp-sync audit with the published build (compile only).
set -u
T=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809; B=$T/build-4090
D=/mnt/data/wlx/sycl_workspace/DPC++
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
export PATH=$D/bin:/usr/local/cuda-12.2/bin:$PATH
export PYTHONPATH=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214:$T/python TVM_LIBRARY_PATH=$B
export LD_LIBRARY_PATH=$B:$D/lib:$D/lib64:$D/install/lib:/usr/local/cuda-12.2/lib64
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=1 ONEAPI_DEVICE_SELECTOR=cuda:*
mkdir -p $R/audit
/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python $R/repo/revision_round2/experiments/audit_warp_sync.py \
  --q3-db /mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune/sycl \
  --out-json $R/audit/warp_sync_audit_published.json --src-dir $R/audit/src > $R/audit/audit.log 2>&1
echo "rc=$?" >> $R/audit/audit.log
