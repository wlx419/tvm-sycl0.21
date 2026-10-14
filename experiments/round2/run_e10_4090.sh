#!/usr/bin/env bash
# E10 on the RTX 4090 with the published build. Error metrics only: a functional run outside the timing
# windows, so latency fields in the output are not used.
set -u
T=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809; B=$T/build-4090
D=/mnt/data/wlx/sycl_workspace/DPC++
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
export PATH=$D/bin:/usr/local/cuda-12.2/bin:$PATH
export PYTHONPATH=$T/python TVM_LIBRARY_PATH=$B
export LD_LIBRARY_PATH=$B:$D/lib:$D/lib64:$D/install/lib:/usr/local/cuda-12.2/lib64
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=1 ONEAPI_DEVICE_SELECTOR=cuda:*
mkdir -p $R/e10 && cd $R/e10
/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python $R/repo/revision_round2/experiments/tvm_sycl_extended_probe_r2.py \
  --suite correctness --out-dir $R/e10 --stem e10_operators_4090 > $R/e10/e10.log 2>&1
echo "rc=$?" >> $R/e10/e10.log
