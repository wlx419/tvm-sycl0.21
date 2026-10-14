#!/usr/bin/env bash
# Window 6g on RTX 4090 device 1: host
# cost of replaying a recorded command graph. SYCL (sycl_ext_oneapi_graph): cost against the number
# of earlier replays of the same executable graph, and with the graph re-finalized every N replays;
# CUDA (cudaGraphLaunch) for reference. 10 and 26 small kernels per graph, two processes each.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window6g; mkdir -p $W
LOG=$W/window.log
source $E/env_4090.sh r2 > /dev/null 2>&1
SYCL_FLAGS_BENCH="-std=c++17 -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -Wno-unknown-cuda-version"
clang++ $SYCL_FLAGS_BENCH $E/sycl_graph_replay_overhead.cpp -o $W/sycl_graph_replay2 || exit 4
[ -x $W/cuda_graph_replay ] || /usr/local/cuda-12.2/bin/nvcc -O2 -arch=sm_89 $E/cuda_graph_replay_overhead.cu -o $W/cuda_graph_replay || exit 4
trap 'exit 130' INT TERM HUP
echo "=== window start (v2) $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
for r in 1 2; do
  for k in 10 26; do
    timeout 300 $W/sycl_graph_replay2 growth $k 2>&1 | sed "s/^/sycl run=$r /" >> $LOG
    for n in 8 16 32 64 128; do
      timeout 300 $W/sycl_graph_replay2 refin $k $n 2>&1 | grep refin | sed "s/^/sycl run=$r /" >> $LOG
    done
    timeout 300 $W/cuda_graph_replay 3000 $k 2>&1 | sed "s/^/cuda run=$r k=$k /" >> $LOG
  done
done
echo "=== window end $(date -Iseconds)" >> $LOG
