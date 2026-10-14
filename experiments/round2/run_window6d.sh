#!/usr/bin/env bash
# Diagnostic window 6d on RTX 4090 device 1: per-kernel traces of the retuned ResNet-18 (window-3 SYCL database) with N1 on and off,
# and CUDA's retuned database for reference. N1 lowers every kernel's registers (about 125 -> 64)
# yet the model is 25% slower; the traces show which kernels and whether launch gaps change.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window6d; mkdir -p $W/e1
LOG=$W/window.log
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
W3=$R/window3
e1() {  # e1 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -s $W/e1/$name.json ] && return 0
  ( source $E/env_4090.sh r2
    env "${envs[@]}" nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 30 --nsys-iters 20 --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
}
e1 iso_sycl_n1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
e1 iso_sycl_non1 TVM_SYCL_NATIVE_IDS=0 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
e1 iso_cuda X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso
echo "=== window end $(date -Iseconds)" >> $LOG
