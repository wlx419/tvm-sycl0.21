#!/usr/bin/env bash
# SYCL tuning diagnostic under Nsight Systems on GPU0 (shared): first batches of tasks 0-2.
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh r2
D=$R/tune_diag_nsys; rm -rf $D; mkdir -p $D
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
cd /tmp
RUNNER_SLOW_SEC=0.5 nsys profile --trace=cuda --sample=none --cpuctxsw=none -o $D/diag --force-overwrite true \
  $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --total-trials 192 --tune-seed 20261012 \
  --runner-recycle-on-error --work-dir $D --warmup 2 --measure 5 --out-json $D/tune.json > $D/tune.log 2>&1
