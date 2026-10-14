#!/usr/bin/env bash
# B1/B2: SYCL JIT compile time for ResNet-18 B1 (published SYCL database), overlay_r2 build.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/jit; mkdir -p $D
DB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune/sycl
ST="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
run() {  # name cache jobs
  uptime > $D/$1.load
  TVM_SYCL_JIT_CACHE=$2 TVM_SYCL_JIT_JOBS=$3 $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" \
    --apply-db-dir $DB --warmup 2 --measure 5 --out-json $D/$1.json > $D/$1.log 2>&1
  tail -1 $D/$1.log
}
run j1_nocache 0 1
run j8_nocache 0 8
run j16_nocache 0 16
run j8_cache_cold 1 8
run j8_cache_warm 1 8
echo JIT_TEST_DONE
