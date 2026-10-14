#!/usr/bin/env bash
# Correctness with TVM_SYCL_DISCARD_EVENTS=1 (eager and command-graph replay).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk5; mkdir -p $D
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
export TVM_SYCL_DISCARD_EVENTS=1
for g in "" "--graph"; do
  $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 $g --out-json $D/rn18$g.json > $D/rn18$g.log 2>&1; tail -1 $D/rn18$g.log
  for m in mlp transformer; do
    $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$STA" --warmup 2 --measure 5 $g --out-json $D/$m$g.json > $D/$m$g.log 2>&1; tail -1 $D/$m$g.log
  done
done
echo FUNC5_DONE
