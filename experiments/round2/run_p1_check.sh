#!/usr/bin/env bash
# P1/C8 check: CUDA's full published database (pipelined records included) applied to SYCL.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk3; mkdir -p $D
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 \
  --dump-source $D/src_p1 --out-json $D/p1_sycl_with_full_cuda_db.json > $D/p1.log 2>&1; tail -2 $D/p1.log
grep -h -c "_p_\[2\]" $D/src_p1/*.src 2>/dev/null | paste -sd+ | bc
echo P1_DONE
