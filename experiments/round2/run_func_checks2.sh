#!/usr/bin/env bash
# Correctness checks for P1 (pipelined schedules on SYCL), C5 (sub-group shuffles), C6 (arch).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk2; mkdir -p $D
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
$PY -c "import tvm; print('from_device:', tvm.target.Target.from_device(tvm.device('oneapi', 0)))" > $D/detect.txt 2>&1; cat $D/detect.txt
# P1: CUDA's full database (pipelined records included) compiled for SYCL
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 \
  --dump-source $D/src_p1 --out-json $D/p1_sycl_with_full_cuda_db.json > $D/p1.log 2>&1; tail -1 $D/p1.log
# C5: sub-group shuffles
for m in transformer vit; do
  TVM_SYCL_SUBGROUP_REDUCE=1 $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$STA" --warmup 2 --measure 5 --out-json $D/c5_$m.json > $D/c5_$m.log 2>&1; tail -1 $D/c5_$m.log
done
TVM_SYCL_SUBGROUP_REDUCE=1 $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/sycl --warmup 2 --measure 5 \
  --dump-source $D/src_c5 --out-json $D/c5_resnet18.json > $D/c5_rn.log 2>&1; tail -1 $D/c5_rn.log
grep -l "select_from_group\|shift_group_left" $D/src_c5/*.src 2>/dev/null | head -3
grep -c "shift_group_left(item" $D/src_c5/*.src 2>/dev/null | head -3
echo FUNC2_DONE
