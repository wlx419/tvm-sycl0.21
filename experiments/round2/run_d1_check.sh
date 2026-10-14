#!/usr/bin/env bash
# D1 (DLight uses CUDA's configurations for SYCL targets with -arch=sm_XY): correctness on GPU0
# (shared; timings not meaningful), staged Python tree, r2 libraries.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
O=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/code/overlay_r2
N=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/tree_next
rsync -a --checksum --itemize-changes $O/python/ $N/python/ | grep '^>f'
GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh next
D=$R/funcchk_d1; mkdir -p $D
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
$PY -c "import tvm, tvm.dlight.base.utils as u; print(tvm.__file__); print(u.schedule_kind(tvm.target.Target('$STA')))"
for m in mlp cnn transformer vit; do
  $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$STA" --warmup 2 --measure 5 --out-json $D/$m.json > $D/$m.log 2>&1; tail -1 $D/$m.log
done
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 --dump-source $D/src_rn18 --out-json $D/rn18_cudadb.json > $D/rn18_cudadb.log 2>&1; tail -1 $D/rn18_cudadb.log
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --pipeline default --warmup 2 --measure 5 --out-json $D/rn18_dlight.json > $D/rn18_dlight.log 2>&1; tail -1 $D/rn18_dlight.log
echo D1_DONE
