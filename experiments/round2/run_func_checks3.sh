#!/usr/bin/env bash
# Correctness after C8 (3-lane vectors), C9 (group-local memory without zero-fill), C10 (launch bounds).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk4; mkdir -p $D
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
ST0="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$STA" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 --dump-source $D/src_cudadb --out-json $D/rn18_cudadb.json > $D/rn18_cudadb.log 2>&1; tail -1 $D/rn18_cudadb.log
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST0" --apply-db-dir $PUB/sycl --warmup 2 --measure 5 --dump-source $D/src_sycldb --out-json $D/rn18_sycldb.json > $D/rn18_sycldb.log 2>&1; tail -1 $D/rn18_sycldb.log
for m in mlp cnn transformer vit; do
  $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$ST0" --warmup 2 --measure 5 --out-json $D/$m.json > $D/$m.log 2>&1; tail -1 $D/$m.log
  TVM_SYCL_SUBGROUP_REDUCE=1 $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$STA" --warmup 2 --measure 5 --out-json $D/${m}_sg.json > $D/${m}_sg.log 2>&1; tail -1 $D/${m}_sg.log
done
echo "max_work_group_size kernels: $(grep -h -c 'intel::max_work_group_size' $D/src_cudadb/*.src | paste -sd+ | bc)"
echo "for_overwrite allocs: $(grep -h -c 'group_local_memory_for_overwrite' $D/src_cudadb/*.src | paste -sd+ | bc)"
echo "zero-init allocs: $(grep -h -c 'group_local_memory<' $D/src_cudadb/*.src | paste -sd+ | bc)"
echo FUNC4_DONE
