#!/usr/bin/env bash
# Correctness of the deployed r2 tree before window 3 (shared GPU; timings not meaningful).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk6; mkdir -p $D
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
ST0="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
cd /tmp
$PY $E/test_recycle_runner.py sycl "$ST" 2>&1 | grep RECYCLE
for m in mlp cnn transformer vit; do
  for g in "" "--graph"; do
    $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 2 --measure 5 $g --out-json $D/$m$g.json > $D/$m$g.log 2>&1; tail -1 $D/$m$g.log
  done
done
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --pipeline default --warmup 2 --measure 5 --dump-source $D/src_dlight --out-json $D/rn18_dlight.json > $D/rn18_dlight.log 2>&1; tail -1 $D/rn18_dlight.log
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $PUB/cuda --warmup 2 --measure 5 --out-json $D/rn18_cudadb.json > $D/rn18_cudadb.log 2>&1; tail -1 $D/rn18_cudadb.log
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST0" --apply-db-dir $PUB/sycl --warmup 2 --measure 5 --graph --out-json $D/rn18_sycldb_graph.json > $D/rn18_sycldb_graph.log 2>&1; tail -1 $D/rn18_sycldb_graph.log
rm -rf $D/tune_smoke; $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --total-trials 64 --tune-seed 20261012 --runner-recycle-on-error --work-dir $D/tune_smoke --warmup 2 --measure 5 --out-json $D/tune_smoke.json > $D/tune_smoke.log 2>&1; tail -1 $D/tune_smoke.log
echo "smoke records: $(wc -l < $D/tune_smoke/database_tuning_record.json) recycled: $(grep -c 'measurement process replaced' $D/tune_smoke.log)"
grep -h -c "intel::max_work_group_size" $D/src_dlight/*.src | paste -sd+ | bc
echo FUNC6_DONE
