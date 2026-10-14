#!/usr/bin/env bash
# After T2 (synchronizing SYCL timer): first batches of tasks 0-2 on GPU0 (shared), C11 check.
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh r2
D=$R/tune_diag_t2; rm -rf $D; mkdir -p $D
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
cd /tmp
for m in transformer vit; do
  TVM_SYCL_SUBGROUP_REDUCE=1 $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 2 --measure 5 --out-json $D/c11_$m.json 2>&1 | tail -1
done
RUNNER_SLOW_SEC=0.5 timeout 1500 $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --total-trials 256 --tune-seed 20261012 \
  --runner-recycle-on-error --work-dir $D --warmup 2 --measure 5 --out-json $D/tune.json > $D/tune.log 2>&1
grep "\[runner\]\|picks Task" $D/tune.log | tail -25 | cut -c1-150; tail -1 $D/tune.log; wc -l < $D/database_tuning_record.json
