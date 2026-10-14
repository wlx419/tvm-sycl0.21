#!/usr/bin/env bash
# After window 3: rebuild r2 with the latest overlay (C13), check C13 on GPU0, then windows 4 and 5.
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
while ps -u wlx -o args | grep -q "^bash repo/revision_round2/experiments/run_window3.sh"; do sleep 60; done
echo "window3 done $(date -Iseconds)"
bash $E/sync_r2_4090.sh 2>&1 | tail -3
( GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh r2
  ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
  mkdir -p $R/funcchk_c13; cd /tmp
  for m in transformer vit cnn; do
    TVM_SYCL_GROUP_REDUCE=1 $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 2 --measure 5 --out-json $R/funcchk_c13/$m.json 2>&1 | tail -1
  done )
cd $R && bash repo/revision_round2/experiments/run_window4.sh; echo "window4 rc=$? $(date -Iseconds)"
cd $R && bash repo/revision_round2/experiments/run_window5.sh; echo "window5 rc=$? $(date -Iseconds)"
