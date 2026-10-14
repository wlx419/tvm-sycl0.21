#!/usr/bin/env bash
# Correctness after F2 (float min/max as sycl::fmin/fmax) and the dead-code cleanup and F1 (-ffp-contract=fast
# behind the CUDA plugin), on top of M1 and J1; GPU0, shared (timings not meaningful).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh r2
D=$R/funcchk10; mkdir -p $D; cd /tmp
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
for m in mlp cnn transformer vit; do for g in "" "--graph"; do
  $PY $E/compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 2 --measure 5 $g --out-json $D/$m$g.json 2>&1 | tail -1 | cut -c1-110
done; done
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $R/window3/tune_sycl_pipe_iso --warmup 2 --measure 5 --out-json $D/rn18_iso.json 2>&1 | tail -1 | cut -c1-110
$PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --pipeline default --model resnet50 --warmup 2 --measure 5 --out-json $D/rn50.json 2>&1 | tail -1 | cut -c1-110
ls -t /tmp/tvm_sycl/*.cc | head -1 | xargs -I{} sh -c 'echo cache entry: {}'
echo FUNC10_DONE
C=$(ls -t /tmp/tvm_sycl/*.cc | head -1); echo "native ids in newest entry: $(grep -c 'TVM_SYCL_LOCAL_ID(' $C)"
