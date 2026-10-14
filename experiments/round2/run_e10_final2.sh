#!/usr/bin/env bash
# E10 (Table 3) on the RTX 4090 with the final round-2 build and the target the models use
# (-arch=sm_89: M1 native float32 erf/tanh behind the CUDA plugin; F1 contraction; J1 no implicit
# offset). GPU0, outside the timing windows: error metrics only, latency fields are not used.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
GPU_UUID=GPU-7c54a1ad-c307-6424-f337-caadcec1c974 source $E/env_4090.sh r2
export TVM_E10_TARGET="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
OUT=$R/e10_final2_sm89
mkdir -p $OUT && cd $E
nice -n 10 $PY tvm_sycl_extended_probe_r2.py --suite correctness --out-dir $OUT --stem e10_r2final2_sm89_4090 \
  > $OUT/e10.log 2>&1
echo "rc=$?"
grep -c "\-> pass" $OUT/e10.log
grep -m3 "TVM_SYCL_LOCAL_ID\|sycl::erf\|erf(" $(ls -t /tmp/tvm_sycl/*.cc | head -1) | head -3
echo E10_FINAL2_DONE
