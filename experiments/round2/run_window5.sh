#!/usr/bin/env bash
# Timing window 5 on RTX 4090 device 1. Reduction lowering for SYCL:
#   base : work-group barriers only (default, C1)
#   sg   : sub-group shuffles (C5, TVM_SYCL_SUBGROUP_REDUCE=1; kernels require sub-group size 32)
#   grp  : sycl::reduce_over_group for whole-work-group reductions (C13, TVM_SYCL_GROUP_REDUCE=1)
# on the compact models (eager and graph) and DLight-only ResNet-18, three processes each.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window5; mkdir -p $W
LOG=$W/window.log
echo "=== window start $(date -Iseconds)" > $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
compact3() {  # compact3 <name> <env...> -- <compact args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
rn3() {  # rn3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
for m in mlp cnn transformer vit; do
  for v in base sg grp; do
    case $v in base) ev=X=1;; sg) ev=TVM_SYCL_SUBGROUP_REDUCE=1;; grp) ev=TVM_SYCL_GROUP_REDUCE=1;; esac
    compact3 ${m}_${v} $ev -- --model $m --target sycl --target-string "$ST"
    compact3 ${m}_${v}_graph $ev -- --model $m --target sycl --target-string "$ST" --graph
  done
done
for v in base sg grp; do
  case $v in base) ev=X=1;; sg) ev=TVM_SYCL_SUBGROUP_REDUCE=1;; grp) ev=TVM_SYCL_GROUP_REDUCE=1;; esac
  rn3 rn18_dlight_$v $ev -- --target sycl --target-string "$ST" --pipeline default
done
# Where the remaining compact-model gap goes: Transformer/ViT B1 traces, eager and graph, both targets
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
mkdir -p $W/e1
for m in transformer vit; do for t in sycl cuda; do for g in "" "--graph"; do
  name=${m}_${t}${g:+_graph}; ts=$ST; [ $t = cuda ] && ts=$CT
  [ -s $W/e1/$name.json ] && continue
  ( source $E/env_4090.sh r2
    nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/compact_probe_r2.py --model $m --target $t --target-string "$ts" $g --warmup 10 --measure 30 --nsys-iters 20 \
      --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/e1 $name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
done; done; done
echo "=== window end $(date -Iseconds)" >> $LOG
