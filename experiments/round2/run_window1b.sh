#!/usr/bin/env bash
# Timing window 1b on RTX 4090 device 1. Isolates the code-generation changes made after window 1:
#   C8  3-lane vectors accessed element-wise (needed for CUDA's schedules to run on SYCL at all)
#   C9  group_local_memory_for_overwrite (no zero-fill + barrier at work-group start)
#        published behaviour: TVM_SYCL_LOCAL_ZERO_INIT=1
#   C10 intel::max_work_group_size -> PTX .maxntid, the counterpart of CUDA's __launch_bounds__
#        published behaviour: TVM_SYCL_LAUNCH_BOUNDS=0
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window1b; mkdir -p $W
LOG=$W/window.log
echo "=== window start $(date -Iseconds)" > $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
echo "compute apps on GPU1: $APPS" >> $LOG
[ "$APPS" = "0" ] || { echo "GPU1 still busy; aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG

DB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
ST="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
STA="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
OLD="TVM_SYCL_LOCAL_ZERO_INIT=1 TVM_SYCL_LAUNCH_BOUNDS=0"

rn() {  # rn <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
compact() {  # compact <name> <env...> -- <compact args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
e1() {  # e1 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -s $W/e1/$name.json ] && return 0
  ( source $E/env_4090.sh r2
    env "${envs[@]}" nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 30 --nsys-iters 20 --dump-source $W/e1/src_$name --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
}
mkdir -p $W/e1

# ---- Group A: identical schedules (CUDA's published database) on both targets
rn rn18_cudadb_cuda X=1 -- --target cuda --target-string "$CT" --apply-db-dir $DB/cuda
rn rn18_cudadb_sycl_c8 $OLD -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda
rn rn18_cudadb_sycl_c9 TVM_SYCL_LAUNCH_BOUNDS=0 -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda
rn rn18_cudadb_sycl_c10 X=1 -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda
rn rn18_cudadb_sycl_c10_graph X=1 -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda --graph

# ---- Group B: published SYCL database
rn rn18_sycldb_sycl_a2 $OLD -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn rn18_sycldb_sycl_c9 TVM_SYCL_LAUNCH_BOUNDS=0 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn rn18_sycldb_sycl_c10 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl

# ---- Group C: compact models (default pipeline, DLight)
for m in mlp cnn transformer vit; do
  compact ${m}_sycl_a2 $OLD -- --model $m --target sycl --target-string "$ST"
  compact ${m}_sycl_c10 X=1 -- --model $m --target sycl --target-string "$ST"
  compact ${m}_sycl_c10_graph X=1 -- --model $m --target sycl --target-string "$ST" --graph
done

# ---- Group D: per-kernel traces
e1 cudadb_cuda_b1 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $DB/cuda
e1 cudadb_sycl_c8_b1 $OLD -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda
e1 cudadb_sycl_c10_b1 X=1 -- --target sycl --target-string "$STA" --apply-db-dir $DB/cuda
e1 sycldb_sycl_c10_b1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
