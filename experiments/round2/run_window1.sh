#!/usr/bin/env bash
# Timing window 1 on RTX 4090 device 1. Stops this session's own compile-only audit job for the
# duration (resumed on any exit) so the host is quiet.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window1; mkdir -p $W
LOG=$W/window.log
AUDIT_PIDS=$(pgrep -f "run_audit_4090.sh|audit_warp_sync.py" | tr '\n' ' ')
resume_all() {
  [ -n "$AUDIT_PIDS" ] && kill -CONT $AUDIT_PIDS 2>/dev/null
  for p in $(pgrep -f "tvm-sycl-clang"); do kill -CONT $p 2>/dev/null; done
  echo "=== resumed $(date -Iseconds)" >> $LOG
}
echo "=== window start $(date -Iseconds)" > $LOG
[ -n "$AUDIT_PIDS" ] && kill -STOP $AUDIT_PIDS 2>/dev/null
for p in $(pgrep -f "tvm-sycl-clang"); do kill -STOP $p 2>/dev/null; done
# Processes on GPU1 other than this session's own (stopped, compile-only) audit job.
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep e5e525e1 |        awk -F', ' -v own=" $AUDIT_PIDS " 'index(own, " " $2 " ") == 0' | wc -l)
echo "compute apps on GPU1 other than the stopped audit job: $APPS (audit job: $AUDIT_PIDS)" >> $LOG
[ "$APPS" = "0" ] || { echo "GPU1 still busy; aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
uptime >> $LOG

DB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
ST="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"

rn() {  # rn <build> <name> <env...> -- <probe args>
  local build=$1 name=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh $build; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 >> $LOG
    done )
}
compact() {  # compact <name> <env...> -- <compact args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 >> $LOG
    done )
}

# ---- Group A: ResNet-18 B1, paired MetaSchedule databases
rn published rn18_published_sycl X=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn published rn18_published_cuda X=1 -- --target cuda --target-string "$CT" --apply-db-dir $DB/cuda
rn r2 rn18_sycl_a0_sync_legacy TVM_SYCL_SYNC_EACH_LAUNCH=1 TVM_SYCL_LEGACY_WARP_SYNC=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn r2 rn18_sycl_a1_async_legacy TVM_SYCL_SYNC_EACH_LAUNCH=0 TVM_SYCL_LEGACY_WARP_SYNC=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn r2 rn18_sycl_a2_async_fixed X=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl
rn r2 rn18_sycl_a3_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $DB/sycl --graph
rn r2 rn18_cuda_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $DB/cuda
rn r2 rn18_cuda_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $DB/cuda --graph

# ---- Group B: compact models B1, default pipeline (DLight)
for m in mlp cnn transformer vit; do
  compact ${m}_sycl_a0_sync_legacy TVM_SYCL_SYNC_EACH_LAUNCH=1 TVM_SYCL_LEGACY_WARP_SYNC=1 -- --model $m --target sycl --target-string "$ST"
  compact ${m}_sycl_a2_async_fixed X=1 -- --model $m --target sycl --target-string "$ST"
  compact ${m}_sycl_a3_graph X=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact ${m}_cuda_eager X=1 -- --model $m --target cuda --target-string "$CT"
  compact ${m}_cuda_graph X=1 -- --model $m --target cuda --target-string "$CT" --graph
done

# ---- Group C: whole-model Nsight Systems traces (E1)
mkdir -p $W/e1
e1() {  # e1 <build> <name> <env...> -- <probe args>
  local build=$1 name=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  [ -s $W/e1/$name.json ] && return 0
  ( source $E/env_4090.sh $build
    env "${envs[@]}" nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 30 --nsys-iters 20 --dump-source $W/e1/src --out-json $W/e1/$name.json 2>&1 | tail -1 >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
}
for b in 1 16; do
  e1 published sycl_b$b X=1 -- --target sycl --target-string "$ST" --batch $b --apply-db-dir $DB/sycl
  e1 published cuda_b$b X=1 -- --target cuda --target-string "$CT" --batch $b --apply-db-dir $DB/cuda
  e1 r2 sycl_r2async_b$b X=1 -- --target sycl --target-string "$ST" --batch $b --apply-db-dir $DB/sycl
  e1 r2 sycl_r2graph_b$b X=1 -- --target sycl --target-string "$ST" --batch $b --apply-db-dir $DB/sycl --graph
done
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
