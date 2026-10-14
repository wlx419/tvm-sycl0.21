#!/usr/bin/env bash
# Timing window 3 on RTX 4090 device 1. Window 2's retunes failed in the runner wrapper (fixed: worker kill
# instead of a nonexistent PopenPoolExecutor.shutdown; failure path tested on CUDA and SYCL).
#   Part A  D1: DLight with CUDA's configurations for SYCL targets naming sm_XY (compact models,
#           DLight-only ResNet-18); the D1-off reference is the window-2 Part 0 / Part 3 data
#   Part B  equal-budget, seeded retunes, same runner policy (T1) on both targets
#   Part C  apply the new databases (eager, graph, cross), whole-model traces
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window3; mkdir -p $W $W/e1
LOG=$W/window.log
SEED=20261012
TRIALS=${TRIALS:-1024}
echo "=== window start $(date -Iseconds)" > $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/python/tvm/dlight/base/utils.py >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune

retune() {  # retune <name> <target> <target-string> [extra]
  local name=$1 t=$2 ts=$3; shift 3
  local d=$W/tune_$name; mkdir -p $d
  [ -s $d/database_tuning_record.json ] && [ -s $d/tune.json ] && return 0
  echo "--- retune $name start $(date -Iseconds)" >> $LOG
  ( source $E/env_4090.sh r2; export TVM_SYCL_JIT_JOBS=1
    /usr/bin/time -v $PY $E/resnet_relax_probe_r2.py --target $t --target-string "$ts" --total-trials $TRIALS \
      --tune-seed $SEED --runner-recycle-on-error --work-dir $d --warmup 10 --measure 50 --out-json $d/tune.json "$@" \
      > $d/tune.log 2> $d/tune.time; tail -1 $d/tune.log | sed "s/^/$name /" >> $LOG )
  echo "--- retune $name end $(date -Iseconds) records=$(wc -l < $d/database_tuning_record.json) recycled=$(grep -c 'measurement process replaced' $d/tune.log)" >> $LOG
}
apply3() {  # apply3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 \
        --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
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
    env "${envs[@]}" nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 30 --nsys-iters 20 --dump-source $W/e1/src_$name --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
}

# ---- Part A: D1
for m in mlp cnn transformer vit; do
  compact ${m}_sycl_d1 X=1 -- --model $m --target sycl --target-string "$ST"
  compact ${m}_sycl_d1_graph X=1 -- --model $m --target sycl --target-string "$ST" --graph
done
apply3 rn18_dlight_sycl_d1 X=1 -- --target sycl --target-string "$ST" --pipeline default
apply3 rn18_dlight_cuda X=1 -- --target cuda --target-string "$CT" --pipeline default
apply3 rn18_cudadb_sycl_d1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/cuda

# ---- Part B: retunes
retune sycl_pipe_iso sycl "$ST"
retune cuda_iso cuda "$CT"

# ---- Part C: apply the new databases
apply3 rn18_sycl_iso_eager X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso
apply3 rn18_sycl_iso_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso --graph
apply3 rn18_cuda_iso_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso
apply3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso --graph
apply3 rn18_sycl_with_cuda_iso_db X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_cuda_iso
apply3 rn18_cuda_with_sycl_iso_db X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_sycl_pipe_iso
e1 iso_cuda_b1 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso
e1 iso_sycl_b1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso
e1 iso_cuda_graph_b1 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso --graph
e1 iso_sycl_graph_b1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso --graph
e1 iso_cuda_b16 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso --batch 16
e1 iso_sycl_b16 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso --batch 16
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
