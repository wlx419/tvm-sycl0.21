#!/usr/bin/env bash
# Timing window 2 on RTX 4090 device 1.
#   Part 0  launch-overhead microbenchmark; discard_events A/B (compact models, ResNet-18)
#   Part 1  E3 cross-replay of the published databases (apply only)
#   Part 2  equal-budget, seeded MetaSchedule retunes with the same runner policy on both targets:
#             sycl_pipe_iso : SYCL target with arch=sm_89 (pipelined schedules allowed, as for CUDA),
#                             overlay_r2 (async dispatch, fixed allreduce, C8-C10 codegen)
#             cuda_iso      : CUDA target (as published, arch=sm_89)
#           runner process replaced after any failed candidate (T1)
#   Part 3  three apply-only processes per database, eager and graph; whole-model traces
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window2; mkdir -p $W $W/e1
LOG=$W/window.log
SEED=20261012
TRIALS=${TRIALS:-1024}
echo "=== window start $(date -Iseconds)" > $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
( cd $E/../code && md5sum README_overlay_r2.md ) >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
ST0="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune

retune() {  # retune <name> <target> <target-string> [extra]
  local name=$1 t=$2 ts=$3; shift 3
  local d=$W/tune_$name; mkdir -p $d
  [ -s $d/tune.json ] && return 0
  echo "--- retune $name start $(date -Iseconds)" >> $LOG
  ( source $E/env_4090.sh r2; export TVM_SYCL_JIT_JOBS=1
    /usr/bin/time -v $PY $E/resnet_relax_probe_r2.py --target $t --target-string "$ts" --total-trials $TRIALS \
      --tune-seed $SEED --runner-recycle-on-error --work-dir $d --warmup 10 --measure 50 --out-json $d/tune.json "$@" \
      > $d/tune.log 2> $d/tune.time; tail -1 $d/tune.log | sed "s/^/$name /" >> $LOG )
  echo "--- retune $name end $(date -Iseconds)" >> $LOG
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

# ---- Part 0: launch overhead and discard_events
( source $E/env_4090.sh r2; cd $R/launch
  for r in 1 2 3; do ./sycl_launch 20000; ./cuda_launch 20000; done ) >> $LOG 2>&1
for m in mlp cnn transformer vit; do
  compact ${m}_sycl_c10 X=1 -- --model $m --target sycl --target-string "$ST0"
  compact ${m}_sycl_c10_de TVM_SYCL_DISCARD_EVENTS=1 -- --model $m --target sycl --target-string "$ST0"
  compact ${m}_sycl_c10_graph X=1 -- --model $m --target sycl --target-string "$ST0" --graph
  compact ${m}_sycl_c10_de_graph TVM_SYCL_DISCARD_EVENTS=1 -- --model $m --target sycl --target-string "$ST0" --graph
  compact ${m}_cuda X=1 -- --model $m --target cuda --target-string "$CT"
  compact ${m}_cuda_graph X=1 -- --model $m --target cuda --target-string "$CT" --graph
done
apply3 rn18_cudadb_sycl_c10_de TVM_SYCL_DISCARD_EVENTS=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/cuda
apply3 rn18_cudadb_sycl_c10_de_graph TVM_SYCL_DISCARD_EVENTS=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/cuda --graph

# ---- Part 1: E3 cross-replay with identical schedules (apply only; published databases)
[ -s $W/cuda_db_noasync/database_tuning_record.json ] || python3 $E/../scripts/filter_db_no_async.py $PUB/cuda $W/cuda_db_noasync >> $LOG 2>&1
apply3 e3_sycl_with_cuda_noasync_db X=1 -- --target sycl --target-string "$ST0" --apply-db-dir $W/cuda_db_noasync
apply3 e3_cuda_with_cuda_noasync_db X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/cuda_db_noasync
apply3 e3_cuda_with_sycl_db X=1 -- --target cuda --target-string "$CT" --apply-db-dir $PUB/sycl

# ---- Part 2: retunes
retune sycl_pipe_iso sycl "$ST"
retune cuda_iso cuda "$CT"

# ---- Part 3: apply the new databases
apply3 rn18_sycl_pipe_iso_eager X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso
apply3 rn18_sycl_pipe_iso_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso --graph
apply3 rn18_cuda_iso_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso
apply3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso --graph
apply3 rn18_sycl_with_cuda_iso_db X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_cuda_iso
apply3 rn18_cuda_with_sycl_iso_db X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_sycl_pipe_iso
e1 iso_cuda_b1 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso
e1 iso_sycl_b1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso
e1 iso_cuda_graph_b1 X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W/tune_cuda_iso --graph
e1 iso_sycl_graph_b1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W/tune_sycl_pipe_iso --graph
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
