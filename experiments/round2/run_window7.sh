#!/usr/bin/env bash
# Timing window 7 on RTX 4090 device 1. Retune of ResNet-18 B1 for SYCL with N1 (32-bit work-item/work-group
# indices, TVM_SYCL_NATIVE_IDS=1), with the settings of the window-3 retunes (1,024 trials, seed
# 20261012, 8 candidates per iteration, rule kind cuda, -arch=sm_89, T1 runner, T2 timer); then the
# new database against the window-3 databases, eager and with command graphs, in the same session.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window7; mkdir -p $W/e1
LOG=$W/window.log
SEED=20261012
TRIALS=${TRIALS:-1024}
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
W3=$R/window3

# ---- retune with N1
d=$W/tune_sycl_n1; mkdir -p $d
if ! { [ -s $d/database_tuning_record.json ] && [ -s $d/tune.json ]; }; then
  echo "--- retune sycl_n1 start $(date -Iseconds)" >> $LOG
  ( source $E/env_4090.sh r2; export TVM_SYCL_JIT_JOBS=1 TVM_SYCL_NATIVE_IDS=1
    /usr/bin/time -v $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --total-trials $TRIALS \
      --tune-seed $SEED --runner-recycle-on-error --work-dir $d --warmup 10 --measure 50 --out-json $d/tune.json \
      > $d/tune.log 2> $d/tune.time; tail -1 $d/tune.log | sed "s/^/sycl_n1 /" >> $LOG )
  echo "--- retune sycl_n1 end $(date -Iseconds) records=$(wc -l < $d/database_tuning_record.json) recycled=$(grep -c 'measurement process replaced' $d/tune.log)" >> $LOG
fi

apply3() {  # apply3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 \
        --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
# ---- the new database (N1 on) against the window-3 databases
apply3 rn18_sycl_n1db_eager TVM_SYCL_NATIVE_IDS=1 -- --target sycl --target-string "$ST" --apply-db-dir $d
apply3 rn18_sycl_n1db_graph TVM_SYCL_NATIVE_IDS=1 -- --target sycl --target-string "$ST" --apply-db-dir $d --graph
apply3 rn18_sycl_n1db_off X=1 -- --target sycl --target-string "$ST" --apply-db-dir $d
apply3 rn18_sycl_isodb_eager X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
apply3 rn18_sycl_isodb_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph
apply3 rn18_cuda_iso_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso
apply3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph
# ---- one trace of the new database (per-kernel comparison with the window-6d traces)
[ -s $W/e1/n1db_sycl.json ] || ( source $E/env_4090.sh r2; export TVM_SYCL_NATIVE_IDS=1
  nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/n1db_sycl --force-overwrite true \
    $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $d --warmup 10 --measure 30 --nsys-iters 20 \
    --out-json $W/e1/n1db_sycl.json 2>&1 | tail -1 | sed "s/^/e1 n1db_sycl /" >> $LOG
  nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/n1db_sycl $W/e1/n1db_sycl.nsys-rep > /dev/null 2>&1 )
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
