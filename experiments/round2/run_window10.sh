#!/usr/bin/env bash
# Timing window 10 on RTX 4090 device 1. Follow-up of window 9 (DPC++ v7.1.1 side by side):
#   1. Nsight traces of the window-3 ResNet-18 database on both toolchains (per-kernel times; the
#      v7.1.1 build ran it 6% slower) and the compact models with command graphs on v7.1.1 with the
#      new R2 default (no re-finalization from libsycl 9 on)
#   2. retune of ResNet-18 B1 with DPC++ v7.1.1, settings of the window-3 retunes (1,024 trials,
#      seed 20261012, 8 candidates per iteration, rule kind cuda, -arch=sm_89, T1 runner, T2 timer)
#   3. the new database against the window-3 databases, eager and with command graphs
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window10; mkdir -p $W/e1
LOG=$W/window.log
SEED=20261012
TRIALS=${TRIALS:-1024}
T2=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum $T2/build-4090-r2/libtvm*.so $T2/build-4090-v711/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
W3=$R/window3

# run3 <toolchain> <name> <script> <args...>: three processes
run3() {
  local tc=$1 name=$2 script=$3; shift 3
  ( source $E/env_4090.sh $tc > /dev/null 2>&1; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      $PY $E/$script "$@" --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
# ---- 1. traces and compact graphs on v7.1.1 with the new R2 default
for tc in r2 v711; do
  name=rn18_isodb_$tc; [ -s $W/e1/$name.json ] && continue
  ( source $E/env_4090.sh $tc > /dev/null 2>&1
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 30 --nsys-iters 20 \
      --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/e1 $name /" >> $LOG
    nsys stats --report cuda_gpu_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
done
for m in transformer vit cnn mlp; do
  run3 v711 c_${m}_graph_v711 compact_probe_r2.py --model $m --target sycl --target-string "$ST" --graph --warmup 10 --measure 50
done

# ---- 2. retune with DPC++ v7.1.1
d=$W/tune_sycl_v711; mkdir -p $d
if ! { [ -s $d/database_tuning_record.json ] && [ -s $d/tune.json ]; }; then
  echo "--- retune sycl_v711 start $(date -Iseconds)" >> $LOG
  ( source $E/env_4090.sh v711 > /dev/null 2>&1; export TVM_SYCL_JIT_JOBS=1
    /usr/bin/time -v $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --total-trials $TRIALS \
      --tune-seed $SEED --runner-recycle-on-error --work-dir $d --warmup 10 --measure 50 --out-json $d/tune.json \
      > $d/tune.log 2> $d/tune.time; tail -1 $d/tune.log | sed "s/^/sycl_v711 /" >> $LOG )
  echo "--- retune sycl_v711 end $(date -Iseconds) records=$(wc -l < $d/database_tuning_record.json) recycled=$(grep -c 'measurement process replaced' $d/tune.log)" >> $LOG
fi

# ---- 3. the new database against the window-3 databases (same session)
run3 v711 rn18_v711db_v711_eager resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $d --warmup 10 --measure 50
run3 v711 rn18_v711db_v711_graph resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $d --graph --warmup 10 --measure 50
run3 r2 rn18_v711db_r2_eager resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $d --warmup 10 --measure 50
run3 v711 rn18_isodb_v711_eager resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 50
run3 v711 rn18_isodb_v711_graph resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph --warmup 10 --measure 50
run3 r2 rn18_isodb_r2_eager resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 50
run3 r2 rn18_isodb_r2_graph resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph --warmup 10 --measure 50
run3 r2 rn18_cuda_eager resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --warmup 10 --measure 50
run3 r2 rn18_cuda_graph resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph --warmup 10 --measure 50
# one trace of the new database on v7.1.1
name=rn18_v711db_v711; [ -s $W/e1/$name.json ] || ( source $E/env_4090.sh v711 > /dev/null 2>&1
  nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
    $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $d --warmup 10 --measure 30 --nsys-iters 20 \
    --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/e1 $name /" >> $LOG
  nsys stats --report cuda_gpu_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
