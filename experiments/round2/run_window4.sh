#!/usr/bin/env bash
# Timing window 4 on RTX 4090 device 1. Final-code seven-model batch sweep (default pipeline,
# DLight, no tuning) on both targets, eager and graph, three processes per configuration; plus
# the D1 step of the ResNet-18 ablation on the published SYCL database.
#   SYCL target: device-derived (Target.from_device: arch=sm_89, 1024 threads, warp 32, 48 KB)
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window4; mkdir -p $W
LOG=$W/window.log
echo "=== window start $(date -Iseconds)" > $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
ST0="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune

rn3() {  # rn3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 \
        --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
compact3() {  # compact3 <name> <env...> -- <compact args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}

# ---- ResNet-18 ablation, D1 step: published SYCL database, SYCL target with arch (DLight
#      fallbacks use CUDA's configurations); compare with window 1b rn18_sycldb_sycl_c10
rn3 rn18_sycldb_sycl_d1 X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl
rn3 rn18_sycldb_sycl_d1_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl --graph

# ---- Batch sweep (default pipeline)
for b in 1 8 16; do
  for m in mlp cnn transformer vit; do
    compact3 sweep_${m}_b${b}_sycl X=1 -- --model $m --batch $b --target sycl --target-string "$ST"
    compact3 sweep_${m}_b${b}_cuda X=1 -- --model $m --batch $b --target cuda --target-string "$CT"
    compact3 sweep_${m}_b${b}_sycl_graph X=1 -- --model $m --batch $b --target sycl --target-string "$ST" --graph
    compact3 sweep_${m}_b${b}_cuda_graph X=1 -- --model $m --batch $b --target cuda --target-string "$CT" --graph
  done
  for m in resnet18 resnet34 resnet50; do
    rn3 sweep_${m}_b${b}_sycl X=1 -- --model $m --batch $b --target sycl --target-string "$ST" --pipeline default
    rn3 sweep_${m}_b${b}_cuda X=1 -- --model $m --batch $b --target cuda --target-string "$CT" --pipeline default
  done
done
# ---- E11: compute-bound shapes, oneDNN (SYCL runtime, NVIDIA vendor backend) vs TVM DLight
mkdir -p $W/e11
( source $E/env_4090.sh r2
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/data/wlx/anaconda3/envs/looptune2/lib/python3.9/site-packages/nvidia/cudnn/lib
  export ONEAPI_DEVICE_SELECTOR=cuda:gpu
  for r in 1 2 3; do [ -s $W/e11/onednn_run$r.csv ] || $R/e11/onednn_large_shapes $W/e11/onednn_run$r.csv | sed "s/^/e11_onednn /" >> $LOG; done )
( source $E/env_4090.sh r2
  for op in matmul conv2d; do for r in 1 2 3; do
    [ -s $W/e11/sycl_${op}_run$r.json ] || $PY $E/big_ops_probe_r2.py --op $op --target sycl --target-string "$ST" --out-json $W/e11/sycl_${op}_run$r.json 2>&1 | tail -1 | sed "s/^/e11 /" >> $LOG
    [ -s $W/e11/cuda_${op}_run$r.json ] || $PY $E/big_ops_probe_r2.py --op $op --target cuda --target-string "$CT" --out-json $W/e11/cuda_${op}_run$r.json 2>&1 | tail -1 | sed "s/^/e11 /" >> $LOG
  done; done )

# ---- E12: TVM-OpenCL on the same stack (NVIDIA OpenCL ICD; target from the device, DLight)
for b in 1 8 16; do
  for m in mlp cnn transformer vit; do
    compact3 sweep_${m}_b${b}_opencl X=1 -- --model $m --batch $b --target opencl
  done
  for m in resnet18 resnet34 resnet50; do
    rn3 sweep_${m}_b${b}_opencl X=1 -- --model $m --batch $b --target opencl --pipeline default
  done
done
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
