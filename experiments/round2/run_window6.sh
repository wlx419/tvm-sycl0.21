#!/usr/bin/env bash
# Timing window 6 on RTX 4090 device 1. Code generation after M1 (no float64 erf / tanh form behind the CUDA
# plugin), J1 (no implicit global offset) and F1 (-ffp-contract=fast, nvcc's default); N1 (32-bit
# work-item / work-group indices on NVPTX) is opt-in (window 6e). Batch sweep on both targets,
# retuned ResNet-18, the published-database ablation steps, switch-off ablation, E11 TVM shapes,
# and Transformer/ViT traces. Outputs that exist are skipped (CUDA outputs of the two earlier
# attempts are kept; their SYCL outputs, measured with N1 on, are in n1_unmasked/ and n1_hw/).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window6; mkdir -p $W
LOG=$W/window.log
# a signal ends the window
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
W3=$R/window3

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

# ---- Retuned ResNet-18 (window-3 databases, identical tuning settings), both targets
rn3 rn18_sycl_iso_eager X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
rn3 rn18_sycl_iso_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph
rn3 rn18_sycl_iso_eager_none TVM_SYCL_FP_CONTRACT=0 TVM_SYCL_GLOBAL_OFFSET=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
rn3 rn18_cuda_iso_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso
rn3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph
# identical schedules: CUDA's published database on both targets
rn3 rn18_cudadb_sycl X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/cuda
rn3 rn18_cudadb_cuda X=1 -- --target cuda --target-string "$CT" --apply-db-dir $PUB/cuda

# ---- Published SYCL database: final code and each new change switched off (ablation rows)
rn3 rn18_sycldb_sycl_final X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl
rn3 rn18_sycldb_sycl_final_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl --graph
rn3 rn18_sycldb_sycl_withN1 TVM_SYCL_NATIVE_IDS=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl
rn3 rn18_sycldb_sycl_noF1 TVM_SYCL_FP_CONTRACT=0 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl
rn3 rn18_sycldb_sycl_noJ1 TVM_SYCL_GLOBAL_OFFSET=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl

# ---- Switch-off ablation on the compact models (B1, graph replay: kernel time dominates)
for m in transformer vit; do
  compact3 abl_${m}_final_graph X=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 abl_${m}_noM1_graph TVM_SYCL_MATH_WORKAROUNDS=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 abl_${m}_noJ1_graph TVM_SYCL_GLOBAL_OFFSET=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 abl_${m}_withN1_graph TVM_SYCL_NATIVE_IDS=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 abl_${m}_noF1_graph TVM_SYCL_FP_CONTRACT=0 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 abl_${m}_none_graph TVM_SYCL_MATH_WORKAROUNDS=1 TVM_SYCL_GLOBAL_OFFSET=1 TVM_SYCL_FP_CONTRACT=0 -- --model $m --target sycl --target-string "$ST" --graph
done

# ---- Batch sweep (default pipeline), both targets in the same window
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

# ---- E11 TVM side (oneDNN unchanged since window 4)
mkdir -p $W/e11
( source $E/env_4090.sh r2
  for op in matmul conv2d; do for r in 1 2 3; do
    [ -s $W/e11/sycl_${op}_run$r.json ] || $PY $E/big_ops_probe_r2.py --op $op --target sycl --target-string "$ST" --out-json $W/e11/sycl_${op}_run$r.json 2>&1 | tail -1 | sed "s/^/e11 /" >> $LOG
    [ -s $W/e11/cuda_${op}_run$r.json ] || $PY $E/big_ops_probe_r2.py --op $op --target cuda --target-string "$CT" --out-json $W/e11/cuda_${op}_run$r.json 2>&1 | tail -1 | sed "s/^/e11 /" >> $LOG
  done; done )

# ---- Traces: Transformer/ViT B1, eager and graph, both targets
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
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
