#!/usr/bin/env bash
# DCU follow-up to run_dcu_r2.sh (same protocol). The published DCU runtime already dispatched
# asynchronously (its per-launch wait is commented out), so the round-2 "old" switches are not
# its baseline. Adds:
#   pub      : the published DCU library itself, rerun in this session
#   r2_async : round 2 with zero-fill group-local memory and no launch bounds (C9, C10 off),
#              asynchronous dispatch as in the published DCU runtime, published target string
set -uo pipefail
SRC=/root/r2/tvm
PUBSRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
EXP=/workspace/supplement/experiments
OUT=/root/r2/results; mkdir -p $OUT
DPCPP=/root/sycl_workspace/llvm/build
export HIP_VISIBLE_DEVICES=0 SYCL_DEVICE_FILTER=hip:gpu:0
PUBT="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=64 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"

compact() {  # compact <name> <lib> <python> <target-string or ""> <env...>
  local name=$1 lib=$2 py=$3 ts=$4; shift 4
  [ -s $OUT/compact_$name.json ] && return 0
  local extra=(); [ -n "$ts" ] && extra=(--target-string "$ts")
  env LD_LIBRARY_PATH=$lib:$DPCPP/install/lib:$DPCPP/lib:/opt/dtk/lib:/opt/dtk/hip/lib TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" \
    python3 $EXP/relax_model_probe.py \
    --models mlp,cnn,transformer,vit --batches 1,8,16 --targets sycl "${extra[@]}" \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_compact_$name --out-json $OUT/compact_$name.json --out-csv $OUT/compact_$name.csv \
    > $OUT/compact_$name.log 2>&1
  echo "compact_$name rc=$? $(date -Iseconds)"
}
resnet() {  # resnet <name> <models> <batches> <lib> <python> <target-string or ""> <env...>
  local name=$1 models=$2 batches=$3 lib=$4 py=$5 ts=$6; shift 6
  [ -s $OUT/resnet_$name.json ] && return 0
  local extra=(); [ -n "$ts" ] && extra=(--target-string "$ts")
  env LD_LIBRARY_PATH=$lib:$DPCPP/install/lib:$DPCPP/lib:/opt/dtk/lib:/opt/dtk/hip/lib TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" \
    python3 $EXP/resnet_relax_probe.py \
    --models $models --batches $batches --targets sycl "${extra[@]}" \
    --image-size 224 --num-classes 1000 --frontend onnx \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_resnet_$name --out-json $OUT/resnet_$name.json --out-csv $OUT/resnet_$name.csv \
    > $OUT/resnet_$name.log 2>&1
  echo "resnet_$name rc=$? $(date -Iseconds)"
}
rocm_runs() {  # TVM ROCm/HIP baseline with its own libraries first on the search path
  local lib=/root/tvm_rocm_build
  local rts="rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64"
  local envs=(LD_LIBRARY_PATH=$lib:/opt/dtk/lib:/opt/dtk/hip/lib TVM_LIBRARY_PATH=$lib PYTHONPATH=$PUBSRC/python)
  [ -s $OUT/compact_rocm2.json ] || { env "${envs[@]}" python3 $EXP/relax_model_probe.py \
    --models mlp,cnn,transformer,vit --batches 1,8,16 --targets rocm --target-string "$rts" --seed-target-alias sycl \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_compact_rocm2 --out-json $OUT/compact_rocm2.json --out-csv $OUT/compact_rocm2.csv \
    > $OUT/compact_rocm2.log 2>&1; echo "compact_rocm2 rc=$? $(date -Iseconds)"; }
  [ -s $OUT/resnet_rocm2.json ] || { env "${envs[@]}" python3 $EXP/resnet_relax_probe.py \
    --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets rocm --target-string "$rts" \
    --image-size 224 --num-classes 1000 --frontend onnx \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_resnet_rocm2 --out-json $OUT/resnet_rocm2.json --out-csv $OUT/resnet_rocm2.csv \
    > $OUT/resnet_rocm2.log 2>&1; echo "resnet_rocm2 rc=$? $(date -Iseconds)"; }
}
compact pub $PUBSRC/build $PUBSRC/python "$PUBT" X=1
compact r2_async $SRC/build $SRC/python "$PUBT" TVM_SYCL_LOCAL_ZERO_INIT=1 TVM_SYCL_LAUNCH_BOUNDS=0
# Allreduce variants: the published in-warp phase without synchronization (C1 off; relies on
# wavefront lockstep), and sub-group shuffles (C5) with the device-derived target, whose warp
# size (64) equals the sub-group size the shuffles operate on.
compact r2_legacywarp $SRC/build $SRC/python "$PUBT" TVM_SYCL_LEGACY_WARP_SYNC=1
compact r2_dev_sg $SRC/build $SRC/python "" TVM_SYCL_SUBGROUP_REDUCE=1
resnet pub resnet18,resnet34,resnet50 1,8,16 $PUBSRC/build $PUBSRC/python "$PUBT" X=1
resnet r2_async resnet18 1,16 $SRC/build $SRC/python "$PUBT" TVM_SYCL_LOCAL_ZERO_INIT=1 TVM_SYCL_LAUNCH_BOUNDS=0
rocm_runs
echo DCU_R2B_DONE
