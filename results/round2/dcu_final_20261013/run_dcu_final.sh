#!/usr/bin/env bash
# DCU final comparison, card 0 only, one configuration at a time (nothing else on the host).
# Protocol as the published DCU sweep (2026-09-04): static_shape_tuning with 0 trials (DLight),
# 5 warm-ups, 30 synchronized measurements; compact seed 123, ResNets seed 321 via ONNX.
#   pub      : published DCU library (rerun in this session)
#   r2       : overlay_r2 final (async dispatch, C9/C10, C1, discard_events by default on this
#              DPC++), device-derived target + -arch=gfx906 (D1: ROCm's DLight configurations)
#   r2_nod1  : r2 without -arch (D1 off)
#   r2_nodisc: r2 with TVM_SYCL_DISCARD_EVENTS=0
#   rocm     : TVM ROCm/HIP from /root/tvm_rocm_build
set -uo pipefail
SRC=/root/r2/tvm
PUBSRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
EXP=/workspace/supplement/experiments
OUT=/root/r2/results_final; mkdir -p $OUT
DPCPP=/root/sycl_workspace/llvm/build
export HIP_VISIBLE_DEVICES=0 SYCL_DEVICE_FILTER=hip:gpu:0
PUBT="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=64 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
DEVT="sycl -keys=sycl,gpu -l2_cache_size_bytes=8388608 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=64"
ROCM_TS="rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64"

run() {  # run <name> <lib> <python> <target> <target-string> <env...>
  local name=$1 lib=$2 py=$3 tgt=$4 ts=$5; shift 5
  local ld=$lib:$DPCPP/install/lib:$DPCPP/lib:/opt/dtk/lib:/opt/dtk/hip/lib
  local alias=(); [ "$tgt" = rocm ] && alias=(--seed-target-alias sycl)
  [ -s $OUT/compact_$name.json ] || { env LD_LIBRARY_PATH=$ld TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" \
    python3 $EXP/relax_model_probe.py --models mlp,cnn,transformer,vit --batches 1,8,16 --targets $tgt \
    --target-string "$ts" "${alias[@]}" --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 \
    --device-id 0 --work-dir $OUT/work_compact_$name --out-json $OUT/compact_$name.json \
    --out-csv $OUT/compact_$name.csv > $OUT/compact_$name.log 2>&1; echo "compact_$name rc=$? $(date -Iseconds)"; }
  [ -s $OUT/resnet_$name.json ] || { env LD_LIBRARY_PATH=$ld TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" \
    python3 $EXP/resnet_relax_probe.py --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets $tgt \
    --target-string "$ts" --image-size 224 --num-classes 1000 --frontend onnx \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_resnet_$name --out-json $OUT/resnet_$name.json \
    --out-csv $OUT/resnet_$name.csv > $OUT/resnet_$name.log 2>&1; echo "resnet_$name rc=$? $(date -Iseconds)"; }
}
md5sum $SRC/build/libtvm.so $SRC/build/libtvm_runtime.so $PUBSRC/build/libtvm.so
run pub $PUBSRC/build $PUBSRC/python sycl "$PUBT" X=1
run r2 $SRC/build $SRC/python sycl "$DEVT -arch=gfx906" X=1
run r2_nod1 $SRC/build $SRC/python sycl "$DEVT" X=1
run r2_nodisc $SRC/build $SRC/python sycl "$DEVT -arch=gfx906" TVM_SYCL_DISCARD_EVENTS=0
run rocm /root/tvm_rocm_build $PUBSRC/python rocm "$ROCM_TS" X=1
echo DCU_FINAL_DONE
