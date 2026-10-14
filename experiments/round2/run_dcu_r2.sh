#!/usr/bin/env bash
# DCU (gfx906, container tvm_sycl_test): overlay_r2 built in /root/r2/tvm/build (DPC++ 2022 HIP,
# -DTVM_SYCL_NO_ADDRESS_SPACE_CAST). Same protocol as the published DCU sweep (2026-09-04):
# static_shape_tuning with 0 trials (DLight), 5 warm-ups, 30 synchronized measurements, device 0,
# compact models seed 123, ResNets seed 321 via ONNX.
#   r2_old : every round-2 switch set to the published behaviour, published target string
#   r2_cg  : round-2 runtime and code generation, published target string
#   r2_dev : round-2 with the device-derived target (Target.from_device, C6)
#   rocm   : TVM ROCm/HIP baseline from /root/tvm_rocm_build (same protocol, same session)
set -uo pipefail
STAGE=${1:-all}
SRC=/root/r2/tvm
EXP=/workspace/supplement/experiments
OUT=/root/r2/results; mkdir -p $OUT
DPCPP=/root/sycl_workspace/llvm/build
export LD_LIBRARY_PATH=$SRC/build:$DPCPP/install/lib:$DPCPP/lib:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0 SYCL_DEVICE_FILTER=hip:gpu:0
PUBT="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=64 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
OLD="TVM_SYCL_SYNC_EACH_LAUNCH=1 TVM_SYCL_LEGACY_WARP_SYNC=1 TVM_SYCL_LOCAL_ZERO_INIT=1 TVM_SYCL_LAUNCH_BOUNDS=0"
ROCM_TS="rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64"

compact() {  # compact <name> <lib> <python> <targets> <target-string or ""> <env...>
  local name=$1 lib=$2 py=$3 tgt=$4 ts=$5; shift 5
  [ -s $OUT/compact_$name.json ] && return 0
  local extra=(); [ -n "$ts" ] && extra=(--target-string "$ts")
  [ "$tgt" = rocm ] && extra+=(--seed-target-alias sycl)
  env TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" python3 $EXP/relax_model_probe.py \
    --models mlp,cnn,transformer,vit --batches 1,8,16 --targets $tgt "${extra[@]}" \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_compact_$name --out-json $OUT/compact_$name.json --out-csv $OUT/compact_$name.csv \
    > $OUT/compact_$name.log 2>&1
  echo "compact_$name rc=$? $(date -Iseconds)"
}
resnet() {  # resnet <name> <models> <batches> <lib> <python> <targets> <target-string or ""> <env...>
  local name=$1 models=$2 batches=$3 lib=$4 py=$5 tgt=$6 ts=$7; shift 7
  [ -s $OUT/resnet_$name.json ] && return 0
  local extra=(); [ -n "$ts" ] && extra=(--target-string "$ts")
  env TVM_LIBRARY_PATH=$lib PYTHONPATH=$py "$@" python3 $EXP/resnet_relax_probe.py \
    --models $models --batches $batches --targets $tgt "${extra[@]}" \
    --image-size 224 --num-classes 1000 --frontend onnx \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_resnet_$name --out-json $OUT/resnet_$name.json --out-csv $OUT/resnet_$name.csv \
    > $OUT/resnet_$name.log 2>&1
  echo "resnet_$name rc=$? $(date -Iseconds)"
}
L=$SRC/build; P=$SRC/python
if [ "$STAGE" = smoke ]; then
  env TVM_LIBRARY_PATH=$L PYTHONPATH=$P python3 -c "import tvm; d=tvm.device('oneapi',0); print(tvm.__file__, d.exist); print(tvm.target.Target.from_device(d))"
  exit 0
fi
compact r2_old $L $P sycl "$PUBT" $OLD
compact r2_cg $L $P sycl "$PUBT" X=1
compact r2_dev $L $P sycl "" X=1
compact rocm /root/tvm_rocm_build /home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0/python rocm "$ROCM_TS" X=1
resnet r2_old resnet18 1 $L $P sycl "$PUBT" $OLD
resnet r2_cg resnet18,resnet34,resnet50 1,8,16 $L $P sycl "$PUBT" X=1
resnet r2_dev resnet18,resnet34,resnet50 1,8,16 $L $P sycl "" X=1
resnet rocm resnet18,resnet34,resnet50 1,8,16 /root/tvm_rocm_build /home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0/python rocm "$ROCM_TS" X=1
echo DCU_R2_DONE
