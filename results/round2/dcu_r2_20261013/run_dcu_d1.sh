#!/usr/bin/env bash
# DCU card 1: D1 for gfx906 (DLight uses ROCm's configurations for a SYCL target naming gfx*).
# Same protocol as run_dcu_r2.sh; both arms on card 1 in the same session.
#   dev    : device-derived target (Target.from_device), as r2_dev
#   dev_d1 : the same target plus -arch=gfx906
set -uo pipefail
SRC=/root/r2/tvm
EXP=/workspace/supplement/experiments
OUT=/root/r2/results_d1; mkdir -p $OUT
DPCPP=/root/sycl_workspace/llvm/build
export LD_LIBRARY_PATH=$SRC/build:$DPCPP/install/lib:$DPCPP/lib:/opt/dtk/lib:/opt/dtk/hip/lib
export TVM_LIBRARY_PATH=$SRC/build PYTHONPATH=$SRC/python
export HIP_VISIBLE_DEVICES=1 SYCL_DEVICE_FILTER=hip:gpu:0
DEVT="sycl -keys=sycl,gpu -l2_cache_size_bytes=8388608 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=64"
for name in dev dev_d1; do
  ts="$DEVT"; [ $name = dev_d1 ] && ts="$DEVT -arch=gfx906"
  [ -s $OUT/compact_$name.json ] || { python3 $EXP/relax_model_probe.py \
    --models mlp,cnn,transformer,vit --batches 1,8,16 --targets sycl --target-string "$ts" \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_compact_$name --out-json $OUT/compact_$name.json --out-csv $OUT/compact_$name.csv \
    > $OUT/compact_$name.log 2>&1; echo "compact_$name rc=$? $(date -Iseconds)"; }
  [ -s $OUT/resnet_$name.json ] || { python3 $EXP/resnet_relax_probe.py \
    --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets sycl --target-string "$ts" \
    --image-size 224 --num-classes 1000 --frontend onnx \
    --pipeline static_shape_tuning --total-trials 0 --warmup 5 --measure 30 --device-id 0 \
    --work-dir $OUT/work_resnet_$name --out-json $OUT/resnet_$name.json --out-csv $OUT/resnet_$name.csv \
    > $OUT/resnet_$name.log 2>&1; echo "resnet_$name rc=$? $(date -Iseconds)"; }
done
echo DCU_D1_DONE
