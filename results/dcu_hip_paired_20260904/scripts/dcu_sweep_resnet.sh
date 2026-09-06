#!/usr/bin/env bash
# Install onnx, then run the ResNet half of the paired DCU sweep.
# The compact-model half already completed; it is not repeated.
set -uo pipefail

SRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
export TVM_LIBRARY_PATH=/root/tvm_rocm_build
export PYTHONPATH="${SRC}/python"
EXP=/workspace/supplement/experiments
OUT=/root/rocm_baseline

echo "=== env check (onnx + NumPy 1.x for torch 2.1) ==="
python3 -c "import numpy, torch, torchvision, onnx; print('numpy', numpy.__version__, 'torch', torch.__version__, 'onnx', onnx.__version__)" || exit 1

ROCM_TS="rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64"

echo "############ resnets on ROCm ############"
python3 "${EXP}/resnet_relax_probe.py" \
  --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets rocm \
  --target-string "${ROCM_TS}" \
  --image-size 224 --num-classes 1000 --frontend onnx \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_resnet_rocm2" \
  --out-json "${OUT}/resnet_rocm.json" --out-csv "${OUT}/resnet_rocm.csv" 2>&1 | tail -14

echo "############ resnets on SYCL (same binary) ############"
python3 "${EXP}/resnet_relax_probe.py" \
  --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets sycl \
  --image-size 224 --num-classes 1000 --frontend onnx \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_resnet_sycl2" \
  --out-json "${OUT}/resnet_sycl.json" --out-csv "${OUT}/resnet_sycl.csv" 2>&1 | tail -14

echo "############ RESNET_DONE ############"
