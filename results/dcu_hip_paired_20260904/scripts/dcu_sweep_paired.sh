#!/usr/bin/env bash
# Paired DCU sweep: seven models x {1,8,16} on BOTH the SYCL backend and the
# TVM ROCm/HIP backend, from one libtvm.so that has both runtimes.
#
# Protocol mirrors the published DCU SYCL sweep exactly:
#   compact models : seed 123, static_shape_tuning, 0 trials, 5 warmup, 30 measure
#   resnets        : seed 321, onnx frontend, 224px, 1000 classes, BN folded
# For the compact models the input seed depends on the target name, so the ROCm
# runs pass --seed-target-alias sycl to get bit-identical inputs.
set -uo pipefail

SRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
export TVM_LIBRARY_PATH=/root/tvm_rocm_build
export PYTHONPATH="${SRC}/python"
EXP=/workspace/supplement/experiments
OUT=/root/rocm_baseline
mkdir -p "${OUT}"

ROCM_TS="rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 -max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64"

echo "############ 1/4 compact models on ROCm ############"
python3 "${EXP}/relax_model_probe.py" \
  --models mlp,cnn,transformer,vit --batches 1,8,16 --targets rocm \
  --target-string "${ROCM_TS}" --seed-target-alias sycl \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_compact_rocm" \
  --out-json "${OUT}/compact_rocm.json" --out-csv "${OUT}/compact_rocm.csv" 2>&1 | tail -16

echo "############ 2/4 compact models on SYCL (same binary) ############"
python3 "${EXP}/relax_model_probe.py" \
  --models mlp,cnn,transformer,vit --batches 1,8,16 --targets sycl \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_compact_sycl" \
  --out-json "${OUT}/compact_sycl.json" --out-csv "${OUT}/compact_sycl.csv" 2>&1 | tail -16

echo "############ 3/4 resnets on ROCm ############"
python3 "${EXP}/resnet_relax_probe.py" \
  --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets rocm \
  --target-string "${ROCM_TS}" \
  --image-size 224 --num-classes 1000 --frontend onnx \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_resnet_rocm" \
  --out-json "${OUT}/resnet_rocm.json" --out-csv "${OUT}/resnet_rocm.csv" 2>&1 | tail -16

echo "############ 4/4 resnets on SYCL (same binary) ############"
python3 "${EXP}/resnet_relax_probe.py" \
  --models resnet18,resnet34,resnet50 --batches 1,8,16 --targets sycl \
  --image-size 224 --num-classes 1000 --frontend onnx \
  --pipeline static_shape_tuning --total-trials 0 \
  --warmup 5 --measure 30 --device-id 0 \
  --work-dir "${OUT}/work_resnet_sycl" \
  --out-json "${OUT}/resnet_sycl.json" --out-csv "${OUT}/resnet_sycl.csv" 2>&1 | tail -16

echo "############ SWEEP_DONE ############"
ls -la "${OUT}"/*.json
