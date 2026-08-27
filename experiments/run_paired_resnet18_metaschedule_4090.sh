#!/usr/bin/env bash
# Paired TVM v0.21 Relax/MetaSchedule ResNet-18 experiment on RTX 4090.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build-4090}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_CXX=${DPCPP_CXX:-/mnt/data/wlx/sycl_workspace/DPC++/bin/tvm-sycl-clang++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
OUT_ROOT=${OUT_ROOT:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813}
TRIALS=${TRIALS:-1024}
IMAGE_SIZE=${IMAGE_SIZE:-224}
WARMUP=${WARMUP:-10}
MEASURE=${MEASURE:-50}
APPLY_RUNS=${APPLY_RUNS:-3}
TARGETS=${TARGETS:-cuda sycl}
PHASES=${PHASES:-tune apply summarize}
WATCHDOG_SECONDS=${WATCHDOG_SECONDS:-10800}
CUDA_TARGET_STRING=${CUDA_TARGET_STRING:-cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32}
SYCL_TARGET_STRING=${SYCL_TARGET_STRING:-sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32}

DPCPP_ROOT=$(cd -- "$(dirname -- "${DPCPP_CXX}")/.." && pwd)
export PATH="$(dirname -- "${DPCPP_CXX}"):${CUDA_ROOT}/bin:${PATH}"
export PYTHONPATH="${PYTHON_PACKAGES}:${TVM_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_LIBRARY_PATH="${TVM_BUILD}"
export LD_LIBRARY_PATH="${TVM_BUILD}:${DPCPP_ROOT}/lib:${DPCPP_ROOT}/lib64:${DPCPP_ROOT}/install/lib:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
unset SYCL_DEVICE_FILTER

PROBE=${PROBE:-${SCRIPT_DIR}/resnet_relax_probe.py}
SUMMARIZER=${SUMMARIZER:-${SCRIPT_DIR}/summarize_paired_resnet18_metaschedule.py}
watchdog_pid=""

resume_process() {
  if [[ -n "${PAUSE_PID:-}" ]]; then
    kill -CONT "${PAUSE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
    watchdog_pid=""
  fi
}

trap resume_process EXIT HUP INT TERM
if [[ -n "${PAUSE_PID:-}" ]]; then
  kill -STOP "${PAUSE_PID}"
  "${PYTHON_BIN}" -c \
    "import os, signal, time; time.sleep(${WATCHDOG_SECONDS}); os.kill(${PAUSE_PID}, signal.SIGCONT)" &
  watchdog_pid="$!"
  sleep 3
fi

mkdir -p "${OUT_ROOT}/tune" "${OUT_ROOT}/apply"
{
  date --iso-8601=seconds
  hostname
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "${DPCPP_CXX}" --version | head -n 1
  "${PYTHON_BIN}" - <<'PY'
import torch
import torchvision
import tvm
print(f"TVM {tvm.__version__}")
print(f"PyTorch {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
PY
} >"${OUT_ROOT}/environment.txt" 2>&1

contains_word() {
  [[ " ${1} " == *" ${2} "* ]]
}

target_string() {
  if [[ "$1" == "cuda" ]]; then
    printf '%s' "${CUDA_TARGET_STRING}"
  else
    printf '%s' "${SYCL_TARGET_STRING}"
  fi
}

result_passes() {
  local path=$1
  [[ -f "${path}" ]] || return 1
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1], encoding="utf-8")).get("results", [])
raise SystemExit(0 if len(rows) == 1 and rows[0].get("status") == "pass" else 1)
PY
}

if contains_word "${PHASES}" tune; then
  for target in ${TARGETS}; do
    work_dir="${OUT_ROOT}/tune/${target}"
    mkdir -p "${work_dir}"
    if result_passes "${work_dir}/probe.json"; then
      echo "[skip] tuned ${target} result already passes"
      continue
    fi
    if [[ -e "${work_dir}/database_workload.json" || -e "${work_dir}/database_tuning_record.json" ]]; then
      echo "Refusing to reuse incomplete tuning database: ${work_dir}" >&2
      exit 1
    fi
    echo "[tune] ${target}: ${TRIALS} trials"
    "${PYTHON_BIN}" "${PROBE}" \
      --models resnet18 --batches 1 --targets "${target}" \
      --target-string "$(target_string "${target}")" \
      --pipeline static_shape_tuning --total-trials "${TRIALS}" \
      --work-dir "${work_dir}" --image-size "${IMAGE_SIZE}" \
      --warmup "${WARMUP}" --measure "${MEASURE}" \
      --out-json "${work_dir}/probe.json" --out-csv "${work_dir}/probe.csv" \
      >"${work_dir}/probe.log" 2>&1
    result_passes "${work_dir}/probe.json"
  done
fi

if contains_word "${PHASES}" apply; then
  for target in ${TARGETS}; do
    work_dir="${OUT_ROOT}/tune/${target}"
    [[ -f "${work_dir}/database_workload.json" ]]
    [[ -f "${work_dir}/database_tuning_record.json" ]]
    mkdir -p "${OUT_ROOT}/apply/${target}"
    for run in $(seq 1 "${APPLY_RUNS}"); do
      stem="${OUT_ROOT}/apply/${target}/run${run}"
      if result_passes "${stem}.json"; then
        echo "[skip] apply ${target} run ${run} already passes"
        continue
      fi
      echo "[apply] ${target}: run ${run}/${APPLY_RUNS}"
      "${PYTHON_BIN}" "${PROBE}" \
        --models resnet18 --batches 1 --targets "${target}" \
        --target-string "$(target_string "${target}")" \
        --pipeline static_shape_tuning --total-trials 0 \
        --work-dir "${work_dir}" --image-size "${IMAGE_SIZE}" \
        --warmup "${WARMUP}" --measure "${MEASURE}" \
        --out-json "${stem}.json" --out-csv "${stem}.csv" \
        >"${stem}.log" 2>&1
      result_passes "${stem}.json"
    done
  done
fi

if contains_word "${PHASES}" summarize; then
  "${PYTHON_BIN}" "${SUMMARIZER}" \
    --root "${OUT_ROOT}" --apply-runs "${APPLY_RUNS}" \
    --trials "${TRIALS}" --image-size "${IMAGE_SIZE}"
fi

resume_process
trap - EXIT HUP INT TERM
echo "Paired Relax/MetaSchedule experiment complete: ${OUT_ROOT}"
