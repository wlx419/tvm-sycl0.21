#!/usr/bin/env bash
# Rebuild and profile selected workloads from the paired MetaSchedule databases.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build-4090}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_ROOT=${DPCPP_ROOT:-/mnt/data/wlx/sycl_workspace/DPC++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
ROOT=${ROOT:?Set ROOT to the paired experiment directory}
WORKLOAD_KEYS=${WORKLOAD_KEYS:?Set WORKLOAD_KEYS to one or more structural hashes}
PROFILE=${PROFILE:-${SCRIPT_DIR}/profile_paired_metaschedule_operator.py}
WARMUP=${WARMUP:-10}
MEASURE=${MEASURE:-100}
NSYS_MEASURE=${NSYS_MEASURE:-20}
WATCHDOG_SECONDS=${WATCHDOG_SECONDS:-3600}

export PATH="${DPCPP_ROOT}/bin:${CUDA_ROOT}/bin:${PATH}"
export PYTHONPATH="${PYTHON_PACKAGES}:${TVM_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_LIBRARY_PATH="${TVM_BUILD}"
export LD_LIBRARY_PATH="${TVM_BUILD}:${DPCPP_ROOT}/lib:${DPCPP_ROOT}/lib64:${DPCPP_ROOT}/install/lib:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
unset SYCL_DEVICE_FILTER

watchdog_pid=""
resume_process() {
  if [[ -n "${PAUSE_PID:-}" ]]; then
    kill -CONT "${PAUSE_PID}" 2>/dev/null || true
  fi
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  fi
}
trap resume_process EXIT HUP INT TERM

if [[ -n "${PAUSE_PID:-}" ]]; then
  kill -STOP "${PAUSE_PID}"
  "${PYTHON_BIN}" -c \
    "import os, signal, time; time.sleep(${WATCHDOG_SECONDS}); os.kill(${PAUSE_PID}, signal.SIGCONT)" &
  watchdog_pid=$!
  sleep 3
fi

for key in ${WORKLOAD_KEYS}; do
  out="${ROOT}/operator_profiles/${key}"
  mkdir -p "${out}/nsys"
  "${PYTHON_BIN}" "${PROFILE}" "${ROOT}" "${key}" \
    --warmup "${WARMUP}" --measure "${MEASURE}" --out-dir "${out}" \
    >"${out}/profile.log" 2>&1

  for target in cuda sycl; do
    report="${out}/nsys/${target}"
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=true --output="${report}" \
      "${PYTHON_BIN}" "${PROFILE}" "${ROOT}" "${key}" \
      --warmup 0 --measure "${NSYS_MEASURE}" --only-target "${target}" \
      --out-dir "${out}/nsys/${target}_run" \
      >"${out}/nsys/${target}_profile.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_api_sum --format csv \
      "${report}.nsys-rep" >"${out}/nsys/${target}_stats.csv" 2>&1
  done
done

echo "Paired MetaSchedule operator profiles complete: ${ROOT}/operator_profiles"
