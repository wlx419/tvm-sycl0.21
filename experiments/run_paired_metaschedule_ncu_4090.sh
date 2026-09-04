#!/usr/bin/env bash
# Collect Nsight Compute hardware counters for the paired MetaSchedule
# convolution profiles. Counters require administrative access because the
# driver is loaded with RmProfilingAdminOnly=1, so ncu runs under sudo while
# the workload itself is rebuilt from the same paired databases used by
# run_paired_metaschedule_profiles_4090.sh.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build-4090}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_ROOT=${DPCPP_ROOT:-/mnt/data/wlx/sycl_workspace/DPC++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
NCU_BIN=${NCU_BIN:-${CUDA_ROOT}/bin/ncu}
ROOT=${ROOT:?Set ROOT to the paired experiment directory}
WORKLOAD_KEYS=${WORKLOAD_KEYS:?Set WORKLOAD_KEYS to one or more structural hashes}
PROFILE=${PROFILE:-${SCRIPT_DIR}/profile_paired_metaschedule_operator.py}
WARMUP=${WARMUP:-3}
MEASURE=${MEASURE:-5}
LAUNCH_SKIP=${LAUNCH_SKIP:-3}
LAUNCH_COUNT=${LAUNCH_COUNT:-3}
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

# Refresh the sudo credential cache once from stdin so that the per-kernel ncu
# invocations below do not each need a password.
sudo -S -p '' -v

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
  sleep 5
fi

SECTIONS=(--section SpeedOfLight --section MemoryWorkloadAnalysis
          --section LaunchStats --section Occupancy)

for key in ${WORKLOAD_KEYS}; do
  out="${ROOT}/operator_profiles/${key}/ncu"
  mkdir -p "${out}"
  for target in cuda sycl; do
    sudo -E env \
      PATH="${PATH}" \
      PYTHONPATH="${PYTHONPATH}" \
      TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR}" \
      "${NCU_BIN}" --csv --target-processes all \
        --launch-skip "${LAUNCH_SKIP}" --launch-count "${LAUNCH_COUNT}" \
        "${SECTIONS[@]}" \
        "${PYTHON_BIN}" "${PROFILE}" "${ROOT}" "${key}" \
        --warmup "${WARMUP}" --measure "${MEASURE}" --only-target "${target}" \
        --out-dir "${out}/${target}_run" \
        >"${out}/${target}_ncu.csv" 2>"${out}/${target}_ncu.log"
    echo "collected ${key} ${target}"
  done
done

echo "Nsight Compute counters complete: ${ROOT}/operator_profiles/*/ncu"
