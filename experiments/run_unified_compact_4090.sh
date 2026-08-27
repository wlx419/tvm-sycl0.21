#!/usr/bin/env bash
# Formal RTX 4090 run for compact TVM v0.21 Relax models.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_CXX=${DPCPP_CXX:-/mnt/data/wlx/sycl_workspace/DPC++/bin/tvm-sycl-clang++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
OUT_ROOT=${OUT_ROOT:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/unified_compact_relax}
TRIALS=${TRIALS:-1024}
WARMUP=${WARMUP:-5}
MEASURE=${MEASURE:-30}
RUN_COVERAGE=${RUN_COVERAGE:-1}
RUN_SWEEP=${RUN_SWEEP:-1}
SYCL_TARGET_STRING=${SYCL_TARGET_STRING:-sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32}

DPCPP_ROOT=$(cd -- "$(dirname -- "${DPCPP_CXX}")/.." && pwd)
export PATH="$(dirname -- "${DPCPP_CXX}"):${CUDA_ROOT}/bin:${PATH}"
export PYTHONPATH="${PYTHON_PACKAGES}:${TVM_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export TVM_LIBRARY_PATH="${TVM_BUILD}"
export LD_LIBRARY_PATH="${TVM_BUILD}:${DPCPP_ROOT}/lib:${DPCPP_ROOT}/lib64:${DPCPP_ROOT}/install/lib:${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}

PROBE=${PROBE:-${SCRIPT_DIR}/relax_model_probe.py}
SUMMARIZER=${SUMMARIZER:-${SCRIPT_DIR}/summarize_unified_compact.py}
MODELS=(mlp cnn transformer vit)
BATCHES=(1 8 16)
TARGETS=(cuda sycl)
failures=0

mkdir -p "${OUT_ROOT}/coverage" "${OUT_ROOT}/sweep/cuda" "${OUT_ROOT}/sweep/sycl"

{
  date --iso-8601=seconds
  hostname
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "${DPCPP_CXX}" --version | head -n 1
  "${PYTHON_BIN}" - <<'PY'
import tvm

print(f"TVM {tvm.__version__}")
PY
} >"${OUT_ROOT}/environment.txt" 2>&1

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

coverage_usable() {
  local work_dir=$1
  "${PYTHON_BIN}" - "${work_dir}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
workload_path = root / "database_workload.json"
record_path = root / "database_tuning_record.json"
if not workload_path.exists() or not record_path.exists():
    raise SystemExit(1)
workloads = [line for line in workload_path.read_text(encoding="utf-8").splitlines() if line]
covered = set()
for line in record_path.read_text(encoding="utf-8").splitlines():
    if not line:
        continue
    record = json.loads(line)
    try:
        values = [float(value) for value in record[1][1]]
    except (IndexError, TypeError, ValueError):
        continue
    if any(math.isfinite(value) and value < 1e9 for value in values):
        covered.add(int(record[0]))
raise SystemExit(0 if workloads and covered else 1)
PY
}

run_coverage() {
  local model=$1
  local work_dir="${OUT_ROOT}/coverage/${model}"
  local result="${work_dir}/probe.json"
  mkdir -p "${work_dir}"
  if result_passes "${result}" && coverage_usable "${work_dir}"; then
    echo "[skip] coverage ${model}: existing complete passing result"
    return
  fi
  echo "[run] coverage ${model}: ${TRIALS} MetaSchedule trials"
  "${PYTHON_BIN}" "${PROBE}" \
    --models "${model}" --batches 1 --targets sycl \
    --target-string "${SYCL_TARGET_STRING}" \
    --pipeline static_shape_tuning --total-trials "${TRIALS}" \
    --work-dir "${work_dir}" --warmup "${WARMUP}" --measure "${MEASURE}" \
    --out-json "${result}" --out-csv "${work_dir}/probe.csv" \
    >"${work_dir}/probe.log" 2>&1
  if ! result_passes "${result}" || ! coverage_usable "${work_dir}"; then
    echo "[fail] coverage ${model}; see ${work_dir}/probe.log" >&2
    failures=$((failures + 1))
  fi
}

run_sweep_point() {
  local target=$1
  local model=$2
  local batch=$3
  local stem="${OUT_ROOT}/sweep/${target}/${model}_b${batch}"
  local target_args=()
  if [[ "${target}" == sycl ]]; then
    target_args=(--target-string "${SYCL_TARGET_STRING}")
  fi
  if result_passes "${stem}.json"; then
    echo "[skip] ${target} ${model} B${batch}: existing passing result"
    return
  fi
  echo "[run] ${target} ${model} B${batch}"
  "${PYTHON_BIN}" "${PROBE}" \
    --models "${model}" --batches "${batch}" --targets "${target}" \
    --seed-target-alias sycl --pipeline default \
    "${target_args[@]}" \
    --warmup "${WARMUP}" --measure "${MEASURE}" \
    --out-json "${stem}.json" --out-csv "${stem}.csv" \
    >"${stem}.log" 2>&1
  if ! result_passes "${stem}.json"; then
    echo "[fail] ${target} ${model} B${batch}; see ${stem}.log" >&2
    failures=$((failures + 1))
  fi
}

if [[ "${RUN_COVERAGE}" == 1 ]]; then
  for model in "${MODELS[@]}"; do
    run_coverage "${model}"
  done
fi

if [[ "${RUN_SWEEP}" == 1 ]]; then
  for target in "${TARGETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for batch in "${BATCHES[@]}"; do
        run_sweep_point "${target}" "${model}" "${batch}"
      done
    done
  done
fi

if [[ "${RUN_COVERAGE}" == 1 && "${RUN_SWEEP}" == 1 ]]; then
  if ! "${PYTHON_BIN}" "${SUMMARIZER}" --root "${OUT_ROOT}"; then
    failures=$((failures + 1))
  fi
fi

if ((failures > 0)); then
  echo "Unified compact-model run finished with ${failures} failed checks." >&2
  exit 1
fi
echo "Unified compact-model run completed for the requested stages."
