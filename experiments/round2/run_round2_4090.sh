#!/usr/bin/env bash
# Round-2 experiments on the RTX 4090 host. Each stage is resumable: it skips work whose
# output already exists. Run a single stage with:  STAGES="e1" bash run_round2_4090.sh
#
# Stages
#   preflight  environment + provenance record; refuses to run if another process uses the GPU
#   build      overlay_r2 build in a separate tree (published build left untouched)
#   e1         whole-model Nsight Systems traces, ResNet-18 B1/B16, CUDA and SYCL, published DBs
#   e3         cross-replay: CUDA DB compiled for SYCL, SYCL DB compiled for CUDA
#   e8e9       grid-limit and rsqrt-ULP probes (standalone SYCL)
#   e10        operator probes with refined error metrics
#   abl        runtime/codegen ablation on the published SYCL DB with the overlay_r2 build:
#              (sync each launch, legacy warp) -> (async, legacy warp) -> (async, fixed allreduce)
#   ems        CUDA retune in the SYCL-equivalent search space (no -arch), seeded
#   iso        fault-isolated runner retunes (CUDA original space, SYCL) with overlay_r2
#   filt       SYCL retune with per-workload minimum block counts from the CUDA DB
#   e6         tile-variant 2x2 for the 1x1 convolution, SYCL and CUDA, with nsys/ncu
#   e12        TVM-OpenCL (and Vulkan if available) default-pipeline sweep
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAGES=${STAGES:-"preflight build e1 e3 e8e9 e10 abl ems iso filt e6 e12"}

# ---- paths (same as experiments/run_paired_resnet18_metaschedule_4090.sh) -----------------
TVM_ROOT=${TVM_ROOT:-/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809}
TVM_BUILD=${TVM_BUILD:-${TVM_ROOT}/build-4090}
TVM_ROOT_R2=${TVM_ROOT_R2:-${TVM_ROOT}-r2}
TVM_BUILD_R2=${TVM_BUILD_R2:-${TVM_ROOT_R2}/build-4090-r2}
OVERLAY_R2=${OVERLAY_R2:-${HERE}/../code/overlay_r2}
PYTHON_BIN=${PYTHON_BIN:-/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python}
DPCPP_CXX=${DPCPP_CXX:-/mnt/data/wlx/sycl_workspace/DPC++/bin/tvm-sycl-clang++}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.2}
PYTHON_PACKAGES=${PYTHON_PACKAGES:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214}
PUBLISHED=${PUBLISHED:-/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024}
OUT=${OUT:-/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012}
SEED=${SEED:-20261012}
TRIALS=${TRIALS:-1024}
CUDA_TARGET=${CUDA_TARGET:-"cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"}
CUDA_TARGET_MS=${CUDA_TARGET_MS:-"cuda -keys=cuda,gpu -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"}
SYCL_TARGET=${SYCL_TARGET:-"sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"}

DPCPP_ROOT=$(cd -- "$(dirname -- "${DPCPP_CXX}")/.." && pwd)
export PATH="$(dirname -- "${DPCPP_CXX}"):${CUDA_ROOT}/bin:${PATH}"
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
unset SYCL_DEVICE_FILTER
mkdir -p "${OUT}"
LOG="${OUT}/run_round2.log"

use_build() {  # $1 = published|r2
  local root build
  if [[ "$1" == r2 ]]; then root=${TVM_ROOT_R2}; build=${TVM_BUILD_R2}; else root=${TVM_ROOT}; build=${TVM_BUILD}; fi
  export PYTHONPATH="${PYTHON_PACKAGES}:${root}/python"
  export TVM_LIBRARY_PATH="${build}"
  export LD_LIBRARY_PATH="${build}:${DPCPP_ROOT}/lib:${DPCPP_ROOT}/lib64:${DPCPP_ROOT}/install/lib:${CUDA_ROOT}/lib64"
}

gpu_idle_or_die() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES}" || true)
  if [[ -n "${apps}" && "${ALLOW_SHARED_GPU:-0}" != 1 ]]; then
    echo "GPU ${CUDA_VISIBLE_DEVICES} is in use by other processes; aborting:" | tee -a "${LOG}"
    echo "${apps}" | tee -a "${LOG}"
    exit 3
  fi
}

probe() {  # probe <out.json> <args...>  (resnet_relax_probe_r2.py)
  local out=$1; shift
  [[ -s "${out}" ]] && { echo "skip ${out}" >> "${LOG}"; return 0; }
  "${PYTHON_BIN}" "${HERE}/resnet_relax_probe_r2.py" --out-json "${out}" "$@" 2>&1 | tail -3 | tee -a "${LOG}"
}

stage_preflight() {
  local f="${OUT}/environment.txt"
  {
    date -Iseconds; hostname
    nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,clocks.sm,memory.total --format=csv
    nvidia-smi -q -d CLOCK | sed -n '1,40p'
    "${CUDA_ROOT}/bin/nvcc" --version | tail -2
    "${DPCPP_CXX}" --version | head -2
    "${PYTHON_BIN}" -c "import sys, torch, torchvision, numpy; print(sys.version); print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'numpy', numpy.__version__)"
    echo "--- published TVM tree vs released overlay (md5)"
    for f2 in src/target/source/codegen_sycl.cc src/runtime/sycl/sycl_module.cc src/runtime/sycl/sycl_device_api.cc \
              src/runtime/sycl/sycl_common.h src/target/target_kind.cc src/meta_schedule/utils.h python/tvm/relax/pipeline.py \
              src/tir/transforms/lower_thread_allreduce.cc; do
      printf "%s  tree=%s  overlay_r2=%s\n" "$f2" "$(md5sum < "${TVM_ROOT}/$f2" | cut -c1-12)" \
        "$( [[ -f "${OVERLAY_R2}/$f2" ]] && md5sum < "${OVERLAY_R2}/$f2" | cut -c1-12 || echo -)"
    done
    ls -la --time-style=long-iso "${TVM_BUILD}/libtvm.so"
    echo "--- sudo for clock locking:"; sudo -n true 2>/dev/null && echo yes || echo no
    echo "--- OpenCL ICDs:"; ls /etc/OpenCL/vendors 2>/dev/null || echo none
    echo "--- Vulkan:"; (ls /usr/include/vulkan/vulkan.h && which glslangValidator spirv-opt) 2>/dev/null || echo "vulkan headers/tools missing"
    echo "--- compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  } > "${f}" 2>&1
  cat "${f}"
}

stage_build() {
  [[ -f "${TVM_BUILD_R2}/libtvm.so" ]] && { echo "skip build" >> "${LOG}"; return 0; }
  rsync -a --exclude 'build*' "${TVM_ROOT}/" "${TVM_ROOT_R2}/"
  rsync -a "${OVERLAY_R2}/" "${TVM_ROOT_R2}/" --exclude README.md --exclude CMakeLists.txt.patch
  mkdir -p "${TVM_BUILD_R2}"
  cp "${TVM_BUILD}/config.cmake" "${TVM_BUILD_R2}/config.cmake"
  if [[ -d /etc/OpenCL/vendors ]]; then echo 'set(USE_OPENCL ON)' >> "${TVM_BUILD_R2}/config.cmake"; fi
  if [[ -f /usr/include/vulkan/vulkan.h ]]; then echo 'set(USE_VULKAN ON)' >> "${TVM_BUILD_R2}/config.cmake"; fi
  (cd "${TVM_BUILD_R2}" && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 && ninja -j 48 > build.log 2>&1)
  ls -la "${TVM_BUILD_R2}/libtvm.so" | tee -a "${LOG}"
}

stage_e1() {  # whole-model traces with the published build and published databases
  use_build published; gpu_idle_or_die
  local d="${OUT}/e1"; mkdir -p "$d"
  for b in 1 16; do
    for t in cuda sycl; do
      local ts=$([[ $t == cuda ]] && echo "${CUDA_TARGET}" || echo "${SYCL_TARGET}")
      [[ -s "$d/${t}_b${b}.nsys-rep" ]] && continue
      TVM_SYCL_SYNC_EACH_LAUNCH=1 TVM_SYCL_LEGACY_WARP_SYNC=1 \
      nsys profile --trace=cuda,osrt --sample=none --cuda-memory-usage=false -o "$d/${t}_b${b}" --force-overwrite true \
        "${PYTHON_BIN}" "${HERE}/resnet_relax_probe_r2.py" --target "$t" --target-string "$ts" --batch "$b" \
          --apply-db-dir "${PUBLISHED}/tune/$t" --warmup 10 --measure 50 --nsys-iters 20 \
          --dump-source "$d/src" --out-json "$d/${t}_b${b}.json" 2>&1 | tail -3 | tee -a "${LOG}"
      nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o "$d/${t}_b${b}" "$d/${t}_b${b}.nsys-rep" > /dev/null 2>&1 || true
    done
  done
}

stage_e3() {
  use_build published; gpu_idle_or_die
  local d="${OUT}/e3"; mkdir -p "$d"
  for r in 1 2 3; do
    probe "$d/sycl_with_cuda_db_run$r.json" --target sycl --target-string "${SYCL_TARGET}" --apply-db-dir "${PUBLISHED}/tune/cuda" --dump-source "$d/src_run$r"
    probe "$d/cuda_with_sycl_db_run$r.json" --target cuda --target-string "${CUDA_TARGET}" --apply-db-dir "${PUBLISHED}/tune/sycl"
  done
}

stage_e8e9() {
  local d="${OUT}/e8e9"; mkdir -p "$d"
  local F="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_89 -Wno-deprecated-declarations"
  for p in sycl_grid_limit sycl_rsqrt_ulp; do
    [[ -x "$d/$p" ]] || "${DPCPP_CXX}" $F "${HERE}/$p.cpp" -o "$d/$p"
    [[ -s "$d/$p.json" ]] || LD_LIBRARY_PATH="${DPCPP_ROOT}/lib:${CUDA_ROOT}/lib64" "$d/$p" > "$d/$p.json" 2> "$d/$p.err"
  done
}

stage_e10() {
  use_build published; gpu_idle_or_die
  local d="${OUT}/e10"; mkdir -p "$d"
  [[ -s "$d/e10_operators_4090.json" ]] || (cd "$d" && "${PYTHON_BIN}" "${HERE}/tvm_sycl_extended_probe_r2.py" --suite correctness \
      --out-dir "$d" --stem e10_operators_4090 > "$d/e10.log" 2>&1)
}

stage_abl() {  # same published SYCL DB, overlay_r2 build, three runtime/codegen settings
  use_build r2; gpu_idle_or_die
  local d="${OUT}/abl"; mkdir -p "$d"
  for r in 1 2 3; do
    TVM_SYCL_SYNC_EACH_LAUNCH=1 TVM_SYCL_LEGACY_WARP_SYNC=1 probe "$d/a0_sync_legacy_run$r.json" --target sycl --target-string "${SYCL_TARGET}" --apply-db-dir "${PUBLISHED}/tune/sycl"
    TVM_SYCL_SYNC_EACH_LAUNCH=0 TVM_SYCL_LEGACY_WARP_SYNC=1 probe "$d/a1_async_legacy_run$r.json" --target sycl --target-string "${SYCL_TARGET}" --apply-db-dir "${PUBLISHED}/tune/sycl"
    TVM_SYCL_SYNC_EACH_LAUNCH=0 TVM_SYCL_LEGACY_WARP_SYNC=0 probe "$d/a2_async_fixed_run$r.json" --target sycl --target-string "${SYCL_TARGET}" --apply-db-dir "${PUBLISHED}/tune/sycl" --dump-source "$d/src_run$r"
    probe "$d/cuda_ref_run$r.json" --target cuda --target-string "${CUDA_TARGET}" --apply-db-dir "${PUBLISHED}/tune/cuda"
  done
}

retune() {  # retune <name> <target> <target-string> [extra args]; then three apply runs
  local name=$1 t=$2 ts=$3; shift 3
  local d="${OUT}/tune_${name}"; mkdir -p "$d"
  [[ -s "$d/database_tuning_record.json" && -s "$d/tune.json" ]] || \
    probe "$d/tune.json" --target "$t" --target-string "$ts" --total-trials "${TRIALS}" --tune-seed "${SEED}" --work-dir "$d" "$@"
  for r in 1 2 3; do probe "$d/apply_run$r.json" --target "$t" --target-string "$ts" --apply-db-dir "$d"; done
}

stage_ems()  { use_build r2; gpu_idle_or_die; retune cuda_matched_space cuda "${CUDA_TARGET_MS}" --runner-max-uses 1; }
stage_iso()  { use_build r2; gpu_idle_or_die
               retune cuda_iso cuda "${CUDA_TARGET}" --runner-max-uses 1
               retune sycl_iso sycl "${SYCL_TARGET}" --runner-max-uses 1; }
stage_filt() { use_build r2; gpu_idle_or_die
               retune sycl_iso_minblocks sycl "${SYCL_TARGET}" --runner-max-uses 1 --min-blocks-json "${HERE}/min_blocks_from_cuda.json"; }

stage_e6() {
  use_build r2; gpu_idle_or_die
  local d="${OUT}/e6"; mkdir -p "$d"
  for t in sycl cuda; do
    local ts=$([[ $t == cuda ]] && echo "${CUDA_TARGET}" || echo "${SYCL_TARGET}")
    [[ -s "$d/e6_$t.json" ]] || nsys profile --trace=cuda --sample=none -o "$d/e6_$t" --force-overwrite true \
      "${PYTHON_BIN}" "${HERE}/e6_tile_variants.py" --db-dir "${PUBLISHED}/tune/sycl" --target "$t" --target-string "$ts" \
        --device "$([[ $t == sycl ]] && echo oneapi || echo cuda)" --dump-source "$d/src" --out-json "$d/e6_$t.json" | tee -a "${LOG}"
    [[ -s "$d/e6_${t}_ncu.csv" ]] || ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size \
      --csv "${PYTHON_BIN}" "${HERE}/e6_tile_variants.py" --db-dir "${PUBLISHED}/tune/sycl" --target "$t" --target-string "$ts" \
        --device "$([[ $t == sycl ]] && echo oneapi || echo cuda)" --warmup 0 --measure 1 --out-json "$d/e6_${t}_ncu_run.json" > "$d/e6_${t}_ncu.csv" 2>&1 || true
  done
}

stage_e12() {
  use_build r2; gpu_idle_or_die
  local d="${OUT}/e12"; mkdir -p "$d"
  for t in opencl vulkan; do
    "${PYTHON_BIN}" -c "import tvm,sys; sys.exit(0 if tvm.device('$t',0).exist else 1)" || { echo "$t unavailable" | tee -a "${LOG}"; continue; }
    for m in resnet18 resnet34 resnet50; do for b in 1 8 16; do
      probe "$d/${t}_${m}_b$b.json" --target "$t" --model "$m" --batch "$b" --pipeline default --warmup 5 --measure 30
    done; done
  done
  # compact models use experiments/relax_model_probe.py; see run_round2_compact.sh
}

for s in ${STAGES}; do
  echo "=== stage $s $(date -Iseconds)" | tee -a "${LOG}"
  "stage_$s"
done
