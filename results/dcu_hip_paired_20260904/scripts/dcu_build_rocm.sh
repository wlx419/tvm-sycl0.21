#!/usr/bin/env bash
# Build a libtvm.so that has BOTH the SYCL backend and the ROCm runtime, in an
# ISOLATED build dir. The existing build at ${SRC}/build is never touched:
# every published DCU SYCL result depends on that library.
#
# USE_SYCL must stay ON: this fork guards add_subdirectory(ffi) behind
# if(USE_SYCL), so turning SYCL off leaves add_dsymutil() undefined.
set -uo pipefail

SRC=/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0
BUILD=/root/tvm_rocm_build
JOBS=${JOBS:-64}
# /usr/local/bin/cmake is broken (points at /opt/cmake, no Modules dir).
# The pip-installed 3.31 is the one the existing SYCL build used.
CMAKE_BIN=/usr/local/lib/python3.10/site-packages/cmake/data/bin/cmake

echo "whoami: $(whoami)"
echo "cmake:  $(${CMAKE_BIN} --version | head -1)"
df -h / | tail -1

echo "=== SYCL toolchain present? ==="
ls -d /root/sycl_workspace/llvm/build 2>/dev/null || { echo "MISSING SYCL LLVM"; exit 1; }
ls /root/sycl_workspace/llvm/build/bin/clang++ 2>/dev/null || echo "warn: clang++ not found"

mkdir -p "${BUILD}"
cd "${BUILD}" || exit 1

# Start from the known-good SYCL config and only add the ROCm runtime.
cp "${SRC}/build/config.cmake" ./config.cmake || exit 1

python3 - <<'PY'
import re, pathlib
p = pathlib.Path("config.cmake")
s = p.read_text()
pat = re.compile(r'^set\(USE_ROCM\s+.*?\)\s*$', re.M)
line = 'set(USE_ROCM /opt/rocm)'
s = pat.sub(line, s) if pat.search(s) else s + "\n" + line + "\n"
p.write_text(s)
print("--- effective backend flags ---")
for ln in s.splitlines():
    if ln.strip().startswith("set(") and re.search(r'USE_(ROCM|SYCL|LLVM|CUDA)\b', ln):
        print(ln)
PY

echo "=== configure ==="
"${CMAKE_BIN}" "${SRC}" -DCMAKE_BUILD_TYPE=Release > cmake_configure.log 2>&1
rc=$?
echo "configure rc=${rc}"
if [ ${rc} -ne 0 ]; then
  tail -25 cmake_configure.log
  exit 1
fi
grep -iE "Build with ROCM|Build with SYCL|Use llvm-config" cmake_configure.log | head -6

echo "=== make -j${JOBS} (this takes a while) ==="
make -j"${JOBS}" > build.log 2>&1
rc=$?
echo "make rc=${rc}"
if [ ${rc} -ne 0 ]; then
  echo "--- first errors ---"
  grep -iE "error:" build.log | head -25
  tail -25 build.log
  exit 1
fi

ls -la "${BUILD}/libtvm.so"

echo "=== verify rocm runtime is live ==="
TVM_LIBRARY_PATH="${BUILD}" PYTHONPATH="${SRC}/python" python3 - <<'PY'
import tvm, tvm.ffi
print("tvm", tvm.__version__)
print("device_api.rocm:", tvm.ffi.get_global_func("device_api.rocm", allow_missing=True) is not None)
d = tvm.device("rocm", 0)
print("rocm exist:", d.exist)
if d.exist:
    print("target:", tvm.target.Target.from_device(d))
PY

echo "BUILD_OK"
