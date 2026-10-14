#!/usr/bin/env bash
# Build the open-source DPC++ v7.1.1 (intel/llvm tag v7.1.1, 2026-09-15; clang 22, SYCL runtime 9.0)
# with the CUDA plugin, installed side by side with the toolchain of the main results (646ab08 in
# /mnt/data/wlx/sycl_workspace/DPC++), which is left untouched. Source: the release tarball
# https://github.com/intel/llvm/archive/refs/tags/v7.1.1.tar.gz (sha256 in build-v7.1.1.log).
# Run only while no timing window is active (it uses most CPU cores for 1-3 h).
set -euo pipefail
W=/mnt/data/wlx/sycl_workspace
SRC=$W/llvm-v7.1.1
PREFIX=$W/DPC++-v7.1.1
JOBS=${JOBS:-48}
cd $W
sha256sum llvm-v7.1.1.tar.gz
if [ ! -d $SRC ]; then mkdir -p $SRC && tar xzf llvm-v7.1.1.tar.gz -C $SRC --strip-components=1; fi
cd $SRC
# CMake fetches a few dependencies with git; bypass the host's git proxy for this process only
# (the proxy at the configured address times out; the direct route works).
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=
export PATH=/usr/local/cuda-12.2/bin:$PATH
# The new clang compiles libdevice during the build and picks GCC 12, which has no libstdc++ headers
# on this host ('cstddef' file not found); use GCC 11, as the tvm-sycl-clang++ wrapper does.
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu
python3 buildbot/configure.py --cuda -o build \
  --cmake-opt="-DCMAKE_INSTALL_PREFIX=$PREFIX" \
  --cmake-opt="-DCUDAToolkit_ROOT=/usr/local/cuda-12.2" \
  --cmake-opt="-DSYCL_ENABLE_BACKENDS=opencl;cuda"
# (without the last option configure also fetches the Level Zero loader and headers, whose
# sparse git checkout fails here; the build is for the CUDA plugin only)
nice -n 5 python3 buildbot/compile.py -o build -j $JOBS
# Same wrapper as the toolchain of the main results: GCC 11 C++ headers and libraries for host compilation.
sed "s#/mnt/data/wlx/sycl_workspace/DPC++/bin/clang++#$PREFIX/bin/clang++#" \
  $W/DPC++/bin/tvm-sycl-clang++ > $PREFIX/bin/tvm-sycl-clang++
chmod +x $PREFIX/bin/tvm-sycl-clang++
$PREFIX/bin/clang++ --version | head -2
ls $PREFIX/lib | grep -i "libsycl\|ur_adapter" | head
echo DPCPP_V711_DONE
