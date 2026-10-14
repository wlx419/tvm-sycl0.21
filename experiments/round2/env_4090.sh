# source env_4090.sh published|r2|next|v711
#   next: the r2 libraries with the Python package from $R/tree_next/python (staging copy used to
#         test Python-side changes while a timing window runs on the r2 tree)
#   v711: the r2 sources built against DPC++ v7.1.1 (side-by-side toolchain evaluation)
# GPU_UUID overrides the device (default: device 1).
T=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
MODE=${1:-published}
if [ "$MODE" = r2 ] || [ "$MODE" = next ]; then T=${T}-r2; B=$T/build-4090-r2;
elif [ "$MODE" = v711 ]; then T=${T}-r2; B=$T/build-4090-v711;
else B=$T/build-4090; fi
PYDIR=$T/python
[ "$MODE" = next ] && PYDIR=$R/tree_next/python
D=/mnt/data/wlx/sycl_workspace/DPC++
[ "$MODE" = v711 ] && D=/mnt/data/wlx/sycl_workspace/DPC++-v7.1.1
export PATH=$D/bin:/usr/local/cuda-12.2/bin:$PATH
export PYTHONPATH=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/python_pkgs/xgboost214:$PYDIR
export TVM_LIBRARY_PATH=$B
export LD_LIBRARY_PATH=$B:$D/lib:$D/lib64:$D/install/lib:/usr/local/cuda-12.2/lib64
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/x86_64-linux-gnu/c++/11:/usr/include/c++/11/backward
export LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/11:/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=${GPU_UUID:-GPU-e5e525e1-6c5a-fb72-18b0-3e59e3ca9983}
export ONEAPI_DEVICE_SELECTOR='cuda:*'
unset SYCL_DEVICE_FILTER
PY=/mnt/data/wlx/anaconda3/envs/tvm-sycl/bin/python
