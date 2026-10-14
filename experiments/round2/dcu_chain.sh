#!/usr/bin/env bash
# DCU: wait for run_dcu_r2b.sh, rebuild overlay_r2 (latest sources), smoke test, final sweep.
while ps -eo args | grep -q "^bash /root/r2/run_dcu_r2b.sh"; do sleep 30; done
cd /root/r2 && rsync -a --checksum --exclude /cmake/ --exclude /README.md --exclude /CMakeLists.txt.patch overlay_r2/ tvm/
cd /root/r2/tvm/build && make -j96 > /root/r2/build_final.log 2>&1; echo "BUILD_EXIT=$?"
export LD_LIBRARY_PATH=/root/r2/tvm/build:/root/sycl_workspace/llvm/build/install/lib:/root/sycl_workspace/llvm/build/lib:/opt/dtk/lib:/opt/dtk/hip/lib
HIP_VISIBLE_DEVICES=0 SYCL_DEVICE_FILTER=hip:gpu:0 TVM_LIBRARY_PATH=/root/r2/tvm/build PYTHONPATH=/root/r2/tvm/python \
  python3 -c "import tvm; d=tvm.device('oneapi',0); print('smoke', d.exist, tvm.target.Target.from_device(d))"
bash /root/r2/run_dcu_final.sh
