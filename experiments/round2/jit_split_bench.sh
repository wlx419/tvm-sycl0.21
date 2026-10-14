#!/usr/bin/env bash
# Time DPC++ compilation of one generated SYCL module: single TU vs. parallel split TUs.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
SRC=${1:-$R/audit/src/q3_resnet18_b1.sycl.cc}
W=$R/jit/splitbench; rm -rf $W; mkdir -p $W; cd $W
C=/mnt/data/wlx/sycl_workspace/DPC++/bin/tvm-sycl-clang++
F="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -Wno-sycl-target -Wno-unknown-cuda-version"
cp $SRC full.cc
echo "kernels: $(grep -c '// Function: ' full.cc)"
t0=$(date +%s.%N); $C $F -shared -DTVM_SYCL_MODULE_ID=mX full.cc -o full.so; t1=$(date +%s.%N)
echo "single TU: $(echo "$t1 - $t0" | bc) s"
# split at the "// CodeGenSYCL: NOTE" lines (sources dumped before the TU markers existed)
python3 - <<'PY'
src = open("full.cc").read()
marker = "// CodeGenSYCL: NOTE: Auto-generated packed function"
parts = src.split(marker)
prelude, chunks = parts[0], [marker + p for p in parts[1:]]
J = 8
units = [prelude for _ in range(J)]
for i, c in enumerate(sorted(chunks, key=len, reverse=True)):
    j = min(range(J), key=lambda k: len(units[k]))
    units[j] += "\n" + c
for j, u in enumerate(units):
    open(f"tu{j}.cc", "w").write(u)
print("chunks", len(chunks))
PY
t0=$(date +%s.%N)
for j in 0 1 2 3 4 5 6 7; do ( $C $F -fno-sycl-rdc -c -DTVM_SYCL_MODULE_ID=mX tu$j.cc -o tu$j.o; echo "tu$j $(date +%s.%N)" ) & done; wait
t1=$(date +%s.%N)
$C $F -shared -fno-sycl-rdc tu*.o -o split.so; t2=$(date +%s.%N)
echo "split compile (8 parallel): $(echo "$t1 - $t0" | bc) s; link: $(echo "$t2 - $t1" | bc) s"
# where does single-TU time go?
t0=$(date +%s.%N); $C $F -fsycl-device-only -DTVM_SYCL_MODULE_ID=mX full.cc -o dev.bc 2>/dev/null; t1=$(date +%s.%N)
echo "device-only frontend: $(echo "$t1 - $t0" | bc) s"
t0=$(date +%s.%N); $C $F -c -fno-sycl-rdc -DTVM_SYCL_MODULE_ID=mX full.cc -o full.o -ftime-report 2> timereport.txt; t1=$(date +%s.%N)
echo "single TU -c -fno-sycl-rdc: $(echo "$t1 - $t0" | bc) s"
