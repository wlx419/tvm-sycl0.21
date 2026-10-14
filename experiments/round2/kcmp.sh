#!/usr/bin/env bash
# Compare one kernel's machine code between the SYCL build (final JIT flags; nd_item indices and N1)
# and nvcc's build of TVM-CUDA's source for the same schedule: registers, spills, SASS opcode counts.
# usage: kcmp.sh <dir> <sycl.cpp> <cuda.cu> <tvm function name, e.g. conv2d_kernel>
source /mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments/env_4090.sh r2 >/dev/null 2>&1
D=$1; SRC=$2; CU=$3; FN=$4
FL="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -shared -Wno-sycl-target -Wno-unknown-cuda-version -Xclang -mllvm -Xclang -enable-global-offset=false -ffp-contract=fast"
CUDA=/usr/local/cuda-12.2/bin
mkdir -p $D && cd $D
awk '/^\/\/ Function: /{exit} {print}' $SRC > prelude.h
awk -v fn="$FN" '$0=="// Function: "fn{f=1; print; next} f&&/^\/\/ Function: /{exit} f{print}' $SRC > kern.inc
hist() { $CUDA/cuobjdump -sass ${2:+-fun $2} $1 | grep -oP '^\s+/\*[0-9a-f]{4}\*/\s+(@!?U?P[T0-9]\s+)?\K[A-Z][A-Z0-9_.]*' | sed 's/\..*//' | sort | uniq -c | sort -rn; }
for v in p n1; do
  { [ $v = n1 ] && grep -v '^#define TVM_SYCL_PORTABLE_IDS$' prelude.h || cat prelude.h; cat kern.inc; } > k_$v.cc
  rm -rf t_$v; mkdir t_$v
  ( cd t_$v && nice -n 19 clang++ $FL -save-temps ../k_$v.cc -o k.so > ../b_$v.log 2>&1 ) || { echo "$v build failed"; grep -m3 error b_$v.log; continue; }
  cp $(ls t_$v/*-sm_89-*.s | head -1) k_$v.ptx
  nice -n 19 $CUDA/ptxas -arch=sm_89 -v k_$v.ptx -o k_$v.cubin 2>&1 | grep -A2 "Compiling entry.*_ZTS" | grep -o "[0-9]* bytes spill stores\|Used [0-9]* registers" | tr '\n' ' ' | sed "s/^/sycl_$v: /"; echo
  hist k_$v.cubin > h_$v.txt
done
cp $CU k.cu
nice -n 19 $CUDA/nvcc -O3 -arch=sm_89 -cubin -Xptxas -v k.cu -o k_cuda.cubin 2>&1 | grep -A2 "$FN" | grep -o "[0-9]* bytes spill stores\|Used [0-9]* registers" | tr '\n' ' ' | sed 's/^/cuda: /'; echo
hist k_cuda.cubin $FN > h_cuda.txt
echo "opcode           cuda  sycl_p  sycl_n1"
cat h_cuda.txt h_p.txt h_n1.txt | awk '{print $2}' | sort -u | while read op; do
  c=$(awk -v o=$op '$2==o{print $1}' h_cuda.txt); p=$(awk -v o=$op '$2==o{print $1}' h_p.txt); n=$(awk -v o=$op '$2==o{print $1}' h_n1.txt)
  printf "%-14s %6s %7s %8s\n" $op ${c:-0} ${p:-0} ${n:-0}
done | sort -k2 -n -r | head -30
echo "total: cuda $(awk '{s+=$1} END{print s}' h_cuda.txt) sycl_p $(awk '{s+=$1} END{print s}' h_p.txt) sycl_n1 $(awk '{s+=$1} END{print s}' h_n1.txt)"
