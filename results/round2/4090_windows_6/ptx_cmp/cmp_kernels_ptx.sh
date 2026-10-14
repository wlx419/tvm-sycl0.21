#!/bin/bash
# Compile two cached SYCL TUs (same kernels, different codegen) and compare per-kernel PTX size and
# ptxas registers. usage: cmp_kernels_ptx.sh <outdir> <a.cc> <b.cc>
source /mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments/env_4090.sh r2 >/dev/null 2>&1
OUT=$1; A=$2; B=$3
FL="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -shared -Wno-sycl-target -Wno-unknown-cuda-version -Xclang -mllvm -Xclang -enable-global-offset=false -ffp-contract=fast"
for v in a b; do
  src=$A; [ $v = b ] && src=$B
  mkdir -p $OUT/$v && cd $OUT/$v && rm -f ./*
  cp $src tr.cc
  nice -n 19 timeout 1500 clang++ $FL -save-temps tr.cc -o tr.so > build.log 2>&1 || { echo "$v build failed"; grep -m3 error build.log; }
  S=$(ls ./*-sm_89-*.s | grep -v "\.s\.s" | head -1)
  nice -n 19 /usr/local/cuda-12.2/bin/ptxas -arch=sm_89 -v "$S" -o /dev/null 2>&1 | awk '/Compiling entry/{match($0,/_ZTS[0-9]+(h[0-9]+)_/,m); k=m[1]} /Used/{match($0,/Used ([0-9]+) registers/,r); print k, r[1]}' | sort > regs.txt
  awk '/^\.visible \.entry|^\.entry|\.entry _ZTS/{match($0,/_ZTS[0-9]+(h[0-9]+)_/,m); k=m[1]; n=0} k!=""{n++} /^}/{if(k!=""){print k, n}; k=""}' "$S" | sort > lines.txt
  # kernel hash -> TVM function name
  grep -B0 -A1 "^// Function: " tr.cc | paste - - - 2>/dev/null | sed -n 's/^\/\/ Function: \([^ \t]*\).*TVM_SYCL_KERNEL_NAME(\(h[0-9]*\),.*/\2 \1/p' | sort > names.txt
done
cd $OUT
join a/regs.txt b/regs.txt | join - a/lines.txt | join - b/lines.txt | join -a1 - a/names.txt | awk '{printf "%-12s regs %3d -> %3d  ptx %5d -> %5d  %s\n", $1, $2, $3, $4, $5, $6}' | sort -k9
