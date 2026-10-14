#!/usr/bin/env bash
# Compare the machine code that DPC++ 646ab08 (env r2) and DPC++ v7.1.1 (env v711) produce for the
# same generated SYCL module, with the JIT flags of the runtime (sycl_common.h DefaultJitFlags):
# per kernel, ptxas registers / spill stores and the SASS instruction count, matched across the two
# builds by TVM's kernel hash (h<hash>_m<module> in both the named and the lambda kernel names).
# usage: tcmp_toolchains.sh <work dir> <generated .src file>   (WD: env_4090.sh sets D)
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
WD=$1; SRC=$2
CUDA=/usr/local/cuda-12.2/bin
FL="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -shared -Wno-sycl-target -Wno-unknown-cuda-version -Xclang -mllvm -Xclang -enable-global-offset=false -ffp-contract=fast -DTVM_SYCL_MODULE_ID=m0"
mkdir -p $WD
for tc in r2 v711; do
  rm -rf $WD/t_$tc; mkdir -p $WD/t_$tc
  cp $SRC $WD/t_$tc/k.cc
  ( source $E/env_4090.sh $tc > /dev/null 2>&1
    cd $WD/t_$tc && nice -n 19 clang++ $FL -save-temps k.cc -o k.so > build.log 2>&1 ) || { echo "$tc build failed"; grep -m5 error $WD/t_$tc/build.log; continue; }
  ptx=$(grep -l "^\.target sm_89" $WD/t_$tc/*.s 2>/dev/null | head -1)
  [ -n "$ptx" ] || { echo "$tc: no PTX found"; ls $WD/t_$tc; continue; }
  cp $ptx $WD/$tc.ptx
  nice -n 19 $CUDA/ptxas -arch=sm_89 -v $WD/$tc.ptx -o $WD/$tc.cubin > $WD/$tc.ptxas.txt 2>&1
  # kernel -> registers, spill stores
  awk '/Compiling entry function/{match($0,/h[0-9]+_m/); k=substr($0,RSTART,RLENGTH-2)}
       /bytes spill stores/{match($0,/[0-9]+ bytes spill stores/); s=substr($0,RSTART); split(s,a," "); sp=a[1]}
       /Used [0-9]+ registers/{match($0,/Used [0-9]+ registers/); split(substr($0,RSTART),b," "); if(k!="") print k, b[2], sp; k=""}' \
       $WD/$tc.ptxas.txt | sort > $WD/$tc.regs.txt
  # kernel -> SASS instruction count
  $CUDA/cuobjdump -sass $WD/$tc.cubin | awk '/Function :/{match($0,/h[0-9]+_m/); k=substr($0,RSTART,RLENGTH-2)}
       /^ +\/\*[0-9a-f]{4}\*\//{if(k!="") n[k]++} END{for(k in n) print k, n[k]}' | sort > $WD/$tc.sass.txt
done
echo "kernel            regs_r2 regs_v711 spill_r2 spill_v711 sass_r2 sass_v711"
join $WD/r2.regs.txt $WD/v711.regs.txt | join - <(join $WD/r2.sass.txt $WD/v711.sass.txt) | \
  awk '{printf "%-16s %7s %9s %8s %10s %7s %9s\n", $1, $2, $4, $3, $5, $6, $7}'
