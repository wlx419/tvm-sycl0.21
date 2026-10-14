#!/bin/bash
# Build variants of the transformer SYCL TU and compare the matmul kernel's PTX and registers.
source /mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments/env_4090.sh r2 >/dev/null 2>&1
D=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/ptx_cmp
BASE="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -shared -Wno-sycl-target -Wno-unknown-cuda-version -Xclang -mllvm -Xclang -enable-global-offset=false -ffp-contract=fast"
KS="h1350265394_m0 h3276024936_m0 h1951265625_m0"

mk_variant() {  # $1 name
  case $1 in
    base) cp $D/tr_sycl.cpp $D/$1/tr_sycl.cpp ;;
    nvvm)
      { cat <<'EOF'
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define TVM_SYCL_LOCAL_ID(item, d) ((d) == 2 ? (int)__nvvm_read_ptx_sreg_tid_x() : (d) == 1 ? (int)__nvvm_read_ptx_sreg_tid_y() : (int)__nvvm_read_ptx_sreg_tid_z())
#else
#define TVM_SYCL_LOCAL_ID(item, d) ((int)(item).get_local_id(d))
#endif
EOF
        sed 's/((int)item\.get_local_id(\([0-2]\)))/TVM_SYCL_LOCAL_ID(item, \1)/g' $D/tr_sycl.cpp; } > $D/$1/tr_sycl.cpp ;;
    u32)
      sed 's/((int)item\.get_local_id(\([0-2]\)))/((int)(unsigned)item.get_local_id(\1))/g' $D/tr_sycl.cpp > $D/$1/tr_sycl.cpp ;;
  esac
}

for v in "$@"; do
  mkdir -p $D/$v && cd $D/$v && rm -f ./*
  mk_variant $v
  echo "== $v sed-hits: $(grep -c 'TVM_SYCL_LOCAL_ID(item\|(int)(unsigned)item' tr_sycl.cpp)"
  timeout 900 clang++ $BASE -save-temps tr_sycl.cpp -o tr.so > build.log 2>&1
  echo "rc=$?"; grep -m3 " error" build.log
  S=$(ls ./*-sm_89-*.s 2>/dev/null | head -1)
  [ -z "$S" ] && continue
  for k in $KS; do
    awk "/.entry _ZTS[0-9]*${k}\(/{f=1} f{print} f&&/^}/{exit}" "$S" > k_$k.ptx
    echo "$k fma=$(grep -c fma.rn.f32 k_$k.ptx) add.s64=$(grep -c add.s64 k_$k.ptx) shared=$(grep -c ld.shared k_$k.ptx) lines=$(wc -l < k_$k.ptx)"
  done
  /usr/local/cuda-12.2/bin/ptxas -arch=sm_89 -v "$S" -o /dev/null 2>&1 | grep -A1 "Compiling entry.*\(h1350265394_m0\|h3276024936_m0\|h1951265625_m0\)" | grep -o "Function properties for.*\|Used [0-9]* registers" | paste - - 2>/dev/null
  /usr/local/cuda-12.2/bin/ptxas -arch=sm_89 -v "$S" -o /dev/null 2>&1 | awk '/Compiling entry/{n=$0} /Used/{print n" :: "$0}' | grep "h1350265394_m0\|h3276024936_m0\|h1951265625_m0" | sed 's/ptxas info    ://g'
done
