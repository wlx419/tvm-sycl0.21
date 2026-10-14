#!/bin/bash
# Single-kernel TUs from a cached SYCL TU (N1 build) in index variants; ptxas registers/spills.
# usage: one_kernel_variants.sh <tu.cc> <outdir> <tvm function name> [extra clang flags...]
source /mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments/env_4090.sh r2 >/dev/null 2>&1
TU=$1; OUT=$2; FN=$3; shift 3; EXTRA="$*"
FL="-std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -fPIC -shared -Wno-sycl-target -Wno-unknown-cuda-version -Xclang -mllvm -Xclang -enable-global-offset=false -ffp-contract=fast"
mkdir -p $OUT; cd $OUT
# prelude: everything before the first kernel; kernel: from its "// Function:" line to the next one
awk '/^\/\/ Function: /{exit} {print}' $TU > prelude.h
awk -v fn="$FN" '$0=="// Function: "fn{f=1; print; next} f&&/^\/\/ Function: /{exit} f{print}' $TU > kern.inc
for v in ${VARIANTS:-P N NL NG}; do
  case $v in
    M|MA|M31) defs="" ;;
    P)  defs="#define TVM_SYCL_PORTABLE_IDS" ;;
    N)  defs="" ;;
    NL) defs="#define TVM_SYCL_GROUP_ID_OVERRIDE" ;;
    NG) defs="#define TVM_SYCL_LOCAL_ID_OVERRIDE" ;;
  esac
  { echo "$defs"; cat prelude.h
    [ $v = NL ] && echo '#undef TVM_SYCL_GROUP_ID
#define TVM_SYCL_GROUP_ID(d) ((int)item.get_group(d))'
    [ $v = NG ] && echo '#undef TVM_SYCL_LOCAL_ID
#define TVM_SYCL_LOCAL_ID(d) ((int)item.get_local_id(d))'
    # M: native ids masked to their hardware ranges (non-negative, bounded)
    [ $v = M ] && echo '#undef TVM_SYCL_LOCAL_ID_2
#undef TVM_SYCL_LOCAL_ID_1
#undef TVM_SYCL_LOCAL_ID_0
#undef TVM_SYCL_GROUP_ID_2
#undef TVM_SYCL_GROUP_ID_1
#undef TVM_SYCL_GROUP_ID_0
#define TVM_SYCL_LOCAL_ID_2 ((int)(__nvvm_read_ptx_sreg_tid_x() & 0x3ff))
#define TVM_SYCL_LOCAL_ID_1 ((int)(__nvvm_read_ptx_sreg_tid_y() & 0x3ff))
#define TVM_SYCL_LOCAL_ID_0 ((int)(__nvvm_read_ptx_sreg_tid_z() & 0x3f))
#define TVM_SYCL_GROUP_ID_2 ((int)(__nvvm_read_ptx_sreg_ctaid_x() & 0x7fffffff))
#define TVM_SYCL_GROUP_ID_1 ((int)(__nvvm_read_ptx_sreg_ctaid_y() & 0xffff))
#define TVM_SYCL_GROUP_ID_0 ((int)(__nvvm_read_ptx_sreg_ctaid_z() & 0xffff))'
    # M31: native ids masked to 31 bits (non-negative only; no narrow value range)
    [ $v = M31 ] && echo '#undef TVM_SYCL_LOCAL_ID_2
#undef TVM_SYCL_LOCAL_ID_1
#undef TVM_SYCL_LOCAL_ID_0
#undef TVM_SYCL_GROUP_ID_2
#undef TVM_SYCL_GROUP_ID_1
#undef TVM_SYCL_GROUP_ID_0
#define TVM_SYCL_LOCAL_ID_2 ((int)(__nvvm_read_ptx_sreg_tid_x() & 0x7fffffff))
#define TVM_SYCL_LOCAL_ID_1 ((int)(__nvvm_read_ptx_sreg_tid_y() & 0x7fffffff))
#define TVM_SYCL_LOCAL_ID_0 ((int)(__nvvm_read_ptx_sreg_tid_z() & 0x7fffffff))
#define TVM_SYCL_GROUP_ID_2 ((int)(__nvvm_read_ptx_sreg_ctaid_x() & 0x7fffffff))
#define TVM_SYCL_GROUP_ID_1 ((int)(__nvvm_read_ptx_sreg_ctaid_y() & 0x7fffffff))
#define TVM_SYCL_GROUP_ID_0 ((int)(__nvvm_read_ptx_sreg_ctaid_z() & 0x7fffffff))'
    # MA: native ids with __builtin_assume ranges
    [ $v = MA ] && echo '#undef TVM_SYCL_LOCAL_ID_2
#undef TVM_SYCL_LOCAL_ID_1
#undef TVM_SYCL_LOCAL_ID_0
#undef TVM_SYCL_GROUP_ID_2
#undef TVM_SYCL_GROUP_ID_1
#undef TVM_SYCL_GROUP_ID_0
#define TVM_SYCL_RANGED(e, hi) ({ int _id_ = (int)(e); __builtin_assume(_id_ >= 0 && _id_ < (hi)); _id_; })
#define TVM_SYCL_LOCAL_ID_2 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_tid_x(), 1024)
#define TVM_SYCL_LOCAL_ID_1 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_tid_y(), 1024)
#define TVM_SYCL_LOCAL_ID_0 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_tid_z(), 64)
#define TVM_SYCL_GROUP_ID_2 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_ctaid_x(), 0x7fffffff)
#define TVM_SYCL_GROUP_ID_1 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_ctaid_y(), 65536)
#define TVM_SYCL_GROUP_ID_0 TVM_SYCL_RANGED(__nvvm_read_ptx_sreg_ctaid_z(), 65536)'
    cat kern.inc; } > k_$v.cc
  rm -rf t_$v; mkdir t_$v; ( cd t_$v && nice -n 19 clang++ $FL $EXTRA -save-temps ../k_$v.cc -o k.so > ../b_$v.log 2>&1 ) || { echo "$v failed"; grep -m3 error b_$v.log; continue; }
  cp $(ls t_$v/*-sm_89-*.s | head -1) k_$v.ptx
  r=$(nice -n 19 /usr/local/cuda-12.2/bin/ptxas -arch=sm_89 -v k_$v.ptx -o /dev/null 2>&1 | awk '/Compiling entry.*_ZTS/{k=1} k&&/spill stores/{match($0,/([0-9]+) bytes spill stores/,s)} k&&/Used/{match($0,/Used ([0-9]+) registers/,r); print "regs=" r[1] " spill=" s[1]; exit}')
  echo "$FN $v $r ptx=$(grep -c . k_$v.ptx)"
done
