#!/usr/bin/env bash
# Window 6e on RTX 4090 device 1:
# the three forms of the work-item / work-group indices, measured on the same models:
#   p   : nd_item indices (TVM_SYCL_NATIVE_IDS=0)
#   m31 : native PTX registers masked to 31 bits (default)
#   hw  : native PTX registers masked to the hardware ranges (TVM_SYCL_NATIVE_IDS=hw)
# Models: retuned and published-database ResNet-18, DLight ResNet-18/50, Transformer/ViT
# (graph replay), E11 matmul; plus a trace of the retuned ResNet-18 with m31.
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window6e; mkdir -p $W/e1
LOG=$W/window.log
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
W3=$R/window3
rn3() {  # rn3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 \
        --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
compact3() {  # compact3 <name> <env...> -- <compact args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --warmup 10 --measure 50 --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
for v in p m31 hw; do
  case $v in p) ev=TVM_SYCL_NATIVE_IDS=0;; m31) ev=X=1;; hw) ev=TVM_SYCL_NATIVE_IDS=hw;; esac
  rn3 rn18_iso_$v $ev -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso
  rn3 rn18_pub_$v $ev -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl
  rn3 rn18_dlight_$v $ev -- --target sycl --target-string "$ST" --pipeline default
  rn3 rn50_dlight_$v $ev -- --model resnet50 --target sycl --target-string "$ST" --pipeline default
  rn3 rn50_dlight_b16_$v $ev -- --model resnet50 --batch 16 --target sycl --target-string "$ST" --pipeline default
  for m in transformer vit; do
    compact3 ${m}_graph_$v $ev -- --model $m --target sycl --target-string "$ST" --graph
    compact3 ${m}_b16_graph_$v $ev -- --model $m --batch 16 --target sycl --target-string "$ST" --graph
  done
  ( source $E/env_4090.sh r2
    [ -s $W/e11_matmul_$v.json ] || env $ev $PY $E/big_ops_probe_r2.py --op matmul --target sycl --target-string "$ST" --out-json $W/e11_matmul_$v.json 2>&1 | tail -1 | sed "s/^/e11_matmul_$v /" >> $LOG )
done
( source $E/env_4090.sh r2
  nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/iso_sycl_m31 --force-overwrite true \
    $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 30 --nsys-iters 20 --out-json $W/e1/iso_sycl_m31.json 2>&1 | tail -1 | sed "s/^/e1 iso_sycl_m31 /" >> $LOG
  nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/iso_sycl_m31 $W/e1/iso_sycl_m31.nsys-rep > /dev/null 2>&1 )
echo "=== window end $(date -Iseconds)" >> $LOG
