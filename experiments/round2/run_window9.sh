#!/usr/bin/env bash
# Timing window 9 on RTX 4090 device 1: side-by-side evaluation of the SYCL toolchain. The same r2 sources and
# default configuration, built against DPC++ 646ab08 (the toolchain of the main results, env "r2") and against
# DPC++ v7.1.1 (env "v711"); runs of the two toolchains alternate within each configuration.
#   1. launch and graph-replay microbenchmarks compiled with each toolchain
#   2. compact models (B=1), eager and with command graphs; for v7.1.1 also with L1 off
#      (TVM_SYCL_DIRECT_SUBMIT=0: queue::submit instead of the handler- and event-free launch)
#   3. DLight ResNet-18/50, the retuned ResNet-18 (window-3 database), E11 conv2d and matmul
#   4. TVM-CUDA references of the same session for 2 and 3
#   5. Nsight traces of the retuned ResNet-18 (646ab08 build and CUDA)
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window9; mkdir -p $W/e11 $W/bin
LOG=$W/window.log
T2=/mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2
SYCL_FLAGS_BENCH="-std=c++17 -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --offload-arch=sm_89 -Wno-unknown-cuda-version -Wno-deprecated-declarations -D__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS"
for tc in r2 v711; do
  [ -x $W/bin/launch_$tc ] && [ -x $W/bin/graph_$tc ] && continue
  ( source $E/env_4090.sh $tc > /dev/null 2>&1
    clang++ --version | head -1 | sed "s/^/$tc /" >> $LOG
    clang++ $SYCL_FLAGS_BENCH $E/sycl_launch_overhead.cpp -o $W/bin/launch_$tc &&
    clang++ $SYCL_FLAGS_BENCH $E/sycl_graph_replay_overhead.cpp -o $W/bin/graph_$tc ) || { echo "bench build $tc failed" >> $LOG; exit 4; }
done
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum $T2/build-4090-r2/libtvm*.so $T2/build-4090-v711/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
W3=$R/window3

# ---- 1. microbenchmarks (two processes per toolchain, alternating)
if ! grep -q "^micro done" $LOG; then
  for r in 1 2; do for tc in r2 v711; do
    ( source $E/env_4090.sh $tc > /dev/null 2>&1
      timeout 300 $W/bin/launch_$tc 20000 2>&1 | sed "s/^/launch $tc run=$r /" >> $LOG
      timeout 300 $W/bin/graph_$tc growth 10 2>&1 | sed "s/^/graph $tc run=$r /" >> $LOG
      timeout 300 $W/bin/graph_$tc refin 10 32 2>&1 | grep refin | sed "s/^/graph $tc run=$r /" >> $LOG )
  done; done
  echo "micro done" >> $LOG
fi

# probe <tool> <name> <env...> -- <args>: three processes per toolchain, alternating r2 / v711
# (tool "both" runs both toolchains, "v711" only DPC++ v7.1.1, "cuda" the CUDA target on the r2 build)
probe() {
  local tool=$1 name=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  local tcs=(r2 v711); [ "$tool" = cuda ] && tcs=(r2); [ "$tool" = v711 ] && tcs=(v711)
  local script=$1; shift
  for r in 1 2 3; do for tc in "${tcs[@]}"; do
    local tag=$tc; [ "$tool" = cuda ] && tag=cuda
    local out=$W/${name}_${tag}_run$r.json
    [ -s $out ] && continue
    ( source $E/env_4090.sh $tc > /dev/null 2>&1
      env "${envs[@]}" $PY $E/$script "$@" --out-json $out 2>&1 | tail -1 | sed "s/^/$name $tag /" >> $LOG )
  done; done
}

# ---- 2. compact models
for m in transformer vit cnn mlp; do
  probe both c_${m}_eager X=1 -- compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 10 --measure 50
  probe both c_${m}_graph X=1 -- compact_probe_r2.py --model $m --target sycl --target-string "$ST" --graph --warmup 10 --measure 50
  probe cuda c_${m}_eager X=1 -- compact_probe_r2.py --model $m --target cuda --target-string "$CT" --warmup 10 --measure 50
  probe cuda c_${m}_graph X=1 -- compact_probe_r2.py --model $m --target cuda --target-string "$CT" --graph --warmup 10 --measure 50
  probe v711 c_${m}_eager_l1off TVM_SYCL_DIRECT_SUBMIT=0 -- compact_probe_r2.py --model $m --target sycl --target-string "$ST" --warmup 10 --measure 50
  probe v711 c_${m}_graph_l1off TVM_SYCL_DIRECT_SUBMIT=0 -- compact_probe_r2.py --model $m --target sycl --target-string "$ST" --graph --warmup 10 --measure 50
done
# R2 (periodic re-finalization) off, for the v7.1.1 runtime
probe v711 c_transformer_graph_r2off TVM_SYCL_GRAPH_REFINALIZE=0 -- compact_probe_r2.py --model transformer --target sycl --target-string "$ST" --graph --warmup 10 --measure 50

# ---- 3. ResNets and E11
for m in resnet18 resnet50; do
  probe both dl_${m} X=1 -- resnet_relax_probe_r2.py --model $m --target sycl --target-string "$ST" --pipeline default --warmup 10 --measure 50
  probe cuda dl_${m} X=1 -- resnet_relax_probe_r2.py --model $m --target cuda --target-string "$CT" --pipeline default --warmup 10 --measure 50
done
probe both rn18_tuned_eager X=1 -- resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 50
probe both rn18_tuned_graph X=1 -- resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph --warmup 10 --measure 50
probe v711 rn18_tuned_eager_l1off TVM_SYCL_DIRECT_SUBMIT=0 -- resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --warmup 10 --measure 50
probe cuda rn18_tuned_eager X=1 -- resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --warmup 10 --measure 50
probe cuda rn18_tuned_graph X=1 -- resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph --warmup 10 --measure 50
for op in conv2d matmul; do
  probe both e11/${op} X=1 -- big_ops_probe_r2.py --op $op --target sycl --target-string "$ST"
  probe cuda e11/${op} X=1 -- big_ops_probe_r2.py --op $op --target cuda --target-string "$CT"
done
# ---- 5. Nsight traces of the retuned ResNet-18 with the final backend (646ab08 build, both targets;
#         per-kernel times of the retune comparison)
mkdir -p $W/e1
for t in sycl cuda; do for g in "" "--graph"; do
  name=rn18_${t}${g:+_graph}; [ -s $W/e1/$name.json ] && continue
  ts=$ST; db=$W3/tune_sycl_pipe_iso; [ $t = cuda ] && { ts=$CT; db=$W3/tune_cuda_iso; }
  ( source $E/env_4090.sh r2 > /dev/null 2>&1
    nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/resnet_relax_probe_r2.py --target $t --target-string "$ts" --apply-db-dir $db $g --warmup 10 --measure 30 --nsys-iters 20 \
      --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/e1 $name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
done; done
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
