#!/usr/bin/env bash
# Timing window 8 on RTX 4090 device 1. Code after the cleanup and F2 (float min/max as sycl::fmin/fmax).
#   Part A: switch-off/switch-on rows for F2, sub-group-shuffle reductions (C5) and N1 on the compact
#           models (graphs), DLight ResNet-18/50, E11 conv and matmul.
#   Part B: final configuration FINAL_ENV (set below after window 7) on the batch sweep (both targets,
#           eager and graphs), E11, the retuned ResNet-18 (database DB_SYCL), and Transformer/ViT traces.
# usage: run_window8.sh "<FINAL_ENV, e.g. TVM_SYCL_SUBGROUP_REDUCE=1>" <DB_SYCL dir> [A|B|AB]
set -u
FINAL_ENV=${1:-X=1}
DB_SYCL=${2:-}
PARTS=${3:-AB}
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window8; mkdir -p $W/e1 $W/e11
LOG=$W/window.log
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds) FINAL_ENV='$FINAL_ENV' DB_SYCL='$DB_SYCL'" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv >> $LOG
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG

ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
W3=$R/window3
read -r -a FENV <<< "$FINAL_ENV"

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
e11() {  # e11 <name> <op> <target> <ts> <env...>
  local name=$1 op=$2 t=$3 ts=$4; shift 4
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/e11/${name}_run$r.json ] && continue
      env "$@" $PY $E/big_ops_probe_r2.py --op $op --target $t --target-string "$ts" --out-json $W/e11/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/e11 $name /" >> $LOG
    done )
}

# ---- Part A: single switches on top of the current defaults
if [[ $PARTS == *A* ]]; then
for m in transformer vit cnn mlp; do
  compact3 a_${m}_base_graph X=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 a_${m}_noF2_graph TVM_SYCL_FMINMAX=0 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 a_${m}_sg_graph TVM_SYCL_SUBGROUP_REDUCE=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 a_${m}_n1_graph TVM_SYCL_NATIVE_IDS=1 -- --model $m --target sycl --target-string "$ST" --graph
  compact3 a_${m}_final_graph "${FENV[@]}" -- --model $m --target sycl --target-string "$ST" --graph
done
for m in resnet18 resnet50; do
  rn3 a_${m}_base X=1 -- --model $m --target sycl --target-string "$ST" --pipeline default
  rn3 a_${m}_noF2 TVM_SYCL_FMINMAX=0 -- --model $m --target sycl --target-string "$ST" --pipeline default
  rn3 a_${m}_n1 TVM_SYCL_NATIVE_IDS=1 -- --model $m --target sycl --target-string "$ST" --pipeline default
  rn3 a_${m}_final "${FENV[@]}" -- --model $m --target sycl --target-string "$ST" --pipeline default
done
for op in conv2d matmul; do
  e11 a_${op}_base $op sycl "$ST" X=1
  e11 a_${op}_n1 $op sycl "$ST" TVM_SYCL_NATIVE_IDS=1
  e11 a_${op}_final $op sycl "$ST" "${FENV[@]}"
  e11 a_${op}_cuda $op cuda "$CT" X=1
done

fi
# ---- Part B: final configuration, both targets in the same window
if [[ $PARTS == *B* ]]; then
if [ -n "$DB_SYCL" ]; then
  rn3 rn18_sycl_final_eager "${FENV[@]}" -- --target sycl --target-string "$ST" --apply-db-dir $DB_SYCL
  rn3 rn18_sycl_final_graph "${FENV[@]}" -- --target sycl --target-string "$ST" --apply-db-dir $DB_SYCL --graph
fi
rn3 rn18_cuda_iso_eager X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso
rn3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph
for b in 1 8 16; do
  for m in mlp cnn transformer vit; do
    compact3 sweep_${m}_b${b}_sycl "${FENV[@]}" -- --model $m --batch $b --target sycl --target-string "$ST"
    compact3 sweep_${m}_b${b}_cuda X=1 -- --model $m --batch $b --target cuda --target-string "$CT"
    compact3 sweep_${m}_b${b}_sycl_graph "${FENV[@]}" -- --model $m --batch $b --target sycl --target-string "$ST" --graph
    compact3 sweep_${m}_b${b}_cuda_graph X=1 -- --model $m --batch $b --target cuda --target-string "$CT" --graph
  done
  for m in resnet18 resnet34 resnet50; do
    rn3 sweep_${m}_b${b}_sycl "${FENV[@]}" -- --model $m --batch $b --target sycl --target-string "$ST" --pipeline default
    rn3 sweep_${m}_b${b}_cuda X=1 -- --model $m --batch $b --target cuda --target-string "$CT" --pipeline default
  done
done
for m in transformer vit; do for t in sycl cuda; do for g in "" "--graph"; do
  name=${m}_${t}${g:+_graph}; ts=$ST; envs=("${FENV[@]}"); [ $t = cuda ] && { ts=$CT; envs=(X=1); }
  [ -s $W/e1/$name.json ] && continue
  ( source $E/env_4090.sh r2
    env "${envs[@]}" nsys profile --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -o $W/e1/$name --force-overwrite true \
      $PY $E/compact_probe_r2.py --model $m --target $t --target-string "$ts" $g --warmup 10 --measure 30 --nsys-iters 20 \
      --out-json $W/e1/$name.json 2>&1 | tail -1 | sed "s/^/e1 $name /" >> $LOG
    nsys stats --report cuda_gpu_trace,cuda_api_trace --format csv -o $W/e1/$name $W/e1/$name.nsys-rep > /dev/null 2>&1 )
done; done; done
fi
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw --format=csv >> $LOG
echo "=== window end $(date -Iseconds)" >> $LOG
