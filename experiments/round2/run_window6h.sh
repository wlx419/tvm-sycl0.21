#!/usr/bin/env bash
# Window 6h on RTX 4090 device 1:
# R2 (executable command graphs re-finalized every 32 replays; DPC++ 646ab08's replay cost grows
# with the number of earlier replays).
#   long : 3,000 consecutive inferences with command graphs, R2 off / on, CUDA graphs for reference
#   graph sweep : the command-graph columns of the batch sweep on both targets with the final code
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
R=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012
W=$R/window6h; mkdir -p $W
LOG=$W/window.log
trap 'exit 130' INT TERM HUP
echo "=== window start $(date -Iseconds)" >> $LOG
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c e5e525e1)
[ "$APPS" = "0" ] || { echo "GPU1 still busy ($APPS); aborting" >> $LOG; exit 3; }
md5sum /mnt/data/wlx/workspace/apache-tvm-src-sycl-v0.21.0-20260809-r2/build-4090-r2/libtvm*.so >> $LOG
ST="sycl -keys=sycl,gpu -arch=sm_89 -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
PUB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
W3=$R/window3
compact() {  # compact <name> <runs> <env...> -- <compact args>
  local name=$1 runs=$2; shift 2
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in $(seq 1 $runs); do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/compact_probe_r2.py "$@" --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
rn3() {  # rn3 <name> <env...> -- <probe args>
  local name=$1; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  ( source $E/env_4090.sh r2; for r in 1 2 3; do
      [ -s $W/${name}_run$r.json ] && continue
      env "${envs[@]}" $PY $E/resnet_relax_probe_r2.py "$@" --warmup 10 --measure 50 \
        --out-json $W/${name}_run$r.json 2>&1 | tail -1 | sed "s/^/$name /" >> $LOG
    done )
}
# ---- long runs
for m in transformer vit; do
  compact long_${m}_sycl_r2off 2 TVM_SYCL_GRAPH_REFINALIZE=0 -- --model $m --target sycl --target-string "$ST" --graph --warmup 10 --measure 3000
  compact long_${m}_sycl_r2on 2 X=1 -- --model $m --target sycl --target-string "$ST" --graph --warmup 10 --measure 3000
  compact long_${m}_cuda 2 X=1 -- --model $m --target cuda --target-string "$CT" --graph --warmup 10 --measure 3000
done
# ---- graph columns of the sweep, and the graph rows of Tables identical/ablation/retune
for b in 1 8 16; do for m in mlp cnn transformer vit; do
  compact sweep_${m}_b${b}_sycl_graph 3 X=1 -- --model $m --batch $b --target sycl --target-string "$ST" --graph --warmup 10 --measure 50
  compact sweep_${m}_b${b}_cuda_graph 3 X=1 -- --model $m --batch $b --target cuda --target-string "$CT" --graph --warmup 10 --measure 50
done; done
rn3 rn18_sycl_iso_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $W3/tune_sycl_pipe_iso --graph
rn3 rn18_cuda_iso_graph X=1 -- --target cuda --target-string "$CT" --apply-db-dir $W3/tune_cuda_iso --graph
rn3 rn18_sycldb_sycl_final_graph X=1 -- --target sycl --target-string "$ST" --apply-db-dir $PUB/sycl --graph
echo "=== window end $(date -Iseconds)" >> $LOG
