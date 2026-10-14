#!/usr/bin/env bash
# Functional checks of the r2 improvements (correctness and compile time; GPU latency not used).
set -u
E=/mnt/data/wlx/workspace/tvm-sycl-exp/round2_20261012/repo/revision_round2/experiments
source $E/env_4090.sh r2
D=$R/funcchk; mkdir -p $D
DB=/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune
ST="sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32"
CT="cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32"
$PY - <<'PY' > $D/detect.txt 2>&1
import tvm
print("from_device(oneapi):", tvm.target.Target.from_device(tvm.device("oneapi", 0)))
print("sycl graph builtin:", tvm.get_global_func("vm.builtin.sycl_graph.run_or_capture", allow_missing=True) is not None)
PY
cat $D/detect.txt
run() { local n=$1; shift; uptime > $D/$n.load; "$@" > $D/$n.log 2>&1; tail -1 $D/$n.log; }
# compile time: one device module per target (default) vs. per-kernel modules (old behaviour)
run sycl_grouped_cold   env TVM_SYCL_JIT_CACHE=0 $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $DB/sycl --warmup 1 --measure 3 --out-json $D/sycl_grouped_cold.json
run sycl_perfunc_cold   env TVM_SYCL_JIT_CACHE=0 TVM_SPLIT_DEVICE_MODULE_PER_FUNC=1 $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $DB/sycl --warmup 1 --measure 3 --out-json $D/sycl_perfunc_cold.json
run cuda_grouped        $PY $E/resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $DB/cuda --warmup 1 --measure 3 --out-json $D/cuda_grouped.json
run cuda_perfunc        env TVM_SPLIT_DEVICE_MODULE_PER_FUNC=1 $PY $E/resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $DB/cuda --warmup 1 --measure 3 --out-json $D/cuda_perfunc.json
run sycl_grouped_warm   $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $DB/sycl --warmup 1 --measure 3 --out-json $D/sycl_grouped_warm.json
# graph capture correctness
run sycl_graph          $PY $E/resnet_relax_probe_r2.py --target sycl --target-string "$ST" --apply-db-dir $DB/sycl --graph --warmup 3 --measure 5 --out-json $D/sycl_graph.json
run cuda_graph          $PY $E/resnet_relax_probe_r2.py --target cuda --target-string "$CT" --apply-db-dir $DB/cuda --graph --warmup 3 --measure 5 --out-json $D/cuda_graph.json
for f in $D/*.json; do $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split('/')[-1], d.get('status'), 'compile_s=%.1f' % d.get('compile_s',-1), 'abs=%s' % d.get('max_abs_err'), (d.get('error') or '')[:300])" $f; done
echo FUNC_DONE
