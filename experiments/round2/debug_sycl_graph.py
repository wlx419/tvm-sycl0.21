#!/usr/bin/env python3
"""Compare eager vs SYCL-graph outputs call by call for ResNet-18 B1 (published SYCL DB)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments"))
import numpy as np, torch, tvm
import resnet_relax_probe as base
from tvm import relax, dlight as dl
from tvm.relax import transform as rt

DB = "/mnt/data/wlx/workspace/tvm-sycl-exp/codex_extended/paired_resnet18_metaschedule_20260813/formal1024/tune/sycl"
target = tvm.target.Target("sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32")
dev = tvm.device("oneapi", 0)
torch.manual_seed(0)
model = base.fuse_resnet_batch_norm(base.make_model("resnet18", 1000))
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    ref = model(x).numpy()
def build(graph):
    mod = base.from_exported_program(torch.export.export(model, (x,)), keep_params_as_input=False, unwrap_unit_return_tuple=True)
    with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": graph}):
        with target:
            mod = tvm.transform.Sequential([rt.DecomposeOpsForInference(), rt.CanonicalizeBindings(), relax.get_pipeline("zero")])(mod)
            mod = rt.MetaScheduleApplyDatabase(DB)(mod)
            mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
        ex = tvm.compile(mod, target=target)
    return ex, mod
ex_e, _ = build(False)
vm_e = relax.VirtualMachine(ex_e, dev)
xd0 = tvm.nd.array(x.numpy(), dev)
oe = base.to_numpy(vm_e["main"](xd0)); dev.sync()
print(f"eager build: max abs err vs torch = {np.abs(oe - ref).max():.3e}")
ex_g, mod_g = build(True)
print("graph mode env:", os.environ.get("TVM_SYCL_GRAPH_MODE", "record"))
print("VM text has sycl_graph:", "sycl_graph" in ex_g.as_text(), "cuda_graph:", "cuda_graph" in ex_g.as_text())
txt = mod_g.script()
print("graph builtins in final IR:", txt.count("sycl_graph"), "cuda_graph refs:", txt.count("cuda_graph"))
vm = relax.VirtualMachine(ex_g, dev)
xd = tvm.nd.array(x.numpy(), dev)
for i in range(4):
    out = vm["main"](xd)
    dev.sync()
    o = base.to_numpy(out)
    print(f"graph call {i}: max abs err vs torch = {np.abs(o - ref).max():.3e}")
