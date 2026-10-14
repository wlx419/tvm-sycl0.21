#!/usr/bin/env python3
"""Where does SYCL compile time go? Times each stage of the ResNet-18 B1 apply-only compile and
counts the SYCL JIT invocations (new files in the JIT folder) during tvm.compile."""
import glob, json, os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments"))
import resnet_relax_probe as base
import torch, tvm
from tvm import relax, dlight as dl
from tvm.relax import transform as rt

target_kind = sys.argv[1]; db = sys.argv[2]; ts = sys.argv[3]
target = tvm.target.Target(ts)
t = {}
t0 = time.perf_counter()
model = base.fuse_resnet_batch_norm(base.make_model("resnet18", 1000))
x = torch.randn(1, 3, 224, 224)
mod = base.from_exported_program(torch.export.export(model, (x,)), keep_params_as_input=False, unwrap_unit_return_tuple=True)
t["import"] = time.perf_counter() - t0
t0 = time.perf_counter()
with target:
    mod = tvm.transform.Sequential([rt.DecomposeOpsForInference(), rt.CanonicalizeBindings(), relax.get_pipeline("zero")])(mod)
t["zero_pipeline"] = time.perf_counter() - t0
t0 = time.perf_counter()
with target:
    mod = rt.MetaScheduleApplyDatabase(db)(mod)
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
t["apply_db_dlight"] = time.perf_counter() - t0
before = set(glob.glob("/tmp/tvm_sycl/*.so"))
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable()
ex = tvm.compile(mod, target=target)
pr.disable(); t["tvm_compile"] = time.perf_counter() - t0
t["new_sycl_libs"] = len(set(glob.glob("/tmp/tvm_sycl/*.so")) - before)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(json.dumps(t, indent=1))
print(s.getvalue()[:6000])
