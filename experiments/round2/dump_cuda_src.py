"""Dump the CUDA source/PTX of the published ResNet-18 CUDA build and count async-copy instructions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments"))
import torch
import tvm
import resnet_relax_probe as base
from tvm import dlight as dl
from tvm import relax
from tvm.relax import transform as rt

DB = sys.argv[1]
out = sys.argv[2]
target = tvm.target.Target("cuda -keys=cuda,gpu -arch=sm_89 -max_num_threads=1024 "
                           "-max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32")
model = base.fuse_resnet_batch_norm(base.make_model("resnet18", 1000))
x = torch.randn(1, 3, 224, 224)
mod = base.from_exported_program(torch.export.export(model, (x,)), keep_params_as_input=False,
                                 unwrap_unit_return_tuple=True)
with target:
    mod = tvm.transform.Sequential([rt.DecomposeOpsForInference(), rt.CanonicalizeBindings(),
                                    relax.get_pipeline("zero")])(mod)
    mod = rt.MetaScheduleApplyDatabase(DB)(mod)
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                  dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
ex = tvm.compile(mod, target=target)


def walk(m, seen):
    if id(m) in seen:
        return []
    seen.add(id(m))
    found = [m] if m.type_key == "cuda" else []
    for sub in m.imported_modules:
        found += walk(sub, seen)
    return found


mods = walk(ex.mod, set())
src = "\n".join(m.get_source("cu") for m in mods)
ptx = "\n".join(m.get_source("ptx") for m in mods)
open(out, "w").write(src)
open(out + ".ptx", "w").write(ptx)
print("cuda modules:", len(mods), "cu chars:", len(src), "ptx chars:", len(ptx))
for k in ["cp.async.commit_group", "cp.async.wait_group", "cp.async.ca.shared", "cp.async.cg.shared", "__syncthreads"]:
    print(f"cu  {k}: {src.count(k)}")
for k in ["cp.async.commit_group", "cp.async.wait_group", "cp.async.ca", "cp.async.cg", "bar.sync"]:
    print(f"ptx {k}: {ptx.count(k)}")
