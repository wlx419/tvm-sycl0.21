import tvm
from tvm import relax
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax.backend.sycl import pipeline as sp

@I.ir_module
class M:
    @T.prim_func(private=True)
    def add(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
        for i in T.thread_binding(16, thread="threadIdx.x"):
            with T.block("b"):
                vi = T.axis.spatial(16, i)
                B[vi] = A[vi] + T.float32(1)
    @R.function
    def main(x: R.Tensor((16,), "float32")):
        cls = M
        with R.dataflow():
            y = R.call_tir(cls.add, (x,), out_sinfo=R.Tensor((16,), "float32"))
            z = R.call_tir(cls.add, (y,), out_sinfo=R.Tensor((16,), "float32"))
            R.output(z)
        return z
target = tvm.target.Target("sycl")
with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": True}):
    with target:
        mod = tvm.transform.Sequential(sp.dataflow_lower_passes(target) + [relax.transform.StaticPlanBlockMemory(), relax.transform.RewriteCUDAGraph()])(M)
s = mod.script()
print("after RewriteCUDAGraph: cuda_graph", s.count("cuda_graph"))
import re
for line in s.splitlines():
    if "cuda_graph" in line: print("  ", line.strip()[:160]); break
mod2 = sp.redirect_graph_builtins(mod)
s2 = mod2.script()
print("after redirect: cuda_graph", s2.count("cuda_graph"), "sycl_graph", s2.count("sycl_graph"))
for line in s2.splitlines():
    if "graph" in line: print("   |", line.strip()[:200])
