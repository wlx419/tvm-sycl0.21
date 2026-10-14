"""Exercise RecycleOnErrorRunner's failure path: a missing artifact, then two valid candidates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet_relax_probe_r2 as probe  # noqa: E402

import tvm  # noqa: E402
from tvm import meta_schedule as ms  # noqa: E402
from tvm.script import tir as T  # noqa: E402


@T.prim_func
def add_one(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for bx in T.thread_binding(8, thread="blockIdx.x"):
        for tx in T.thread_binding(128, thread="threadIdx.x"):
            with T.block("B"):
                vi = T.axis.spatial(1024, bx * 128 + tx)
                B[vi] = A[vi] + T.float32(1)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    target = tvm.target.Target(sys.argv[2] if len(sys.argv) > 2 else kind)
    mod = tvm.IRModule({"main": add_one})
    built = ms.builder.LocalBuilder().build([ms.builder.BuilderInput(mod, target)])[0]
    assert built.error_msg is None, built.error_msg
    args = ms.arg_info.ArgInfo.from_prim_func(add_one)
    dev_type = "oneapi" if kind == "sycl" else kind
    good = ms.runner.RunnerInput(built.artifact_path, dev_type, args)
    bad = ms.runner.RunnerInput("/nonexistent/artifact.so", dev_type, args)
    runner = probe.RecycleOnErrorRunner()
    results = [f.result() for f in runner.run([bad, good, good])]
    for i, r in enumerate(results):
        print(i, "error" if r.error_msg else f"ok {[float(c) for c in r.run_secs]}")
    assert results[0].error_msg is not None
    assert results[1].error_msg is None and results[2].error_msg is None
    assert runner.recycled == 1, runner.recycled
    print("RECYCLE_TEST_PASS", kind, "recycled", runner.recycled)


if __name__ == "__main__":
    main()
