"""RecycleOnErrorRunner watchdog: a kernel that never terminates, then a valid candidate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet_relax_probe_r2 as probe  # noqa: E402

import tvm  # noqa: E402
from tvm import meta_schedule as ms  # noqa: E402
from tvm.script import tir as T  # noqa: E402

from test_recycle_runner import add_one  # noqa: E402


@T.prim_func
def spin(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for tx in T.thread_binding(1, thread="threadIdx.x"):
            with T.block("B"):
                T.reads(A[0], B[0])
                T.writes(B[0])
                # A is filled with finite values, so the loop never ends.
                while A[0] < T.float32(3.0e38):
                    B[0] = B[0] + T.float32(1)


def build(func, target):
    out = ms.builder.LocalBuilder().build([ms.builder.BuilderInput(tvm.IRModule({"main": func}), target)])[0]
    assert out.error_msg is None, out.error_msg
    return out.artifact_path


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    target = tvm.target.Target(sys.argv[2] if len(sys.argv) > 2 else kind)
    dev_type = "oneapi" if kind == "sycl" else kind
    hang = ms.runner.RunnerInput(build(spin, target), dev_type, ms.arg_info.ArgInfo.from_prim_func(spin))
    good = ms.runner.RunnerInput(build(add_one, target), dev_type, ms.arg_info.ArgInfo.from_prim_func(add_one))
    runner = probe.RecycleOnErrorRunner(watchdog_sec=20)
    results = [f.result() for f in runner.run([hang, good])]
    for i, r in enumerate(results):
        print(i, "error: " + r.error_msg.strip().splitlines()[-1][:120] if r.error_msg else f"ok {[float(c) for c in r.run_secs]}")
    assert results[0].error_msg is not None and results[1].error_msg is None
    print("WATCHDOG_TEST_PASS", kind)


if __name__ == "__main__":
    main()
