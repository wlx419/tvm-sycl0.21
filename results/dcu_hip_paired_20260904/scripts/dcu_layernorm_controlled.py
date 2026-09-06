#!/usr/bin/env python3
"""Controlled-input probe for the failing ROCm layer_norm kernel.

The generated LLVM IR is correct: mean and 1/sqrt(var+eps) are loaded by every
lane from the same shared-memory scalar after a barrier, so they cannot differ
per lane, and the x / weight / bias / output addressing is a plain linear GEP on
blockIdx.x*64 + threadIdx.x. Yet 48 lanes per row are exact and 16 are Inf.

So the inputs are overwritten at the kernel boundary to isolate which per-lane
term carries the fault. With the generated code

    %94  = (x - mean) * (1/sqrt(var+eps))
    out  = fmuladd(%94, weight[tid], bias[tid])

the cases are:

  g0_barange : g=0, b=arange, x=0  -> out must equal bias[tid] exactly, unless
               %94 is nonfinite, in which case Inf*0 yields NaN. This separates
               "the shared mean/rsqrt term went bad" from "the weight load went
               bad".
  g1_b0_x0   : g=1, b=0, x=0       -> var=0, so out must be 0 everywhere.
  g_arange   : g=arange+1, b=0     -> lets the weight actually used by each lane
               be recovered from a finite output.
"""
import argparse
import sys

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64")

BAD = [8, 9, 10, 11, 24, 25, 26, 27, 40, 41, 42, 43, 56, 57, 58, 59]


def make_cases(ncol):
    ar = np.arange(ncol, dtype="float32")
    return {
        "g0_barange": dict(x=0.0, g=np.zeros(ncol, "float32"), b=ar.copy()),
        "g1_b0_x0": dict(x=0.0, g=np.ones(ncol, "float32"), b=np.zeros(ncol, "float32")),
        "g_arange": dict(x=None, g=(ar + 1.0).copy(), b=np.zeros(ncol, "float32")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/workspace/supplement/experiments")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.exp_dir)
    import relax_model_probe as P  # noqa: E402

    target = tvm.target.Target(ROCM)
    dev = tvm.device("rocm", args.device_id)

    case_seed = P.SEED + args.batch * 17 + sum(ord(c) for c in (args.model + "sycl"))
    rng = np.random.default_rng(case_seed)
    model, input_shape, _ = P.make_case(args.model, args.batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    named_np = P.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")

    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir="/root/rocm_baseline/ctrl_work")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    def run(case):
        state = {"done": False}

        def instrument(func, func_symbol, before_run, ret_value, *call_args):
            if func_symbol != "layer_norm" or state["done"]:
                return None
            t = [a for a in call_args if isinstance(a, tvm.nd.NDArray)]
            if len(t) < 4:
                return None
            xb, gb, bb, ob = t[0], t[1], t[2], t[-1]
            if before_run:
                dev.sync()
                xs = [int(d) for d in xb.shape]
                if case["x"] is not None:
                    xb.copyfrom(np.full(xs, case["x"], dtype="float32"))
                gb.copyfrom(case["g"])
                bb.copyfrom(case["b"])
                dev.sync()
                state["x"] = xb.numpy()
            else:
                dev.sync()
                state["y"] = ob.numpy()
                state["done"] = True
            return None

        vm.set_instrument(instrument)
        vm["forward"](tvm.nd.array(x_np, dev),
                      *[tvm.nd.array(a, dev) for a in named_np.values()])
        return state

    ncol = 64
    for name, case in make_cases(ncol).items():
        st = run(case)
        y = st.get("y")
        if y is None:
            print(f"--- {name}: layer_norm not captured\n")
            continue
        flat = y.reshape(-1, y.shape[-1])
        r0 = flat[0]
        nan = int(np.isnan(flat).sum())
        inf = int(np.isinf(flat).sum())
        print(f"--- {name}: nan={nan} inf={inf} of {flat.size}")
        print(f"    row0 bad cols: {np.flatnonzero(~np.isfinite(r0)).tolist()}")
        print(f"    row0[{BAD[:4]}] = {[float(r0[i]) for i in BAD[:4]]}")
        good = [i for i in range(ncol) if i not in BAD]
        print(f"    row0[{good[:4]}] = {[float(r0[i]) for i in good[:4]]}")

        if name == "g0_barange":
            exp = case["b"]
            match = np.allclose(r0, exp, atol=1e-6, equal_nan=False)
            print(f"    equals bias exactly: {match}")
            if not match:
                print("    -> the shared mean/rsqrt term is nonfinite on the bad "
                      "lanes (Inf*0 = NaN), so the fault is upstream of the "
                      "weight load")
            else:
                print("    -> bias path is fine on every lane; the earlier Inf "
                      "came through the weight multiply")
        if name == "g1_b0_x0":
            print(f"    all zero as required: {np.allclose(r0, 0.0)}")
        print()


if __name__ == "__main__":
    main()
