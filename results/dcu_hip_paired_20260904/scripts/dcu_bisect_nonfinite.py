#!/usr/bin/env python3
"""Find the first Relax VM call that goes nonfinite on the TVM ROCm backend.

Rather than synthesizing primitives, this reuses relax_model_probe.py's own
model construction, parameter materialization, input seeding, and compilation
pipeline, so the module under test is exactly the one that failed the sweep. A
VM instrument hook then inspects every call's output and reports the first call
whose result contains NaN/Inf while all of its inputs were finite.
"""
import argparse
import sys

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn

ROCM_TARGET = (
    "rocm -keys=rocm,gpu -max_num_threads=256 -max_shared_memory_per_block=65536 "
    "-max_threads_per_block=1024 -mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc "
    "-thread_warp_size=64"
)


def summarize(value):
    """Describe an NDArray-ish VM value's finiteness, or None if not a tensor."""
    if not isinstance(value, tvm.nd.NDArray):
        return None
    try:
        arr = value.numpy()
    except Exception:  # noqa: BLE001
        return None
    if arr.dtype.kind != "f":
        return None
    finite = np.isfinite(arr)
    return {
        "shape": tuple(arr.shape),
        "n": int(arr.size),
        "n_nan": int(np.isnan(arr).sum()),
        "n_inf": int(np.isinf(arr).sum()),
        "all_finite": bool(finite.all()),
        "absmax": float(np.abs(arr[finite]).max()) if finite.any() else float("nan"),
    }


def collect(value, out, depth=0):
    """Flatten VM values into tensor summaries.

    Only NDArrays and explicit containers are traversed. A permissive
    "try to iterate" fallback recurses forever on values like str, whose
    elements are themselves iterable of the same type.
    """
    if depth > 4:
        return
    if isinstance(value, tvm.nd.NDArray):
        s = summarize(value)
        if s:
            out.append(s)
        return
    if isinstance(value, (list, tuple, tvm.ir.Array)):
        for v in value:
            collect(v, out, depth + 1)


DEV = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/workspace/supplement/experiments")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--target", default="rocm", choices=["rocm", "sycl"])
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed-target-alias", default="sycl")
    args = ap.parse_args()

    sys.path.insert(0, args.exp_dir)
    import relax_model_probe as P  # noqa: E402

    # Reproduce the sweep's inputs exactly.
    case_seed = P.SEED + args.batch * 17 + sum(
        ord(c) for c in (args.model + args.seed_target_alias)
    )
    rng = np.random.default_rng(case_seed)

    target = tvm.target.Target(ROCM_TARGET if args.target == "rocm" else "sycl")
    dev = tvm.device(args.target, args.device_id)
    print(f"target={args.target} exist={dev.exist} seed={case_seed}")

    model, input_shape, ref_fn = P.make_case(args.model, args.batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    named_np = P.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")

    print(f"inputs: x{input_shape} finite={np.isfinite(x_np).all()} "
          f"params={len(named_np)} "
          f"all_finite={all(np.isfinite(v).all() for v in named_np.values())}")

    # Same lowering as the sweep: static_shape_tuning with 0 trials, then the
    # DLight GPU fallback rules.
    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir=f"/root/rocm_baseline/bisect_{args.target}")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    trace = []

    def instrument(func, func_symbol, before_run, ret_value, *call_args):
        # TIR kernels use destination-passing style: the output buffer is the
        # last argument and the return value is null. So the output must be read
        # from the arguments after the call, and a freshly allocated buffer is
        # expected to hold uninitialized garbage before it.
        if before_run or func_symbol.startswith("vm.builtin."):
            return None
        # Kernel launches are asynchronous. Without this sync the read below can
        # land before the kernel has written the whole output buffer, and the
        # unwritten tail reads back as garbage that looks like a numerical fault.
        dev.sync()
        tensors = []
        for a in call_args:
            collect(a, tensors)
        if not tensors:
            return None
        trace.append({
            "symbol": func_symbol,
            "inputs": tensors[:-1],
            "output": tensors[-1],
        })
        return None

    vm.set_instrument(instrument)

    x_dev = tvm.nd.array(x_np, dev)
    tvm_params = [tvm.nd.array(arr, dev) for arr in named_np.values()]
    out = vm["forward"](x_dev, *tvm_params).numpy()

    print(f"\nfinal output: finite={np.isfinite(out).all()} "
          f"nan={int(np.isnan(out).sum())} inf={int(np.isinf(out).sum())}")

    print(f"\n{'#':>3} {'kernel':<34} {'in_finite':>10} {'out_finite':>11} "
          f"{'nan':>6} {'inf':>6} {'of':>7} {'absmax':>11}")
    first_bad = None
    for i, t in enumerate(trace):
        ins, o = t["inputs"], t["output"]
        in_ok = all(s["all_finite"] for s in ins) if ins else None
        mark = ""
        # The culprit is the first kernel handed finite inputs that still
        # produces a nonfinite output.
        if in_ok is not False and not o["all_finite"] and first_bad is None:
            first_bad = i
            mark = "  <== FIRST"
        print(f"{i:>3} {t['symbol'][:34]:<34} {str(in_ok):>10} "
              f"{str(o['all_finite']):>11} {o['n_nan']:>6} {o['n_inf']:>6} "
              f"{o['n']:>7} {o['absmax']:>11.4g}{mark}")

    if first_bad is None:
        bad_any = [t for t in trace if not t["output"]["all_finite"]]
        print("\nno kernel turned finite inputs into a nonfinite output"
              f" ({len(bad_any)} kernels had nonfinite output with nonfinite input)")
        if not bad_any:
            return
        first_bad = trace.index(bad_any[0])

    t = trace[first_bad]
    print(f"\n=== {t['symbol']} ===")
    for j, s in enumerate(t["inputs"]):
        print(f"  in[{j}]  shape={s['shape']} finite={s['all_finite']} "
              f"nan={s['n_nan']} inf={s['n_inf']} absmax={s['absmax']:.6g}")
    o = t["output"]
    print(f"  out    shape={o['shape']} finite={o['all_finite']} "
          f"nan={o['n_nan']} inf={o['n_inf']} of {o['n']}")

    # Show the offending TIR so the failing lowering is visible.
    name = t["symbol"]
    for gv, func in mod.functions_items():
        if gv.name_hint == name:
            print(f"\n=== TIR for {name} ===")
            print(str(func)[:4000])
            break


if __name__ == "__main__":
    main()
