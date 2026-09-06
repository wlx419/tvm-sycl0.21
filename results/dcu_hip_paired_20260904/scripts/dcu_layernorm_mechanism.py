#!/usr/bin/env python3
"""Characterize the ROCm layer_norm failure found by the bisect.

layer_norm is the first kernel that turns finite inputs into a nonfinite output
on the TVM ROCm backend. Its DLight schedule reduces 64 elements across
threadIdx.x, and gfx906 reports thread_warp_size=64, so the reduction lands on
TVM's warp-shuffle allreduce path. This script:

  1. captures the actual layer_norm input/output from the real model, reports
     which rows go nonfinite, and compares against a NumPy reference; and
  2. re-runs with thread_warp_size forced to 32, which pushes the same schedule
     onto the shared-memory reduction path instead of a 64-wide warp shuffle.

If (2) is finite while the default is not, the warp-shuffle allreduce lowering
is implicated rather than layer_norm's arithmetic.
"""
import argparse
import sys

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn

BASE = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size={ws}")


def build_and_capture(target, dev, P, model_name, batch, seed_alias):
    case_seed = P.SEED + batch * 17 + sum(ord(c) for c in (model_name + seed_alias))
    rng = np.random.default_rng(case_seed)
    model, input_shape, _ = P.make_case(model_name, batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    named_np = P.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")

    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir="/root/rocm_baseline/bisect_mech")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    grabbed = {}

    def instrument(func, func_symbol, before_run, ret_value, *call_args):
        if before_run or func_symbol != "layer_norm" or "args" in grabbed:
            return None
        # Kernel launches are asynchronous; sync before reading the output or
        # the unwritten tail reads back as garbage.
        dev.sync()
        arrs = [a.numpy() for a in call_args if isinstance(a, tvm.nd.NDArray)]
        grabbed["args"] = arrs
        return None

    vm.set_instrument(instrument)
    x_dev = tvm.nd.array(x_np, dev)
    tvm_params = [tvm.nd.array(a, dev) for a in named_np.values()]
    out = vm["forward"](x_dev, *tvm_params).numpy()
    return grabbed.get("args"), out, mod


def report(tag, arrs, out):
    print(f"\n===== {tag} =====")
    print(f"model output: finite={np.isfinite(out).all()} "
          f"nan={int(np.isnan(out).sum())} inf={int(np.isinf(out).sum())}")
    if not arrs or len(arrs) < 4:
        print("  layer_norm not captured")
        return
    x, g, b, y = arrs[0], arrs[1], arrs[2], arrs[-1]
    print(f"  layer_norm in  x{x.shape} finite={np.isfinite(x).all()} absmax={np.abs(x).max():.6g}")
    print(f"  layer_norm out y{y.shape} finite={np.isfinite(y).all()} "
          f"nan={int(np.isnan(y).sum())} inf={int(np.isinf(y).sum())} of {y.size}")

    # NumPy reference over the last axis, matching TVM's layer_norm.
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    ref = (x - mu) / np.sqrt(var + 1e-5) * g + b

    rows_bad = ~np.isfinite(y).all(-1)
    n_rows = int(rows_bad.size)
    print(f"  rows nonfinite: {int(rows_bad.sum())}/{n_rows}")
    if rows_bad.any():
        idx = np.argwhere(rows_bad)
        print(f"  bad row indices (first 8): {[tuple(int(v) for v in i) for i in idx[:8]]}")
        print("  per-row stats from NumPy (these are all well-conditioned if the")
        print("  arithmetic is fine, which would point at the reduction instead):")
        for i in idx[:4]:
            t = tuple(int(v) for v in i)
            print(f"    row {t}: mean={float(mu[t][0]):+.6g} var={float(var[t][0]):.6g} "
                  f"var+eps={float(var[t][0]) + 1e-5:.6g} "
                  f"rsqrt={1.0 / np.sqrt(float(var[t][0]) + 1e-5):.6g}")
    good = np.isfinite(y)
    if good.any():
        print(f"  max abs err on finite elements: {np.abs(y[good] - ref[good]).max():.6g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/workspace/supplement/experiments")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seed-target-alias", default="sycl")
    args = ap.parse_args()

    sys.path.insert(0, args.exp_dir)
    import relax_model_probe as P  # noqa: E402

    dev = tvm.device("rocm", args.device_id)
    print(f"rocm exist={dev.exist} model={args.model} B{args.batch}")

    for ws in (64, 32):
        target = tvm.target.Target(BASE.format(ws=ws))
        try:
            arrs, out, mod = build_and_capture(
                target, dev, P, args.model, args.batch, args.seed_target_alias
            )
            report(f"thread_warp_size={ws}", arrs, out)
        except Exception as exc:  # noqa: BLE001
            print(f"\n===== thread_warp_size={ws} =====")
            print(f"  failed: {type(exc).__name__}: {' '.join(str(exc).split())[:200]}")


if __name__ == "__main__":
    main()
