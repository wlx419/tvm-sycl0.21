#!/usr/bin/env python3
"""Pin down the ROCm layer_norm failure pattern and whether it is schedule-bound.

Established so far: layer_norm is the first kernel that turns finite inputs into
a nonfinite output on the TVM ROCm backend, 256 of 1024 elements come back Inf,
the remaining elements are correct to float32 precision, an explicit device sync
does not change it, and forcing thread_warp_size=32 does not either.

This script reports which columns go Inf, and rebuilds the same model under
different DLight rule sets. If some rule set produces a finite layer_norm, the
fault is in how that reduction is scheduled and lowered rather than in ROCm's
math or in layer_norm's arithmetic.
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

RULE_SETS = {
    "default(Matmul,GEMV,Reduction,GeneralReduction,Fallback)": lambda: (
        dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
    ),
    "no-Reduction(GeneralReduction first)": lambda: (
        dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
    ),
    "Fallback-only": lambda: (dl.gpu.Fallback(),),
}


def build(target, dev, P, model_name, batch, alias, rules):
    case_seed = P.SEED + batch * 17 + sum(ord(c) for c in (model_name + alias))
    rng = np.random.default_rng(case_seed)
    model, input_shape, _ = P.make_case(model_name, batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    named_np = P.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")

    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir="/root/rocm_baseline/bisect_pat")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(*rules)(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    grabbed = {}

    def instrument(func, func_symbol, before_run, ret_value, *call_args):
        if before_run or func_symbol != "layer_norm" or "y" in grabbed:
            return None
        dev.sync()
        arrs = [a.numpy() for a in call_args if isinstance(a, tvm.nd.NDArray)]
        if len(arrs) >= 4:
            grabbed["x"], grabbed["y"] = arrs[0], arrs[-1]
        return None

    vm.set_instrument(instrument)
    out = vm["forward"](tvm.nd.array(x_np, dev),
                        *[tvm.nd.array(a, dev) for a in named_np.values()]).numpy()
    return grabbed, out, mod


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
    print(f"rocm exist={dev.exist} model={args.model} B{args.batch}\n")

    for label, mk in RULE_SETS.items():
        try:
            g, out, mod = build(target, dev, P, args.model, args.batch, "sycl", mk())
        except Exception as exc:  # noqa: BLE001
            print(f"--- {label}\n    build/run failed: {type(exc).__name__}: "
                  f"{' '.join(str(exc).split())[:160]}\n")
            continue

        y = g.get("y")
        print(f"--- {label}")
        print(f"    model output finite={np.isfinite(out).all()} "
              f"nan={int(np.isnan(out).sum())} inf={int(np.isinf(out).sum())}")
        if y is None:
            print("    layer_norm not captured (fused away?)\n")
            continue
        bad = ~np.isfinite(y)
        print(f"    layer_norm out finite={not bad.any()} "
              f"inf={int(np.isinf(y).sum())} nan={int(np.isnan(y).sum())} of {y.size}")
        if bad.any():
            flat = y.reshape(-1, y.shape[-1])
            badf = bad.reshape(-1, y.shape[-1])
            cols = np.flatnonzero(badf[0])
            print(f"    row0 bad columns ({len(cols)} of {flat.shape[1]}): {cols.tolist()}")
            same = all(np.array_equal(np.flatnonzero(badf[r]), cols)
                       for r in range(badf.shape[0]))
            print(f"    same columns in every row: {same}")
        print()

    print("=== lowered layer_norm TIR under the default rules ===")
    _, _, mod = build(target, dev, P, args.model, args.batch, "sycl",
                      RULE_SETS["default(Matmul,GEMV,Reduction,GeneralReduction,Fallback)"]())
    for gv, func in mod.functions_items():
        if gv.name_hint == "layer_norm":
            text = str(func)
            # Show the reduction body and the barrier structure.
            for line in text.splitlines():
                if any(t in line for t in ("thread_binding", "allreduce", "sync",
                                           "shared", "rsqrt", "T.launch_thread",
                                           "tvm_storage_sync")):
                    print("   ", line.strip()[:150])
            break


if __name__ == "__main__":
    main()
