#!/usr/bin/env python3
"""Pin down which primitive is miscompiled on the DCU ROCm path.

Established: a two-value cross-thread allreduce returns a bit-identical, correct
value on all 64 lanes, but 1.0/sqrt(that value) is correct on 48 lanes and Inf
on the 16 lanes with lane%16 in {8,9,10,11}. The reduction is therefore
innocent; the fault is in the scalar arithmetic applied afterwards.

The real layer_norm TIR asks for T.rsqrt, yet the emitted ISA is v_sqrt_f32
followed by the full IEEE division expansion (v_div_scale_f32, v_rcp_f32,
v_div_fmas_f32, v_div_fixup_f32). v_div_fmas_f32 reads VCC as an implicit
per-lane operand and scales its result by 2^32 where VCC is set, which is a
per-lane mechanism that matches the observed per-lane failure.

This applies a matrix of primitives to the same reduced, lane-uniform value and
reports which ones diverge across lanes, separating fdiv from sqrt and rsqrt.
"""
import argparse
import os

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import te, tir

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64")

ROWS, COLS = 16, 64

# name -> (fn of the reduced value, numpy reference)
OPS = [
    ("sqrt",        lambda v: te.sqrt(v),                    lambda v: np.sqrt(v)),
    ("rsqrt",       lambda v: te.rsqrt(v),                   lambda v: 1.0 / np.sqrt(v)),
    ("one_div_v",   lambda v: tir.const(1.0, "float32") / v, lambda v: 1.0 / v),
    ("one_div_sqrt", lambda v: tir.const(1.0, "float32") / te.sqrt(v),
     lambda v: 1.0 / np.sqrt(v)),
    ("v_div_three", lambda v: v / tir.const(3.0, "float32"), lambda v: v / 3.0),
    ("exp",         lambda v: te.exp(v),                     lambda v: np.exp(v)),
    ("mul_two",     lambda v: v * tir.const(2.0, "float32"), lambda v: v * 2.0),
]


def build_mod():
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    k = te.reduce_axis((0, COLS), name="k")

    def fcombine(x, y):
        return x[0] + y[0], x[1] + y[1]

    def fidentity(t0, t1):
        return tir.const(0, t0), tir.const(0, t1)

    pair = te.comm_reducer(fcombine, fidentity, name="pair")
    S, Q = te.compute(
        (ROWS,),
        lambda i: pair((A[i, k], A[i, k] * A[i, k]), axis=k),
        name="red",
    )

    # var+eps, exactly as layer_norm forms it, so the operand is the same value
    # the failing kernel feeds to its reciprocal-sqrt.
    def val(i):
        mean = S[i] / COLS
        return Q[i] / COLS - mean * mean + tir.const(1e-5, "float32")

    def make(fn):
        # te.compute rejects lambdas with default arguments, so fn is captured
        # through a factory instead.
        def body(i, j):
            return fn(val(i))
        return body

    outs = [te.compute((ROWS, COLS), make(fn), name=nm) for nm, fn, _ in OPS]
    return tvm.IRModule({"main": te.create_prim_func([A] + outs)}), OPS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa5")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    for kind in ("rocm", "sycl"):
        target = tvm.target.Target(ROCM if kind == "rocm" else "sycl")
        dev = tvm.device(kind, 0)
        mod, ops = build_mod()
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
            )(mod)
        lib = tvm.compile(mod, target=target)

        a = np.zeros((ROWS, COLS), "float32")  # so var+eps is exactly 1e-5
        outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
                for _ in ops]
        lib["main"](tvm.nd.array(a, dev), *outs)
        dev.sync()

        v = 1e-5
        print(f"=== {kind}: operand is a lane-uniform {v} from the allreduce")
        print(f"{'op':<14}{'uniform':>9}{'bad':>5}{'expected':>14}"
              f"{'good lanes':>14}{'bad lanes':>14}   lane%16")
        for (nm, _, ref), o in zip(ops, outs):
            got = o.numpy()[0]
            exp = float(ref(v))
            bad = np.flatnonzero(~np.isclose(got, exp, rtol=1e-4, atol=1e-6))
            good = [i for i in range(COLS) if i not in set(bad.tolist())]
            gv = float(got[good[0]]) if good else float("nan")
            bv = float(got[bad[0]]) if len(bad) else float("nan")
            mod16 = sorted(set(int(i) % 16 for i in bad)) if len(bad) else []
            print(f"{nm:<14}{str(bool(np.allclose(got, got[0], equal_nan=True))):>9}"
                  f"{len(bad):>5}{exp:>14.6g}{gv:>14.6g}{bv:>14.6g}   {mod16}")

        if kind == "rocm":
            for m in [lib.mod] + list(lib.mod.imported_modules):
                for c in [m] + list(m.imported_modules):
                    try:
                        src = c.get_source("asm")
                    except Exception:  # noqa: BLE001
                        continue
                    if src and ("ds_bpermute" in src or "s_barrier" in src):
                        with open(f"{args.dump_dir}/matrix.s", "w") as fh:
                            fh.write(src)
                        for ins in ("v_rsq_f32", "v_sqrt_f32", "v_rcp_f32",
                                    "v_div_scale_f32", "v_div_fmas_f32",
                                    "v_div_fixup_f32"):
                            print(f"  ISA {ins:<18}{src.count(ins)}")
                        break
        print()


if __name__ == "__main__":
    main()
