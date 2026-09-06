#!/usr/bin/env python3
"""Does the DCU sqrt fault depend on where the operand came from?

Inside one fused reduction kernel, sqrt is applied to two operands that are
numerically identical (the input is all zeros, so both are exactly 1e-5) but
differ in provenance:

  from_reduction : var+eps built from the cross-thread allreduce, lane-uniform
  from_lane      : A[i,j]+1e-5, an ordinary per-lane global load

The operand itself is exported too, to show it is correct in every lane. If only
the reduction-derived operand fails, the trigger is the cross-lane dataflow
feeding the transcendental unit rather than sqrt or the operand value.
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
EPS = tir.const(1e-5, "float32")


def build():
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    k = te.reduce_axis((0, COLS), name="k")

    def fcombine(x, y):
        return x[0] + y[0], x[1] + y[1]

    def fidentity(t0, t1):
        return tir.const(0, t0), tir.const(0, t1)

    pair = te.comm_reducer(fcombine, fidentity, name="pair")
    S, Q = te.compute((ROWS,),
                      lambda i: pair((A[i, k], A[i, k] * A[i, k]), axis=k),
                      name="red")

    def red_val(i):
        mean = S[i] / COLS
        return Q[i] / COLS - mean * mean + EPS

    outs = [
        te.compute((ROWS, COLS), lambda i, j: te.sqrt(red_val(i)),
                   name="sqrt_from_reduction"),
        te.compute((ROWS, COLS), lambda i, j: te.sqrt(A[i, j] + EPS),
                   name="sqrt_from_lane"),
        te.compute((ROWS, COLS), lambda i, j: red_val(i),
                   name="operand_from_reduction"),
        te.compute((ROWS, COLS), lambda i, j: A[i, j] + EPS,
                   name="operand_from_lane"),
    ]
    return tvm.IRModule({"main": te.create_prim_func([A] + outs)}), outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa8")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    names = ["sqrt_from_reduction", "sqrt_from_lane",
             "operand_from_reduction", "operand_from_lane"]
    expect = [float(np.sqrt(1e-5)), float(np.sqrt(1e-5)), 1e-5, 1e-5]

    for kind in ("rocm", "sycl"):
        target = tvm.target.Target(ROCM if kind == "rocm" else "sycl")
        dev = tvm.device(kind, 0)
        mod, _ = build()
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
            )(mod)
        lib = tvm.compile(mod, target=target)

        a = np.zeros((ROWS, COLS), "float32")
        outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
                for _ in names]
        lib["main"](tvm.nd.array(a, dev), *outs)
        dev.sync()

        print(f"=== {kind}: both operands are exactly 1e-05")
        print(f"{'output':<24}{'bad':>5}{'expected':>13}{'good':>13}"
              f"{'bad_val':>13}   lane%16")
        for nm, exp, o in zip(names, expect, outs):
            got = o.numpy()[0]
            bad = np.flatnonzero(~np.isclose(got, exp, rtol=1e-3, atol=1e-12))
            bset = set(bad.tolist())
            good = [i for i in range(COLS) if i not in bset]
            gv = float(got[good[0]]) if good else float("nan")
            bv = float(got[bad[0]]) if len(bad) else float("nan")
            m16 = sorted(set(int(i) % 16 for i in bad)) if len(bad) else []
            print(f"{nm:<24}{len(bad):>5}{exp:>13.6g}{gv:>13.6g}"
                  f"{bv:>13.6g}   {m16}")

        if kind == "rocm":
            for m in [lib.mod] + list(lib.mod.imported_modules):
                for c in [m] + list(m.imported_modules):
                    try:
                        src = c.get_source("asm")
                    except Exception:  # noqa: BLE001
                        continue
                    if src and "v_sqrt" in src and "ds_bpermute" in src:
                        with open(f"{args.dump_dir}/provenance.s", "w") as fh:
                            fh.write(src)
                        ls = [l.strip() for l in src.splitlines()]
                        print("\n  every v_sqrt_f32 with its two preceding "
                              "instructions:")
                        for n, l in enumerate(ls):
                            if "v_sqrt_f32" in l:
                                ctx = [x for x in ls[max(0, n - 2):n + 2]
                                       if x and not x.startswith((".loc", ".cfi",
                                                                  ".Ltmp"))]
                                print("    " + " | ".join(c[:34] for c in ctx))
                        break
        print()


if __name__ == "__main__":
    main()
