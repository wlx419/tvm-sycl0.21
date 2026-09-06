#!/usr/bin/env python3
"""Confirm that a cross-lane reduction in the same kernel is what breaks sqrt.

Two kernels with an identical sqrt expression on an identical per-lane operand:

  with_reduction    the kernel also produces outputs derived from a two-value
                    cross-thread allreduce, so it contains ds_bpermute / LDS
                    cross-lane traffic
  without_reduction the reduction-derived outputs are removed; everything else,
                    including the sqrt, is unchanged

The operand is A[i,j]+1e-5 in both, so the sqrt input is bit-identical. If only
the first fails, the trigger is the cross-lane traffic in the wave rather than
sqrt, the operand, the backend, or the reduction's result.
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


def build(with_reduction):
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    outs = [te.compute((ROWS, COLS), lambda i, j: te.sqrt(A[i, j] + EPS),
                       name="sqrt_lane")]
    if with_reduction:
        k = te.reduce_axis((0, COLS), name="k")

        def fcombine(x, y):
            return x[0] + y[0], x[1] + y[1]

        def fidentity(t0, t1):
            return tir.const(0, t0), tir.const(0, t1)

        pair = te.comm_reducer(fcombine, fidentity, name="pair")
        S, Q = te.compute((ROWS,),
                          lambda i: pair((A[i, k], A[i, k] * A[i, k]), axis=k),
                          name="red")
        outs.append(te.compute((ROWS, COLS), lambda i, j: S[i], name="sum_out"))
        outs.append(te.compute((ROWS, COLS), lambda i, j: Q[i], name="sq_out"))
    return tvm.IRModule({"main": te.create_prim_func([A] + outs)}), len(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa10")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    exp = float(np.sqrt(1e-5))
    for kind in ("rocm", "sycl"):
        print(f"=== {kind}: sqrt(A[i,j]+1e-5), operand bit-identical in both")
        for with_red in (True, False):
            target = tvm.target.Target(ROCM if kind == "rocm" else "sycl")
            dev = tvm.device(kind, 0)
            mod, n_out = build(with_red)
            with target:
                mod = dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
                )(mod)
            lib = tvm.compile(mod, target=target)

            a = np.zeros((ROWS, COLS), "float32")
            outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
                    for _ in range(n_out)]
            lib["main"](tvm.nd.array(a, dev), *outs)
            dev.sync()
            got = outs[0].numpy()[0]
            bad = np.flatnonzero(~np.isclose(got, exp, rtol=1e-3, atol=0))
            bset = set(bad.tolist())
            good = [i for i in range(COLS) if i not in bset]
            gv = float(got[good[0]]) if good else float("nan")
            bv = float(got[bad[0]]) if len(bad) else float("nan")
            m16 = sorted(set(int(i) % 16 for i in bad)) if len(bad) else []

            xlane = ""
            if kind == "rocm":
                for m in [lib.mod] + list(lib.mod.imported_modules):
                    for c in [m] + list(m.imported_modules):
                        try:
                            src = c.get_source("asm")
                        except Exception:  # noqa: BLE001
                            continue
                        if src and "v_sqrt_f32" in src:
                            xlane = (f"bpermute={src.count('ds_bpermute')} "
                                     f"barrier={src.count('s_barrier')} "
                                     f"s_nop={src.count('s_nop')}")
                            tag = "with" if with_red else "without"
                            with open(f"{args.dump_dir}/trigger_{tag}.s", "w") as fh:
                                fh.write(src)
                            break
                    if xlane:
                        break

            tag = "with_reduction   " if with_red else "without_reduction"
            print(f"  {tag} bad={len(bad):<3} expected={exp:.6g} "
                  f"good={gv:.6g} bad_val={bv:<10.6g} lane%16={m16}  {xlane}")
        print()


if __name__ == "__main__":
    main()
