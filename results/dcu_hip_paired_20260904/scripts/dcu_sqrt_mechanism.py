#!/usr/bin/env python3
"""Pin the mechanism: how the sqrt operand is produced decides correctness.

Trail of eliminations, all measured on DCU Z100L (gfx906):
  the cross-thread allreduce is bit-identical and correct on all 64 lanes;
  every output lane is stored; the generated LLVM IR and GCN ISA for layer_norm
  are correct on inspection; fdiv, fmul and exp are correct on the same
  operand; the fault appears on the SYCL backend too; and it appears in kernels
  containing no cross-lane traffic at all (ds_bpermute and s_barrier absent).

What still separated the passing from the failing cases was how the operand
reached v_sqrt_f32. Loading 1e-5 straight from global memory gave correct
results at every magnitude; forming it with one VALU add produced exactly 0 on
lanes with lane%16 in {8,9,10,11}, with s_nop absent from the ISA.

Each row below feeds v_sqrt_f32 the same value 1e-5 by a different route, so the
only variable is the producer of the operand and the distance to the consumer.
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
V = 1e-5


def build():
    """A holds 1e-5, Z holds 0. Every case computes sqrt of exactly 1e-5."""
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    Z = te.placeholder((ROWS, COLS), name="Z", dtype="float32")
    one = tir.const(1.0, "float32")

    cases = [
        # operand straight out of the load, only an s_waitcnt in between
        ("load_direct", lambda i, j: te.sqrt(A[i, j])),
        # operand produced by one VALU op immediately before the sqrt
        ("valu_add_const", lambda i, j: te.sqrt(A[i, j] + tir.const(V, "float32") * Z[i, j])),
        ("valu_add_mem", lambda i, j: te.sqrt(A[i, j] + Z[i, j])),
        ("valu_mul", lambda i, j: te.sqrt(A[i, j] * one)),
        # a longer dependency chain before the sqrt
        ("valu_chain4", lambda i, j: te.sqrt(
            ((A[i, j] + Z[i, j]) * one + Z[i, j]) * one)),
    ]
    outs = [te.compute((ROWS, COLS), fn, name=nm) for nm, fn in cases]
    return tvm.IRModule({"main": te.create_prim_func([A, Z] + outs)}), \
        [nm for nm, _ in cases]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa11")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    exp = float(np.sqrt(V))
    for kind in ("rocm", "sycl"):
        target = tvm.target.Target(ROCM if kind == "rocm" else "sycl")
        dev = tvm.device(kind, 0)
        mod, names = build()
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        lib = tvm.compile(mod, target=target)

        a = np.full((ROWS, COLS), V, "float32")
        z = np.zeros((ROWS, COLS), "float32")
        outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
                for _ in names]
        lib["main"](tvm.nd.array(a, dev), tvm.nd.array(z, dev), *outs)
        dev.sync()

        print(f"=== {kind}: every case computes sqrt(1e-05), expected {exp:.6g}")
        print(f"{'operand route':<18}{'bad':>5}{'good':>13}{'bad_val':>11}"
              f"   lane%16")
        for nm, o in zip(names, outs):
            got = o.numpy()[0]
            bad = np.flatnonzero(~np.isclose(got, exp, rtol=1e-3, atol=0))
            bset = set(bad.tolist())
            good = [i for i in range(COLS) if i not in bset]
            gv = float(got[good[0]]) if good else float("nan")
            bv = float(got[bad[0]]) if len(bad) else float("nan")
            m16 = sorted(set(int(i) % 16 for i in bad)) if len(bad) else []
            print(f"{nm:<18}{len(bad):>5}{gv:>13.6g}{bv:>11.6g}   {m16}")

        if kind == "rocm":
            for m in [lib.mod] + list(lib.mod.imported_modules):
                for c in [m] + list(m.imported_modules):
                    try:
                        src = c.get_source("asm")
                    except Exception:  # noqa: BLE001
                        continue
                    if src and "v_sqrt_f32" in src:
                        with open(f"{args.dump_dir}/mechanism.s", "w") as fh:
                            fh.write(src)
                        ls = [l.strip() for l in src.splitlines()]
                        print(f"\n  s_nop in module: {src.count('s_nop')}")
                        print("  each v_sqrt_f32 with the instruction feeding it:")
                        for n, l in enumerate(ls):
                            if "v_sqrt_f32" in l:
                                prev = [x for x in ls[max(0, n - 2):n]
                                        if x and not x.startswith((".loc", ".cfi",
                                                                   ".Ltmp"))]
                                print(f"    {' ; '.join(p[:38] for p in prev)}"
                                      f"  ==>  {l[:30]}")
                        break
        print()


if __name__ == "__main__":
    main()
