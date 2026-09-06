#!/usr/bin/env python3
"""Dump the per-lane intermediates of a faithful layer_norm replica on DCU ROCm.

Reasoning so far. The real layer_norm kernel's generated code is correct on
inspection, a single-value cross-thread allreduce works, every output lane is
stored, and both allreduce lowerings fail identically. Most tellingly, forcing
x to all zeros still yields NaN on lanes with lane%16 in {8,9,10,11}, which the
dataflow cannot produce: a reduction of zeros is zero under any addressing, and
the LDS read address is uniform across lanes.

So the divergence has to be measured rather than inferred. This rebuilds
layer_norm with the same tuple comm_reducer topi uses and the same DLight
schedule, then exports the intermediates as separate per-lane buffers:

  SUM[i,j]   the reduced sum as seen by lane j
  SQ [i,j]   the reduced sum of squares as seen by lane j
  RS [i,j]   1/sqrt(var+eps) as seen by lane j
  OUT[i,j]   the final normalized value

Every lane of a row must agree on SUM, SQ and RS. Whichever is the first to
disagree localizes the defect to that stage.
"""
import argparse
import os

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import te, tir

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size={ws}")

ROWS, COLS = 16, 64
EPS = 1e-5


def build_mod():
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    G = te.placeholder((COLS,), name="G", dtype="float32")
    Bs = te.placeholder((COLS,), name="Bs", dtype="float32")
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

    def rs(i):
        mean = S[i] / COLS
        var = Q[i] / COLS - mean * mean
        return tir.const(1.0, "float32") / te.sqrt(var + tir.const(EPS, "float32"))

    SUM = te.compute((ROWS, COLS), lambda i, j: S[i], name="SUM")
    SQ = te.compute((ROWS, COLS), lambda i, j: Q[i], name="SQ")
    RS = te.compute((ROWS, COLS), lambda i, j: rs(i), name="RS")
    OUT = te.compute(
        (ROWS, COLS),
        lambda i, j: (A[i, j] - S[i] / COLS) * rs(i) * G[j] + Bs[j],
        name="OUT",
    )
    return tvm.IRModule(
        {"main": te.create_prim_func([A, G, Bs, SUM, SQ, RS, OUT])}
    )


def run(kind, ws, x_mode, dump=None):
    ts = ROCM.format(ws=ws) if kind == "rocm" else "sycl"
    target = tvm.target.Target(ts)
    dev = tvm.device(kind, 0)
    mod = build_mod()
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    lib = tvm.compile(mod, target=target)

    rng = np.random.default_rng(0)
    if x_mode == "zeros":
        a = np.zeros((ROWS, COLS), "float32")
    else:
        a = (rng.standard_normal((ROWS, COLS)) * 0.1).astype("float32")
    g = np.ones(COLS, "float32")
    b = np.zeros(COLS, "float32")

    args = [tvm.nd.array(a, dev), tvm.nd.array(g, dev), tvm.nd.array(b, dev)]
    outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev) for _ in range(4)]
    lib["main"](*args, *outs)
    dev.sync()
    SUM, SQ, RS, OUT = [o.numpy() for o in outs]

    label = f"{kind}" + (f" ws={ws}" if kind == "rocm" else "") + f" x={x_mode}"
    print(f"--- {label}")

    mean = a.sum(1) / COLS
    var = (a * a).sum(1) / COLS - mean * mean
    ref = {"SUM": a.sum(1), "SQ": (a * a).sum(1), "RS": 1.0 / np.sqrt(var + EPS)}
    for name, got in (("SUM", SUM), ("SQ", SQ), ("RS", RS)):
        row = got[0]
        uni = bool(np.allclose(row, row[0], equal_nan=True))
        bad = np.flatnonzero(~np.isclose(row, ref[name][0], rtol=1e-4, atol=1e-4))
        extra = ""
        if len(bad):
            extra = (f" lanes={bad.tolist()[:20]}"
                     f" lane%16={sorted(set(int(i) % 16 for i in bad))}"
                     f" vals={[round(float(row[i]), 4) for i in bad[:4]]}")
        print(f"    {name:<4} uniform={uni!s:<5} expected={ref[name][0]:<12.6f} "
              f"bad={len(bad)}{extra}")
    nf = int((~np.isfinite(OUT)).sum())
    print(f"    OUT  nonfinite={nf} of {OUT.size}"
          + (f" row0 lanes={np.flatnonzero(~np.isfinite(OUT[0])).tolist()}"
             if nf else ""))

    if dump:
        for m in [lib.mod] + list(lib.mod.imported_modules):
            for c in [m] + list(m.imported_modules):
                try:
                    src = c.get_source("asm")
                except Exception:  # noqa: BLE001
                    continue
                if src and ("ds_bpermute" in src or "s_barrier" in src):
                    with open(f"{dump}/stages_{kind}_ws{ws}.s", "w") as fh:
                        fh.write(src)
                    break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa4")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    for x_mode in ("random", "zeros"):
        for kind, ws in (("rocm", 64), ("rocm", 32), ("sycl", 0)):
            try:
                run(kind, ws, x_mode, dump=args.dump_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"--- {kind} ws={ws} x={x_mode} FAILED: "
                      f"{type(exc).__name__}: {' '.join(str(exc).split())[:250]}")


if __name__ == "__main__":
    main()
