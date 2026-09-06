#!/usr/bin/env python3
"""Is the DCU sqrt fault a hardware defect independent of TVM's backends?

The trail: layer_norm's Inf was traced past the allreduce (bit-identical and
correct on all 64 lanes) to sqrt returning exactly 0 on lanes with lane%16 in
{8,9,10,11}. It then reproduced for an ordinary per-lane operand, and on the
SYCL backend as well, with the operand itself verified correct in those lanes.

That points at v_sqrt_f32 rather than at TVM. This kernel removes everything
else: no reduction, no fusion, one elementwise sqrt (or rsqrt) of a uniform
constant, swept over magnitudes, on both backends. It also prints the emitted
instruction so the sqrt and rsqrt paths can be told apart -- the model passes on
SYCL, which would be explained if rsqrt lowers to v_rsq_f32 there while TVM's
ROCm backend expands rsqrt into 1.0/sqrt(x) and so emits v_sqrt_f32.
"""
import argparse
import os

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import te

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size=64")

ROWS, COLS = 16, 64
VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0]


def mod_for(op):
    A = te.placeholder((ROWS, COLS), name="A", dtype="float32")
    fn = te.sqrt if op == "sqrt" else te.rsqrt
    B = te.compute((ROWS, COLS), lambda i, j: fn(A[i, j]), name="B")
    return tvm.IRModule({"main": te.create_prim_func([A, B])})


def emitted(lib, op):
    """Report which transcendental instruction the backend actually emitted."""
    found = []
    for m in [lib.mod] + list(lib.mod.imported_modules):
        for c in [m] + list(m.imported_modules):
            for fmt in ("asm", ""):
                try:
                    src = c.get_source(fmt)
                except Exception:  # noqa: BLE001
                    continue
                if not src:
                    continue
                for ins in ("v_sqrt_f32", "v_rsq_f32", "v_rcp_f32",
                            "sqrt(", "rsqrt(", "native_sqrt"):
                    if ins in src and ins not in found:
                        found.append(ins)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa9")
    args = ap.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    for op in ("sqrt", "rsqrt"):
        for kind in ("rocm", "sycl"):
            target = tvm.target.Target(ROCM if kind == "rocm" else "sycl")
            dev = tvm.device(kind, 0)
            mod = mod_for(op)
            with target:
                mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
            lib = tvm.compile(mod, target=target)
            print(f"=== {op} on {kind}   emitted: {emitted(lib, op)}")
            print(f"    {'operand':>12}{'expected':>14}{'good':>14}"
                  f"{'bad':>5}{'bad_val':>12}   lane%16")
            for c in VALUES:
                a = np.full((ROWS, COLS), c, "float32")
                o = tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
                lib["main"](tvm.nd.array(a, dev), o)
                dev.sync()
                got = o.numpy()[0]
                exp = float(np.sqrt(c) if op == "sqrt" else 1.0 / np.sqrt(c))
                bad = np.flatnonzero(~np.isclose(got, exp, rtol=1e-3, atol=0))
                bset = set(bad.tolist())
                good = [i for i in range(COLS) if i not in bset]
                gv = float(got[good[0]]) if good else float("nan")
                bv = float(got[bad[0]]) if len(bad) else float("nan")
                m16 = sorted(set(int(i) % 16 for i in bad)) if len(bad) else []
                print(f"    {c:>12g}{exp:>14.6g}{gv:>14.6g}"
                      f"{len(bad):>5}{bv:>12.6g}   {m16}")
            print()


if __name__ == "__main__":
    main()
