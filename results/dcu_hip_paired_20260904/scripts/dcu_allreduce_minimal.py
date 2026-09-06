#!/usr/bin/env python3
"""Minimal standalone cross-thread allreduce reproducer for the DCU ROCm bug.

Everything else has been ruled out on the real layer_norm kernel:

  * both allreduce lowerings (ds_bpermute warp shuffle at thread_warp_size=64,
    shared memory + barriers at 32) fail identically;
  * every output lane is stored, so it is not a predication bug;
  * the generated LLVM IR and GCN ISA are correct -- shuffle tree offsets,
    vcc guards, the s_waitcnt before s_barrier, the 8-byte LDS allocation, and
    the uniform ds_read2_b32 address;
  * with x forced to all zeros the dataflow cannot produce NaN, yet 16 of 64
    lanes still do.

That leaves exactly one possibility: those lanes obtain a different reduced
value than the rest. This kernel tests it directly. It reduces a row and stores
the broadcast sum per lane, with no sqrt and no division, so a garbage value is
observed as-is instead of being turned into NaN.

Every lane of a row must receive the same sum. Bad lanes are expected at
lane % 16 in {8,9,10,11} if the earlier characterization holds.
"""
import argparse

import numpy as np
import tvm
from tvm import tir
from tvm.script import tir as T

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size={ws}")

ROWS, COLS = 16, 64


@T.prim_func
def sum_bcast(A: T.Buffer((ROWS, COLS), "float32"),
              B: T.Buffer((ROWS, COLS), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    S = T.alloc_buffer((ROWS,), "float32")
    for i, k in T.grid(ROWS, COLS):
        with T.block("sum"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                S[vi] = T.float32(0)
            S[vi] = S[vi] + A[vi, vk]
    for i, j in T.grid(ROWS, COLS):
        with T.block("bcast"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = S[vi]


@T.prim_func
def sum_sq_bcast(A: T.Buffer((ROWS, COLS), "float32"),
                 B: T.Buffer((ROWS, COLS), "float32"),
                 C: T.Buffer((ROWS, COLS), "float32")):
    """Two accumulators in one reduction block, as layer_norm has.

    This mirrors x_red_temp_v0 (sum) and x_red_temp_v1 (sum of squares), which
    lower to a paired allreduce: two ds_bpermute chains, a ds_write2_b32 and a
    ds_read2_b32 over an 8-byte LDS allocation.
    """
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    S = T.alloc_buffer((ROWS,), "float32")
    Q = T.alloc_buffer((ROWS,), "float32")
    for i, k in T.grid(ROWS, COLS):
        with T.block("red"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                S[vi] = T.float32(0)
                Q[vi] = T.float32(0)
            S[vi] = S[vi] + A[vi, vk]
            Q[vi] = Q[vi] + A[vi, vk] * A[vi, vk]
    for i, j in T.grid(ROWS, COLS):
        with T.block("out"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = S[vi]
            C[vi, vj] = Q[vi]


def schedule(two=False):
    func = sum_sq_bcast if two else sum_bcast
    sch = tir.Schedule(tvm.IRModule({"main": func}))
    b_red = sch.get_block("red" if two else "sum")
    b_out = sch.get_block("out" if two else "bcast")
    i, k = sch.get_loops(b_red)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")
    sch.set_scope(b_red, 0, "shared")
    if two:
        sch.set_scope(b_red, 1, "shared")
    i2, j2 = sch.get_loops(b_out)
    sch.bind(i2, "blockIdx.x")
    sch.bind(j2, "threadIdx.x")
    return sch.mod


def report(tag, got, ref):
    row = got[0]
    bad = np.flatnonzero(~np.isclose(row, ref[0], atol=1e-3))
    print(f"    {tag}: uniform={np.allclose(row, row[0])} "
          f"expected={ref[0]:.6f} bad_lanes={len(bad)}")
    if len(bad):
        print(f"      lanes: {bad.tolist()}")
        print(f"      values: {[round(float(row[i]), 4) for i in bad[:8]]}")
        print(f"      lane%16: {sorted(set(int(i) % 16 for i in bad))}")
    print(f"      max abs err all rows: {np.abs(got - ref[:, None]).max():.6g}")


def run(kind, ws, two=False, dump=None):
    ts = ROCM.format(ws=ws) if kind == "rocm" else "sycl"
    target = tvm.target.Target(ts)
    dev = tvm.device(kind, 0)
    lib = tvm.compile(schedule(two), target=target)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((ROWS, COLS)).astype("float32")

    a_d = tvm.nd.array(a, dev)
    outs = [tvm.nd.array(np.zeros((ROWS, COLS), "float32"), dev)
            for _ in range(2 if two else 1)]
    lib["main"](a_d, *outs)
    dev.sync()

    label = ("2-value " if two else "1-value ") + kind + \
             (f" ws={ws}" if kind == "rocm" else "")
    print(f"--- {label}")
    report("sum", outs[0].numpy(), a.sum(1))
    if two:
        report("sumsq", outs[1].numpy(), (a * a).sum(1))

    if dump:
        for fmt, ext in (("asm", "s"), ("llvm", "ll")):
            for m in [lib.mod] + list(lib.mod.imported_modules):
                for c in [m] + list(m.imported_modules):
                    try:
                        src = c.get_source(fmt)
                    except Exception:  # noqa: BLE001
                        continue
                    if src and ("amdgcn" in src or "ds_bpermute" in src or
                                "s_barrier" in src):
                        tag = "two" if two else "one"
                        with open(f"{dump}/minimal_{tag}_ws{ws}.{ext}", "w") as fh:
                            fh.write(src)
                        break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa3")
    args = ap.parse_args()
    import os
    os.makedirs(args.dump_dir, exist_ok=True)

    print("=== scheduled TIR (2-value) ===")
    print(str(schedule(two=True)["main"])[:1700])
    print()

    for two in (False, True):
        for kind, ws in (("rocm", 64), ("rocm", 32), ("sycl", 0)):
            try:
                run(kind, ws, two=two, dump=args.dump_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"--- {'2' if two else '1'}-value {kind} ws={ws} FAILED: "
                      f"{type(exc).__name__}: {' '.join(str(exc).split())[:250]}")


if __name__ == "__main__":
    main()
