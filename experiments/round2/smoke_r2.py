#!/usr/bin/env python3
"""Smoke test of the overlay_r2 build: allreduce lowering, grid mapping, runtime sync modes."""
import json
import os
import sys

import numpy as np
import tvm
from tvm.script import tir as T

TARGET = tvm.target.Target(
    "sycl -keys=sycl,gpu -max_num_threads=1024 -max_shared_memory_per_block=49152 "
    "-max_threads_per_block=1024 -thread_warp_size=32")
dev = tvm.device("oneapi", 0)
out = {"libtvm": tvm.base._LIB._name if hasattr(tvm.base, "_LIB") else "",
       "env": {k: os.environ.get(k, "") for k in ("TVM_SYCL_SYNC_EACH_LAUNCH", "TVM_SYCL_LEGACY_WARP_SYNC")}}


@T.prim_func
def row_sum(A: T.Buffer((256, 1024), "float32"), B: T.Buffer((256,), "float32")):
    for i in T.thread_binding(256, thread="blockIdx.x"):
        for k in T.thread_binding(16, thread="threadIdx.x"):   # 16-lane cross-thread reduction
            with T.block("sum"):
                vi, vk = T.axis.remap("SR", [i, k])
                with T.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk * 64] + A[vi, vk * 64 + 1]


@T.prim_func
def big_grid(A: T.Buffer((6400000,), "float32"), B: T.Buffer((6400000,), "float32")):
    for bx in T.thread_binding(100000, thread="blockIdx.x"):
        for tx in T.thread_binding(64, thread="threadIdx.x"):
            with T.block("add"):
                v = T.axis.spatial(6400000, bx * 64 + tx)
                B[v] = A[v] + T.float32(1)


def run(name, func, args_np, ref):
    try:
        lib = tvm.build(tvm.IRModule({"main": func}), target=TARGET)
        src = "\n".join(m.get_source() for m in lib.imported_modules)
        arrs = [tvm.nd.array(a, dev) for a in args_np]
        lib(*arrs)
        dev.sync()
        got = arrs[-1].numpy()
        out[name] = {"pass": bool(np.allclose(got, ref, atol=1e-4, rtol=1e-4)),
                     "warp_noop_sites": src.count("warp storage_sync: no-op"),
                     "group_barriers": src.count("group_barrier("),
                     "uses_get_group_2": "item.get_group(2)" in src}
    except Exception as exc:
        out[name] = {"error": repr(exc)[:400]}


a = np.random.default_rng(0).uniform(-1, 1, (256, 1024)).astype("float32")
ref = a[:, 0::64].sum(axis=1) + a[:, 1::64].sum(axis=1)
run("row_sum_allreduce", row_sum, [a, np.zeros(256, "float32")], ref)
x = np.random.default_rng(1).uniform(-1, 1, 6400000).astype("float32")
run("big_grid_100000_blocks", big_grid, [x, np.zeros_like(x)], x + 1)
print(json.dumps(out, indent=1))
