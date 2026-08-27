"""Extended TVM-SYCL correctness and latency probes.

This script is intentionally self-contained so it can be copied to the
NVIDIA and DCU machines and run against an existing TVM-SYCL build.

It exercises nontrivial kernels that complement the elementwise checks in
the paper: matmul, batched matmul, row-wise reduction, conv2d, softmax,
layer normalization, GELU, max-pooling, and transpose/layout conversion.
Each experiment is executed in a subprocess to avoid cross-case state
contamination in TVM/SYCL runtime setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_WARMUP = 10
DEFAULT_MEASURE = 30
DEFAULT_TOL_ABS = 1e-4
DEFAULT_TOL_REL = 1e-3


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def max_errors(actual: Any, expected: Any) -> Tuple[float, float]:
    import numpy as np

    diff = np.abs(actual - expected)
    abs_err = float(np.max(diff))
    denom = np.maximum(np.abs(expected), 1e-7)
    rel_err = float(np.max(diff / denom))
    return abs_err, rel_err


def parse_shape(shape: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in shape.lower().replace("x", ",").split(",") if x)


def build_from_tir_source(src: str) -> Any:
    import tvm

    return tvm.build(tvm.IRModule({"main": tvm.script.from_source(src)}), target="sycl")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_matmul(shape: Tuple[int, ...], wg_size: int) -> Any:
    m, n, k = shape
    total = m * n
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({m}, {k}), "float32"),
         B: T.Buffer(({k}, {n}), "float32"),
         C: T.Buffer(({m}, {n}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {n}
                j = idx - i * {n}
                C[i, j] = T.float32(0)
                for rk in range({k}):
                    C[i, j] = C[i, j] + A[i, rk] * B[rk, j]
"""
    return build_from_tir_source(src)


def build_batch_matmul(shape: Tuple[int, ...], wg_size: int) -> Any:
    batch, m, n, k = shape
    total = batch * m * n
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({batch}, {m}, {k}), "float32"),
         B: T.Buffer(({batch}, {k}, {n}), "float32"),
         C: T.Buffer(({batch}, {m}, {n}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                j = idx % {n}
                t0 = idx // {n}
                i = t0 % {m}
                b = t0 // {m}
                C[b, i, j] = T.float32(0)
                for rk in range({k}):
                    C[b, i, j] = C[b, i, j] + A[b, i, rk] * B[b, rk, j]
"""
    return build_from_tir_source(src)


def build_row_sum(shape: Tuple[int, ...], wg_size: int) -> Any:
    m, k = shape
    nblocks = ceil_div(m, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({m}, {k}), "float32"),
         C: T.Buffer(({m},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            i = bx * {wg_size} + tx
            if i < {m}:
                C[i] = T.float32(0)
                for rk in range({k}):
                    C[i] = C[i] + A[i, rk]
"""
    return build_from_tir_source(src)


def build_layer_norm(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    total = rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                mean = T.float32(0)
                for rk1 in range({cols}):
                    mean = mean + A[i, rk1]
                mean = mean / T.float32({cols}.0)
                var = T.float32(0)
                for rk2 in range({cols}):
                    diff = A[i, rk2] - mean
                    var = var + diff * diff
                var = var / T.float32({cols}.0)
                Out[i, j] = (A[i, j] - mean) / T.sqrt(var + T.float32(0.00001))
"""
    return build_from_tir_source(src)


def build_layer_norm_mean(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    nblocks = ceil_div(rows, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Mean: T.Buffer(({rows},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            i = bx * {wg_size} + tx
            if i < {rows}:
                Mean[i] = T.float32(0)
                for rk in range({cols}):
                    Mean[i] = Mean[i] + A[i, rk]
                Mean[i] = Mean[i] / T.float32({cols}.0)
"""
    return build_from_tir_source(src)


def build_layer_norm_var(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    nblocks = ceil_div(rows, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Mean: T.Buffer(({rows},), "float32"),
         Var: T.Buffer(({rows},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            i = bx * {wg_size} + tx
            if i < {rows}:
                Var[i] = T.float32(0)
                for rk in range({cols}):
                    diff = A[i, rk] - Mean[i]
                    Var[i] = Var[i] + diff * diff
                Var[i] = Var[i] / T.float32({cols}.0)
"""
    return build_from_tir_source(src)


def build_layer_norm_normalize(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    total = rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Mean: T.Buffer(({rows},), "float32"),
         Var: T.Buffer(({rows},), "float32"),
         Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                Out[i, j] = (A[i, j] - Mean[i]) / T.sqrt(Var[i] + T.float32(0.00001))
"""
    return build_from_tir_source(src)


def build_conv2d_nhwc(shape: Tuple[int, ...], wg_size: int) -> Any:
    batch, height, width, in_ch, out_ch, kh, kw = shape
    oh = height - kh + 1
    ow = width - kw + 1
    total = batch * oh * ow * out_ch
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(Data: T.Buffer(({batch}, {height}, {width}, {in_ch}), "float32"),
         Filter: T.Buffer(({kh}, {kw}, {in_ch}, {out_ch}), "float32"),
         Out: T.Buffer(({batch}, {oh}, {ow}, {out_ch}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                co = idx % {out_ch}
                t0 = idx // {out_ch}
                x = t0 % {ow}
                t1 = t0 // {ow}
                y = t1 % {oh}
                n = t1 // {oh}
                Out[n, y, x, co] = T.float32(0)
                for rkh in range({kh}):
                    for rkw in range({kw}):
                        for rc in range({in_ch}):
                            Out[n, y, x, co] = Out[n, y, x, co] + Data[n, y + rkh, x + rkw, rc] * Filter[rkh, rkw, rc, co]
"""
    return build_from_tir_source(src)


def build_max_pool2d_nhwc(shape: Tuple[int, ...], wg_size: int) -> Any:
    batch, height, width, channels, kh, kw, stride = shape
    oh = (height - kh) // stride + 1
    ow = (width - kw) // stride + 1
    total = batch * oh * ow * channels
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(Data: T.Buffer(({batch}, {height}, {width}, {channels}), "float32"),
         Out: T.Buffer(({batch}, {oh}, {ow}, {channels}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                c = idx % {channels}
                t0 = idx // {channels}
                ox = t0 % {ow}
                t1 = t0 // {ow}
                oy = t1 % {oh}
                n = t1 // {oh}
                Out[n, oy, ox, c] = T.float32(-3.4028234663852886e38)
                for rkh in range({kh}):
                    for rkw in range({kw}):
                        val = Data[n, oy * {stride} + rkh, ox * {stride} + rkw, c]
                        Out[n, oy, ox, c] = T.Select(val > Out[n, oy, ox, c], val, Out[n, oy, ox, c])
"""
    return build_from_tir_source(src)


def build_softmax(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    total = rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                row_max = T.float32(-3.4028234663852886e38)
                for rk1 in range({cols}):
                    row_max = T.Select(A[i, rk1] > row_max, A[i, rk1], row_max)
                denom = T.float32(0)
                for rk2 in range({cols}):
                    denom = denom + T.exp(A[i, rk2] - row_max)
                Out[i, j] = T.exp(A[i, j] - row_max) / denom
"""
    return build_from_tir_source(src)


def build_gelu(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    total = rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                x = A[i, j]
                inner = T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x)
                Out[i, j] = T.float32(0.5) * x * (T.float32(1) + T.tanh(inner))
"""
    return build_from_tir_source(src)


def build_transpose_3d(shape: Tuple[int, ...], wg_size: int) -> Any:
    batch, rows, cols = shape
    total = batch * rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({batch}, {rows}, {cols}), "float32"),
         Out: T.Buffer(({batch}, {cols}, {rows}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                c = idx % {cols}
                t0 = idx // {cols}
                r = t0 % {rows}
                b = t0 // {rows}
                Out[b, c, r] = A[b, r, c]
"""
    return build_from_tir_source(src)


def build_softmax_row_max(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    nblocks = ceil_div(rows, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         RowMax: T.Buffer(({rows},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            i = bx * {wg_size} + tx
            if i < {rows}:
                RowMax[i] = T.float32(-3.4028234663852886e38)
                for rk in range({cols}):
                    RowMax[i] = T.Select(A[i, rk] > RowMax[i], A[i, rk], RowMax[i])
"""
    return build_from_tir_source(src)


def build_softmax_denom(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    nblocks = ceil_div(rows, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         RowMax: T.Buffer(({rows},), "float32"),
         Denom: T.Buffer(({rows},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            i = bx * {wg_size} + tx
            if i < {rows}:
                Denom[i] = T.float32(0)
                for rk in range({cols}):
                    Denom[i] = Denom[i] + T.exp(A[i, rk] - RowMax[i])
"""
    return build_from_tir_source(src)


def build_softmax_normalize(shape: Tuple[int, ...], wg_size: int) -> Any:
    rows, cols = shape
    total = rows * cols
    nblocks = ceil_div(total, wg_size)
    src = f"""@T.prim_func
def main(A: T.Buffer(({rows}, {cols}), "float32"),
         RowMax: T.Buffer(({rows},), "float32"),
         Denom: T.Buffer(({rows},), "float32"),
         Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({nblocks}, thread="blockIdx.x"):
        for tx in T.thread_binding({wg_size}, thread="threadIdx.x"):
            idx = bx * {wg_size} + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                Out[i, j] = T.exp(A[i, j] - RowMax[i]) / Denom[i]
"""
    return build_from_tir_source(src)


def reference_conv2d_nhwc(data: Any, filt: Any) -> Any:
    import numpy as np

    batch, height, width, in_ch = data.shape
    kh, kw, _, out_ch = filt.shape
    oh = height - kh + 1
    ow = width - kw + 1
    out = np.zeros((batch, oh, ow, out_ch), dtype="float32")
    for n in range(batch):
        for y in range(oh):
            for x in range(ow):
                patch = data[n, y : y + kh, x : x + kw, :]
                for co in range(out_ch):
                    out[n, y, x, co] = np.sum(patch * filt[:, :, :, co], dtype="float32")
    return out


def reference_max_pool2d_nhwc(data: Any, kh: int, kw: int, stride: int) -> Any:
    import numpy as np

    batch, height, width, channels = data.shape
    oh = (height - kh) // stride + 1
    ow = (width - kw) // stride + 1
    out = np.empty((batch, oh, ow, channels), dtype="float32")
    for n in range(batch):
        for oy in range(oh):
            for ox in range(ow):
                patch = data[n, oy * stride : oy * stride + kh, ox * stride : ox * stride + kw, :]
                out[n, oy, ox, :] = np.max(patch, axis=(0, 1))
    return out


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import tvm

    np.random.seed(args.seed)
    shape = parse_shape(args.shape)
    dev = tvm.device("sycl", args.device_id)
    if not dev.exist:
        raise RuntimeError("tvm.device('sycl', 0) does not exist")

    t_build0 = time.perf_counter()
    invoke = None
    out_arr = None
    extra_fields: Dict[str, Any] = {}

    if args.op == "matmul":
        rt_mod = build_matmul(shape, args.wg_size)
        m, n, k = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(m, k)).astype("float32")
        b_np = np.random.uniform(-1.0, 1.0, size=(k, n)).astype("float32")
        expected = np.matmul(a_np, b_np).astype("float32")
        out_np = np.zeros((m, n), dtype="float32")
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(b_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "batch_matmul":
        rt_mod = build_batch_matmul(shape, args.wg_size)
        batch, m, n, k = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(batch, m, k)).astype("float32")
        b_np = np.random.uniform(-1.0, 1.0, size=(batch, k, n)).astype("float32")
        expected = np.matmul(a_np, b_np).astype("float32")
        out_np = np.zeros((batch, m, n), dtype="float32")
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(b_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "row_sum":
        rt_mod = build_row_sum(shape, args.wg_size)
        m, k = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(m, k)).astype("float32")
        expected = np.sum(a_np, axis=1, dtype="float32")
        out_np = np.zeros((m,), dtype="float32")
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "layer_norm":
        mean_mod = build_layer_norm_mean(shape, args.wg_size)
        var_mod = build_layer_norm_var(shape, args.wg_size)
        norm_mod = build_layer_norm_normalize(shape, args.wg_size)
        rows, cols = shape
        a_np = np.random.uniform(-3.0, 3.0, size=(rows, cols)).astype("float32")
        mean = np.mean(a_np, axis=1, keepdims=True, dtype="float32")
        var = np.mean((a_np - mean) * (a_np - mean), axis=1, keepdims=True, dtype="float32")
        expected = ((a_np - mean) / np.sqrt(var + np.float32(1e-5))).astype("float32")
        out_np = np.zeros((rows, cols), dtype="float32")
        a_arr = tvm.nd.array(a_np, dev)
        mean_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
        var_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
        out_arr = tvm.nd.array(out_np, dev)

        def invoke_layer_norm() -> None:
            mean_mod(a_arr, mean_arr)
            var_mod(a_arr, mean_arr, var_arr)
            norm_mod(a_arr, mean_arr, var_arr, out_arr)

        invoke = invoke_layer_norm
        extra_fields["layer_norm_stages"] = 3
    elif args.op == "conv2d_nhwc":
        rt_mod = build_conv2d_nhwc(shape, args.wg_size)
        batch, height, width, in_ch, out_ch, kh, kw = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(batch, height, width, in_ch)).astype("float32")
        w_np = np.random.uniform(-1.0, 1.0, size=(kh, kw, in_ch, out_ch)).astype("float32")
        expected = reference_conv2d_nhwc(a_np, w_np)
        out_np = np.zeros_like(expected)
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(w_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "max_pool2d_nhwc":
        rt_mod = build_max_pool2d_nhwc(shape, args.wg_size)
        batch, height, width, channels, kh, kw, stride = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(batch, height, width, channels)).astype("float32")
        expected = reference_max_pool2d_nhwc(a_np, kh, kw, stride)
        out_np = np.zeros_like(expected)
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "softmax":
        rows, cols = shape
        max_mod = build_softmax_row_max(shape, args.wg_size)
        denom_mod = build_softmax_denom(shape, args.wg_size)
        norm_mod = build_softmax_normalize(shape, args.wg_size)
        a_np = np.random.uniform(-3.0, 3.0, size=(rows, cols)).astype("float32")
        shifted = a_np - np.max(a_np, axis=1, keepdims=True)
        expv = np.exp(shifted).astype("float32")
        expected = (expv / np.sum(expv, axis=1, keepdims=True)).astype("float32")
        out_np = np.zeros((rows, cols), dtype="float32")
        a_arr = tvm.nd.array(a_np, dev)
        max_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
        denom_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
        out_arr = tvm.nd.array(out_np, dev)

        def invoke_softmax() -> None:
            max_mod(a_arr, max_arr)
            denom_mod(a_arr, max_arr, denom_arr)
            norm_mod(a_arr, max_arr, denom_arr, out_arr)

        invoke = invoke_softmax
        extra_fields["softmax_stages"] = 3
    elif args.op == "gelu":
        rt_mod = build_gelu(shape, args.wg_size)
        rows, cols = shape
        a_np = np.random.uniform(-5.0, 5.0, size=(rows, cols)).astype("float32")
        expected = (
            0.5
            * a_np
            * (1.0 + np.tanh(np.float32(0.7978845608028654) * (a_np + np.float32(0.044715) * a_np * a_np * a_np)))
        ).astype("float32")
        out_np = np.zeros((rows, cols), dtype="float32")
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    elif args.op == "transpose_3d":
        rt_mod = build_transpose_3d(shape, args.wg_size)
        batch, rows, cols = shape
        a_np = np.random.uniform(-1.0, 1.0, size=(batch, rows, cols)).astype("float32")
        expected = np.transpose(a_np, (0, 2, 1)).astype("float32")
        out_np = np.zeros((batch, cols, rows), dtype="float32")
        tvm_args = [tvm.nd.array(a_np, dev), tvm.nd.array(out_np, dev)]
        out_arr = tvm_args[-1]
        invoke = lambda: rt_mod(*tvm_args)
    else:
        raise ValueError(f"unknown op: {args.op}")
    build_s = time.perf_counter() - t_build0

    assert invoke is not None and out_arr is not None
    invoke()
    dev.sync()
    actual = out_arr.numpy()
    abs_err, rel_err = max_errors(actual, expected)

    for _ in range(args.warmup):
        invoke()
    dev.sync()

    lat_ms = []
    for _ in range(args.measure):
        t0 = time.perf_counter_ns()
        invoke()
        dev.sync()
        lat_ms.append((time.perf_counter_ns() - t0) / 1e6)

    p50 = percentile(lat_ms, 0.50)
    p90 = percentile(lat_ms, 0.90)
    p99 = percentile(lat_ms, 0.99)
    is_close = bool(np.allclose(actual, expected, atol=args.tol_abs, rtol=args.tol_rel))
    status = "pass" if is_close else "fail_out_of_tolerance"
    result = {
        "op": args.op,
        "shape": args.shape,
        "wg_size": args.wg_size,
        "status": status,
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "latency_mean_ms": float(np.mean(lat_ms)),
        "latency_p50_ms": p50,
        "latency_p90_ms": p90,
        "latency_p99_ms": p99,
        "tail_ratio_p99_p50": p99 / p50 if p50 > 0 else float("nan"),
        "build_s": build_s,
        "warmup": args.warmup,
        "measure": args.measure,
        "tol_abs": args.tol_abs,
        "tol_rel": args.tol_rel,
    }
    result.update(extra_fields)
    return result


def base_experiments() -> List[Tuple[str, str, int]]:
    return [
        ("matmul", "64x64x64", 256),
        ("matmul", "128x128x128", 256),
        ("matmul", "256x256x256", 256),
        ("batch_matmul", "8x32x32x64", 256),
        ("row_sum", "256x1024", 256),
        ("row_sum", "1024x1024", 256),
        ("layer_norm", "128x768", 256),
        ("conv2d_nhwc", "1x32x32x3x16x3x3", 256),
        ("conv2d_nhwc", "1x32x32x16x32x3x3", 256),
        ("conv2d_nhwc", "8x32x32x16x32x3x3", 256),
        ("conv2d_nhwc", "16x32x32x16x32x3x3", 256),
        ("max_pool2d_nhwc", "8x32x32x16x2x2x2", 256),
        ("softmax", "128x256", 256),
        ("softmax", "512x512", 256),
        ("gelu", "256x1024", 256),
        ("transpose_3d", "16x128x64", 256),
    ]


def wg_experiments() -> List[Tuple[str, str, int]]:
    cases = []
    for wg in [32, 64, 128, 256, 512, 1024]:
        cases.append(("matmul", "128x128x128", wg))
        cases.append(("conv2d_nhwc", "1x32x32x16x32x3x3", wg))
        cases.append(("row_sum", "1024x1024", wg))
    return cases


def all_experiments(suite: str) -> List[Tuple[str, str, int]]:
    if suite == "correctness":
        return base_experiments()
    if suite == "wg_ablation":
        return wg_experiments()
    if suite == "all":
        return base_experiments() + wg_experiments()
    raise ValueError(f"unknown suite: {suite}")


def run_case(
    script_path: Path,
    op: str,
    shape: str,
    wg_size: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--op",
        op,
        "--shape",
        shape,
        "--wg-size",
        str(wg_size),
        "--warmup",
        str(args.warmup),
        "--measure",
        str(args.measure),
        "--seed",
        str(args.seed),
        "--tol-abs",
        str(args.tol_abs),
        "--tol-rel",
        str(args.tol_rel),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_s)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "op": op,
            "shape": shape,
            "wg_size": wg_size,
            "status": "runtime_error",
            "elapsed_s": elapsed,
            "error": (proc.stderr or proc.stdout)[-2000:],
        }
    try:
        rec = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "op": op,
            "shape": shape,
            "wg_size": wg_size,
            "status": "parse_error",
            "elapsed_s": elapsed,
            "error": f"{exc}: stdout={proc.stdout[-2000:]!r}, stderr={proc.stderr[-1000:]!r}",
        }
    rec["elapsed_s"] = elapsed
    return rec


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    passed = [r for r in rows if r.get("status") == "pass"]
    return {
        "total": len(rows),
        "pass": len(passed),
        "non_pass": len(rows) - len(passed),
        "pass_rate": len(passed) / len(rows) if rows else 0.0,
        "max_abs_err": max((float(r.get("max_abs_err", 0.0)) for r in passed), default=float("nan")),
        "max_rel_err": max((float(r.get("max_rel_err", 0.0)) for r in passed), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--suite", choices=["correctness", "wg_ablation", "all"], default="all")
    parser.add_argument("--op")
    parser.add_argument("--shape")
    parser.add_argument("--wg-size", type=int, default=256)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--measure", type=int, default=DEFAULT_MEASURE)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol-abs", type=float, default=DEFAULT_TOL_ABS)
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--stem", default="tvm_sycl_extended_probe")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(run_worker(args)))
        return

    script_path = Path(__file__).resolve()
    rows: List[Dict[str, Any]] = []
    cases = all_experiments(args.suite)
    for idx, (op, shape, wg_size) in enumerate(cases, 1):
        print(f"[{idx}/{len(cases)}] {op} shape={shape} wg={wg_size}", flush=True)
        rec = run_case(script_path, op, shape, wg_size, args)
        print(
            "  -> {status} abs={abs_err} rel={rel_err} p50={p50}".format(
                status=rec.get("status"),
                abs_err=rec.get("max_abs_err", ""),
                rel_err=rec.get("max_rel_err", ""),
                p50=rec.get("latency_p50_ms", ""),
            ),
            flush=True,
        )
        rows.append(rec)

    write_outputs(rows, Path(args.out_dir), args.stem)
    print(json.dumps({"summary": summarize(rows)}, indent=2))


if __name__ == "__main__":
    main()
