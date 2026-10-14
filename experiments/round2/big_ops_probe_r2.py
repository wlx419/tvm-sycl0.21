#!/usr/bin/env python3
"""E11, TVM side: compute-bound matmul and convolution through the default (DLight) pipeline.

Same shapes, input generator, sampled float64 check and timing method (wall time of one call plus
device sync, 10 warm-ups, 30 measurements) as onednn_large_shapes.cpp.
  matmul 4096x4096x4096; conv2d NCHW 16x64x56x56, 64 filters 3x3, stride 1, padding 1
"""
import argparse
import json
import sys  # noqa: F401
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R

SAMPLES = 4096


def default_pipeline(target):
    """As in resnet_relax_probe_r2 (not imported: that module imports torch)."""
    if target.kind.name == "opencl" and "adreno" not in target.keys:
        return relax.backend.gpu_generic.get_default_pipeline(target)
    return relax.get_default_pipeline(target)


def filled(n, scale):
    # size_t arithmetic of the C++ generator (wraps modulo 2^64, as numpy uint64 does)
    i = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        v = (i * np.uint64(1315423911) + np.uint64(2654435761)) % np.uint64(257)
    return (np.float32(scale) * (v.astype(np.int64) - 128).astype(np.float32) / np.float32(257.0)).astype(np.float32)


def sample_idx(total):
    i = np.arange(SAMPLES, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (i * np.uint64(2654435761)) % np.uint64(total)


@tvm.script.ir_module
class MatmulMod:
    @R.function
    def main(a: R.Tensor((4096, 4096), "float32"), b: R.Tensor((4096, 4096), "float32")):
        with R.dataflow():
            c = R.matmul(a, b)
            R.output(c)
        return c


@tvm.script.ir_module
class ConvMod:
    @R.function
    def main(x: R.Tensor((16, 64, 56, 56), "float32"), w: R.Tensor((64, 64, 3, 3), "float32")):
        with R.dataflow():
            y = R.nn.conv2d(x, w, strides=1, padding=1)
            R.output(y)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["matmul", "conv2d"], required=True)
    ap.add_argument("--target", default="sycl")
    ap.add_argument("--target-string", default="")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    dev = tvm.device("oneapi" if a.target == "sycl" else a.target, 0)
    target = tvm.target.Target(a.target_string) if a.target_string else tvm.target.Target.from_device(dev)
    if a.op == "matmul":
        mod, shapes = MatmulMod, [(4096, 4096), (4096, 4096)]
        ha, hb = filled(4096 * 4096, 1.0).reshape(shapes[0]), filled(4096 * 4096, 0.5).reshape(shapes[1])
    else:
        mod, shapes = ConvMod, [(16, 64, 56, 56), (64, 64, 3, 3)]
        ha, hb = filled(16 * 64 * 56 * 56, 1.0).reshape(shapes[0]), filled(64 * 64 * 9, 0.25).reshape(shapes[1])
    row = {"op": a.op, "target": a.target, "resolved_target": str(target)}
    try:
        t0 = time.perf_counter()
        ex = tvm.compile(mod, target=target, relax_pipeline=default_pipeline(target))
        row["compile_s"] = time.perf_counter() - t0
        vm = relax.VirtualMachine(ex, dev)
        da, db = tvm.nd.array(ha, dev), tvm.nd.array(hb, dev)
        fn = lambda: vm["main"](da, db)  # noqa: E731
        out = fn().numpy()
        idx = sample_idx(out.size)
        flat = out.reshape(-1)
        a64, b64 = ha.astype(np.float64), hb.astype(np.float64)
        refs = []
        for k in idx:
            k = int(k)
            if a.op == "matmul":
                m, n = divmod(k, 4096)
                refs.append(float(a64[m] @ b64[:, n]))
            else:
                bb, rem = divmod(k, 64 * 56 * 56)
                f, rem = divmod(rem, 56 * 56)
                oh, ow = divmod(rem, 56)
                xp = np.pad(a64[bb], ((0, 0), (1, 1), (1, 1)))
                refs.append(float(np.sum(xp[:, oh:oh + 3, ow:ow + 3] * b64[f])))
        refs = np.array(refs)
        got = flat[idx.astype(np.int64)].astype(np.float64)
        diff = np.abs(got - refs)
        mask = np.abs(refs) >= 1e-3 * np.abs(refs).max()
        row.update(max_abs_err=float(diff.max()), max_rel_err_masked=float((diff[mask] / np.abs(refs[mask])).max()))
        for _ in range(10):
            fn()
        dev.sync()
        lat = []
        for _ in range(30):
            t0 = time.perf_counter_ns()
            fn()
            dev.sync()
            lat.append((time.perf_counter_ns() - t0) / 1e6)
        lat.sort()
        row.update(status="pass" if row["max_rel_err_masked"] < 1e-3 else "check", latency_p50_ms=lat[len(lat) // 2],
                   latency_p90_ms=lat[int(0.9 * (len(lat) - 1))], latency_samples_ms=lat)
    except Exception as exc:  # report, never hide
        import traceback
        row.update(status="runtime_error", error=repr(exc), traceback=traceback.format_exc())
    Path(a.out_json).write_text(json.dumps(row, indent=1))
    print(a.target, a.op, row.get("status"), "p50=%s" % row.get("latency_p50_ms"),
          "abs=%s rel=%s" % (row.get("max_abs_err"), row.get("max_rel_err_masked")), row.get("error", "")[:200])


if __name__ == "__main__":
    main()
