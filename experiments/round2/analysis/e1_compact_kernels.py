#!/usr/bin/env python3
"""Per-inference kernel time of the compact models from Nsight Systems traces (window e1 runs).

usage: e1_compact_kernels.py <window dir> [names...]
For <window dir>/e1/<name>_cuda_gpu_trace.csv (name = <model>_<target>[_graph]) the trace is split into
inferences at device gaps of >= 10 ms (compact_probe_r2.py --nsys-iters sleeps 30 ms between them);
over the last 20 inferences it reports the median launch count, kernel time and device span. The
device-idle share is 1 - kernel time / untraced p50, the p50 being the B1 sweep entry of the same
window (sweep_<model>_b1_<target>[_graph]_run*.json, median of the per-process p50s): the traced
wall time is inflated by the tracer's per-launch cost and is not used.
"""
import csv
import glob
import json
import os
import statistics
import sys


def per_inference(path, iters=20, gap_ns=10_000_000):
    rows = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"])) for r in csv.DictReader(open(path))
                  if not r["Name"].startswith("[CUDA mem"))
    segs = [[rows[0]]]
    for s, d in rows[1:]:
        ps, pd = segs[-1][-1]
        if s - (ps + pd) >= gap_ns:
            segs.append([])
        segs[-1].append((s, d))
    segs = segs[-iters:]
    kern = statistics.median(sum(d for _, d in g) / 1e3 for g in segs)
    span = statistics.median((g[-1][0] + g[-1][1] - g[0][0]) / 1e3 for g in segs)
    return statistics.median(len(g) for g in segs), kern, span


def untraced_p50_us(wdir, name):
    model, rest = name.split("_", 1)
    files = glob.glob(os.path.join(wdir, f"sweep_{model}_b1_{rest}_run*.json"))
    v = [json.load(open(f))["latency_p50_ms"] for f in files]
    return statistics.median(v) * 1e3 if v else float("nan")


def main():
    wdir = sys.argv[1]
    names = sys.argv[2:] or [f"{m}_{t}{g}" for m in ("transformer", "vit") for g in ("", "_graph")
                             for t in ("cuda", "sycl")]
    for name in names:
        n, kern, span = per_inference(os.path.join(wdir, "e1", name + "_cuda_gpu_trace.csv"))
        p50 = untraced_p50_us(wdir, name)
        print(f"{name:24s} launches={n:3.0f} kernel_us={kern:6.1f} traced_span_us={span:7.1f} "
              f"p50_us={p50:6.1f} idle={1 - kern / p50:5.1%}")


if __name__ == "__main__":
    main()
