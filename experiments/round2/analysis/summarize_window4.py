#!/usr/bin/env python3
"""Window 4 tables: final-code batch sweep (SYCL, CUDA, OpenCL; eager and graph) and E11.

usage: summarize_window4.py <window4 dir> [--out md]
p50 = median over processes of the per-process p50 (ms); ratio = SYCL / CUDA (eager vs eager,
graph vs graph); worst = largest max-abs error over processes and modes.
"""
import csv
import glob
import json
import os
import statistics
import sys

MODELS = ["resnet18", "resnet34", "resnet50", "mlp", "cnn", "transformer", "vit"]


def runs(w, name):
    out = []
    for f in sorted(glob.glob(os.path.join(w, f"{name}_run*.json"))):
        d = json.load(open(f))
        out.append(d)
    return out


def med(rs):
    v = [r["latency_p50_ms"] for r in rs if r.get("status") == "pass" and r.get("latency_p50_ms") is not None]
    return statistics.median(v) if v else None


def worst(*groups):
    v = [r.get("max_abs_err") for g in groups for r in g if r.get("max_abs_err") is not None]
    return max(v) if v else None


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    w = sys.argv[1]
    out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else None
    L = ["## Batch sweep, default pipeline (DLight), RTX 4090, final code", "",
         "| Model | B | CUDA | SYCL | SYCL/CUDA | CUDA graph | SYCL graph | graph ratio | OpenCL | OpenCL/CUDA | worst abs err | procs |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        for b in (1, 8, 16):
            c, s = runs(w, f"sweep_{m}_b{b}_cuda"), runs(w, f"sweep_{m}_b{b}_sycl")
            cg, sg = runs(w, f"sweep_{m}_b{b}_cuda_graph"), runs(w, f"sweep_{m}_b{b}_sycl_graph")
            o = runs(w, f"sweep_{m}_b{b}_opencl")
            mc, ms_, mcg, msg, mo = med(c), med(s), med(cg), med(sg), med(o)
            r = ms_ / mc if mc and ms_ else None
            rg = msg / mcg if mcg and msg else None
            ro = mo / mc if mc and mo else None
            nd = 3 if m.startswith("resnet") else 4
            L.append(f"| {m} | {b} | {fmt(mc, nd)} | {fmt(ms_, nd)} | {fmt(r, 2)} | {fmt(mcg, nd)} | {fmt(msg, nd)} | "
                     f"{fmt(rg, 2)} | {fmt(mo, nd)} | {fmt(ro, 2)} | {worst(c, s, cg, sg, o) or '-'} | "
                     f"{len(c)}/{len(s)}/{len(o)} |")
    L += ["", "## E11: compute-bound shapes (p50 ms; median of 3 processes/runs)", "",
          "| Op | oneDNN (cuBLAS/cuDNN) | TVM-SYCL (DLight) | TVM-CUDA (DLight) | SYCL/CUDA | SYCL/oneDNN | max rel (masked) |",
          "|---|---:|---:|---:|---:|---:|---|"]
    one = {}
    for f in sorted(glob.glob(os.path.join(w, "e11", "onednn_run*.csv"))):
        for row in csv.DictReader(open(f)):
            one.setdefault(row["op"], []).append(float(row["p50_ms"]))
    for op, oname in (("matmul", "matmul"), ("conv2d", "conv2d_nchw")):
        s = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(w, "e11", f"sycl_{op}_run*.json")))]
        c = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(w, "e11", f"cuda_{op}_run*.json")))]
        ms_, mc = med(s), med(c)
        mo = statistics.median(one[oname]) if one.get(oname) else None
        rel = max([r.get("max_rel_err_masked") or 0 for r in s + c] or [0])
        L.append(f"| {op} | {fmt(mo)} | {fmt(ms_)} | {fmt(mc)} | {fmt(ms_ / mc if ms_ and mc else None, 2)} | "
                 f"{fmt(ms_ / mo if ms_ and mo else None, 2)} | {rel:.2e} |")
    text = "\n".join(L)
    print(text)
    if out:
        open(out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
