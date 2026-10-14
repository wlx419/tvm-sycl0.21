#!/usr/bin/env python3
"""ResNet-18 B1 ablation on the RTX 4090 from the timing windows (median of per-process p50).

usage: ablation_4090.py <round2 results root on the server> [--out md]
Each step names the window and configuration it comes from; steps measured in different windows
are linked through a configuration repeated in both (shown in the 'same-window reference' column).
"""
import glob
import json
import os
import statistics
import sys

STEPS = [
    # label, window, config, same-window reference (window, config) or None
    ("Published build, published SYCL database", "window1", "rn18_published_sycl", None),
    ("Round-2 build with published behaviour (per-launch wait, warp no-op, zero-fill, no bounds)",
     "window1", "rn18_sycl_a0_sync_legacy", None),
    ("+ asynchronous dispatch (A1)", "window1", "rn18_sycl_a1_async_legacy", None),
    ("+ barrier-only allreduce (C1)", "window1", "rn18_sycl_a2_async_fixed", None),
    ("+ group-local memory without zero-fill (C9)", "window1b", "rn18_sycldb_sycl_c9",
     ("window1b", "rn18_sycldb_sycl_a2")),
    ("+ launch bounds (C10)", "window1b", "rn18_sycldb_sycl_c10", ("window1b", "rn18_sycldb_sycl_a2")),
    ("+ CUDA's DLight configurations for the fallback workloads (D1)", "window4", "rn18_sycldb_sycl_d1", None),
    ("+ command-graph replay (A3)", "window4", "rn18_sycldb_sycl_d1_graph", None),
    ("Retuned with fault-isolated runner and synchronizing timer (T1, T2; P1, C8)", "window3",
     "rn18_sycl_iso_eager", None),
    ("  ... with command-graph replay", "window3", "rn18_sycl_iso_graph", None),
]
IDENTICAL = [
    ("CUDA's schedules on SYCL: inherited code generation (C8 only)", "window1b", "rn18_cudadb_sycl_c8"),
    ("  + no zero-fill (C9)", "window1b", "rn18_cudadb_sycl_c9"),
    ("  + launch bounds (C10)", "window1b", "rn18_cudadb_sycl_c10"),
    ("  + DLight configurations for the 4 fallback workloads (D1)", "window3", "rn18_cudadb_sycl_d1"),
    ("CUDA with the same schedules", "window1b", "rn18_cudadb_cuda"),
]
CUDA = [
    ("CUDA, published database (published build)", "window1", "rn18_published_cuda"),
    ("CUDA, published database (round-2 build)", "window1", "rn18_cuda_eager"),
    ("CUDA, published database, graph", "window1", "rn18_cuda_graph"),
    ("CUDA, retuned (same seed, runner policy)", "window3", "rn18_cuda_iso_eager"),
    ("CUDA, retuned, graph", "window3", "rn18_cuda_iso_graph"),
]


def p50(root, window, name):
    vals = []
    for f in sorted(glob.glob(os.path.join(root, window, f"{name}_run*.json"))):
        d = json.load(open(f))
        if d.get("status") == "pass" and d.get("latency_p50_ms") is not None:
            vals.append(d["latency_p50_ms"])
    return (statistics.median(vals), len(vals)) if vals else (None, 0)


def fmt(v):
    return "-" if v is None else f"{v:.3f}"


def main():
    root = sys.argv[1]
    out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else None
    L = ["## ResNet-18 B1 ablation (RTX 4090, ms, median of per-process p50)", "",
         "| Step | p50 | procs | source | same-window reference |", "|---|---:|---:|---|---|"]
    for label, w, n, ref in STEPS:
        v, k = p50(root, w, n)
        r = ""
        if ref:
            rv, _ = p50(root, *ref)
            r = f"{ref[1]} = {fmt(rv)}"
        L.append(f"| {label} | {fmt(v)} | {k} | {w}/{n} | {r} |")
    L += ["", "## Identical schedules (CUDA's published database)", "",
          "| Configuration | p50 | procs | source |", "|---|---:|---:|---|"]
    for label, w, n in IDENTICAL:
        v, k = p50(root, w, n)
        L.append(f"| {label} | {fmt(v)} | {k} | {w}/{n} |")
    L += ["", "## CUDA references", "", "| Configuration | p50 | procs | source |", "|---|---:|---:|---|"]
    for label, w, n in CUDA:
        v, k = p50(root, w, n)
        L.append(f"| {label} | {fmt(v)} | {k} | {w}/{n} |")
    text = "\n".join(L)
    print(text)
    if out:
        open(out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
