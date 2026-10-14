#!/usr/bin/env python3
"""Per-workload kernel time of several whole-model Nsight Systems traces (median per inference).

usage: compare_kernels.py --dir <e1 dir> --runs cudadb_cuda_b1,cudadb_sycl_c8_b1,... --out <md>
A run is SYCL when its name contains "sycl"; its kernel-name map is read from <dir>/src_<run>/.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_e1_nsys import kernel_workload, load_kernels, steady_state_inferences, sycl_name_map


def per_workload(d: Path, run: str, iters: int):
    target = "sycl" if "sycl" in run else "cuda"
    ks = load_kernels(d / f"{run}_cuda_gpu_trace.csv")
    hmap = sycl_name_map(d / f"src_{run}") if target == "sycl" else {}
    chunks, launches = steady_state_inferences(ks, iters)
    rows = []
    for s in chunks:
        by = defaultdict(float)
        for _, dur, name in s:
            by[kernel_workload(name, target, hmap)] += dur / 1e6
        span = (s[-1][0] + s[-1][1] - s[0][0]) / 1e6
        rows.append((by, span, sum(x[1] for x in s) / 1e6))
    keys = set().union(*[r[0] for r in rows])
    med = statistics.median
    return ({k: med([r[0].get(k, 0.0) for r in rows]) for k in keys},
            med([r[1] for r in rows]), med([r[2] for r in rows]), launches, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    runs = a.runs.split(",")
    res = {r: per_workload(d, r, a.iters) for r in runs}
    ref = runs[0]
    wls = sorted(res[ref][0], key=lambda k: -res[ref][0][k])
    for r in runs[1:]:
        wls += [k for k in sorted(res[r][0]) if k not in wls]
    head = "| Workload | " + " | ".join(runs) + " | " + " | ".join(f"{r}/{ref}" for r in runs[1:]) + " |"
    L = ["# Per-workload kernel time, ms per inference (median)\n", head,
         "|---|" + "---:|" * (2 * len(runs) - 1)]
    for w in wls:
        v = [res[r][0].get(w, 0.0) for r in runs]
        ratios = [f"{x / v[0]:.2f}" if v[0] > 0 else "-" for x in v[1:]]
        L.append(f"| {w} | " + " | ".join(f"{x:.4f}" for x in v) + " | " + " | ".join(ratios) + " |")
    for label, idx in (("sum of kernels", 2), ("device span", 1)):
        v = [res[r][idx] for r in runs]
        L.append(f"| **{label}** | " + " | ".join(f"{x:.4f}" for x in v) + " | "
                 + " | ".join(f"{x / v[0]:.2f}" for x in v[1:]) + " |")
    L.append("")
    L.append("launches per inference: " + ", ".join(f"{r}={res[r][3]}" for r in runs)
             + "; inferences: " + ", ".join(f"{r}={res[r][4]}" for r in runs))
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
