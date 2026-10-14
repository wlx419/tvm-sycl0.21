#!/usr/bin/env python3
"""Per-launch comparison of two whole-model traces by launch position (same Relax program on both
backends, so position i is the same operator). Median duration per position over the inferences.

usage: compare_by_order.py --dir <e1 dir> --ref <cuda run> --cmp <sycl run> [--iters 20]
"""
import argparse
import statistics
from pathlib import Path

from analyze_e1_nsys import load_kernels, steady_state_inferences


def per_position(d, run, iters):
    ks = load_kernels(Path(d) / f"{run}_cuda_gpu_trace.csv")
    chunks, n = steady_state_inferences(ks, iters)
    names = [k[2] for k in chunks[0]]
    durs = [statistics.median(c[i][1] for c in chunks) / 1e3 for i in range(n)]  # us
    gaps = [statistics.median(c[i][0] - (c[i - 1][0] + c[i - 1][1]) for c in chunks) / 1e3 if i else 0.0
            for i in range(n)]
    return names, durs, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    rn, rd, rg = per_position(a.dir, a.ref, a.iters)
    cn, cd, cg = per_position(a.dir, a.cmp, a.iters)
    assert len(rn) == len(cn), (len(rn), len(cn))
    print(f"{'#':>2} {'ref kernel':42s} {'ref us':>7} {'cmp us':>7} {'ratio':>6} {'ref gap':>7} {'cmp gap':>7}")
    for i in range(len(rn)):
        print(f"{i:2d} {rn[i][:42]:42s} {rd[i]:7.2f} {cd[i]:7.2f} {cd[i] / rd[i]:6.2f} {rg[i]:7.2f} {cg[i]:7.2f}")
    print(f"sum kernels: ref {sum(rd):.2f} us, cmp {sum(cd):.2f} us; sum gaps: ref {sum(rg):.2f}, cmp {sum(cg):.2f}")


if __name__ == "__main__":
    main()
