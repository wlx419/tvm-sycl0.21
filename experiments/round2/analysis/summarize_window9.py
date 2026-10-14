#!/usr/bin/env python3
"""Window 9: the same r2 sources built against DPC++ 646ab08 (r2) and DPC++ v7.1.1 (v711).

usage: summarize_window9.py <window9 dir>
Prints (markdown) the launch and graph-replay microbenchmarks (median over the processes of each
toolchain), and for every model configuration the median of three per-process p50s for r2, v711,
v711 with L1 off (TVM_SYCL_DIRECT_SUBMIT=0) where measured, and TVM-CUDA, with v711/r2 and
SYCL/CUDA ratios. Runs that did not pass are counted and excluded.
"""
import collections
import glob
import json
import os
import re
import statistics
import sys


def micro(log):
    launch, graph = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"launch (\w+) run=\d+ (\S+)\s+submit_us=([\d.]+)", line)
        if m:
            launch[(m.group(1), m.group(2))].append(float(m.group(3)))
            continue
        m = re.match(r"graph (\w+) run=\d+ growth K=(\d+) replays (\d+-\d+) submit_us=([\d.]+)", line)
        if m:
            graph[(m.group(1), f"growth K={m.group(2)} replays {m.group(3)}")].append(float(m.group(4)))
            continue
        m = re.match(r"graph (\w+) run=\d+ refin K=(\d+) every=(\d+) submit_us=([\d.]+)", line)
        if m:
            graph[(m.group(1), f"re-finalize every {m.group(3)} (K={m.group(2)})")].append(float(m.group(4)))
    return launch, graph


def med(d, name, tag):
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f"{name}_{tag}_run*.json")))]
    ok = [r["latency_p50_ms"] for r in rs if r.get("status") == "pass"]
    return (statistics.median(ok) if ok else None), len(ok), len(rs)


def fmt(v):
    return "" if v is None else (f"{v:.4f}" if v < 1 else f"{v:.3f}")


def main():
    d = sys.argv[1]
    launch, graph = micro(os.path.join(d, "window.log"))
    print("## Host cost per launch / graph replay (us, median of processes)\n")
    print("| Benchmark | 646ab08 | v7.1.1 |\n|---|---:|---:|")
    for key in sorted({k for _, k in launch}, key=lambda k: (k.count("+"), k)):
        a, b = launch.get(("r2", key)), launch.get(("v711", key))
        print(f"| launch {key} | {statistics.median(a) if a else '':} | {statistics.median(b) if b else ''} |")
    for key in sorted({k for _, k in graph}):
        a, b = graph.get(("r2", key)), graph.get(("v711", key))
        print(f"| graph {key} | {statistics.median(a) if a else ''} | {statistics.median(b) if b else ''} |")
    print("\n## Models (p50 ms, median of three processes)\n")
    print("| Configuration | 646ab08 | v7.1.1 | v7.1.1 L1 off | CUDA | v711/r2 | r2/CUDA | v711/CUDA | pass |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    names = sorted({re.sub(r"_(r2|v711|cuda)_run\d+\.json$", "", os.path.relpath(f, d)).replace("\\", "/")
                    for f in glob.glob(os.path.join(d, "*_run*.json")) + glob.glob(os.path.join(d, "e11", "*_run*.json"))})
    base = [n for n in names if not n.endswith(("_l1off", "_r2off"))]
    npass = ntot = 0
    for n in base:
        r2, v7, cu = med(d, n, "r2"), med(d, n, "v711"), med(d, n, "cuda")
        off = med(d, n + "_l1off", "v711")
        for x in (r2, v7, cu, off):
            npass += x[1]
            ntot += x[2]
        ratio = lambda a, b: f"{a / b:.2f}x" if a and b else ""
        print(f"| {n} | {fmt(r2[0])} | {fmt(v7[0])} | {fmt(off[0])} | {fmt(cu[0])} | {ratio(v7[0], r2[0])} | "
              f"{ratio(r2[0], cu[0])} | {ratio(v7[0], cu[0])} | {r2[1]}/{v7[1]}/{off[1]}/{cu[1]} |")
    for n in names:
        if n.endswith("_r2off"):
            x = med(d, n, "v711")
            npass += x[1]
            ntot += x[2]
            print(f"| {n} (v7.1.1) | | {fmt(x[0])} | | | | | | {x[1]} |")
    print(f"\npassing runs: {npass}/{ntot}")


if __name__ == "__main__":
    main()
