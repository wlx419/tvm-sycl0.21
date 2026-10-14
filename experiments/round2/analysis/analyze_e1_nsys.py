#!/usr/bin/env python3
"""E1: decompose the complete-model CUDA/SYCL gap from whole-model Nsight Systems traces.

Inputs (per target, per batch), produced by stage e1 of experiments/run_round2_4090.sh:
  <t>_b<B>_cuda_gpu_trace.csv   kernel start/duration/name (nsys stats)
  <t>_b<B>.json                 probe result (synchronized wall p50 and samples)
  src/<t>_resnet18_b<B>_*.src   generated device source (SYCL kernel-name map)
Workload metadata: analysis/m2_workloads (weights, matched set) via the paired CSV.

Per inference (the last --iters segments separated by >= 10 ms of device idle time):
  launches, sum of kernel time, device span (first kernel start -> last kernel end),
  inter-kernel idle = span - sum(kernel); host-only time = wall p50 - span.
Gap decomposition (SYCL - CUDA): kernel time on matched workloads, on workloads whose
schedule source differs or is DLight on both sides, inter-kernel idle, host-only time.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "results/remote_4090_20260813/paired_resnet18_metaschedule_formal1024/record_analysis/paired_metaschedule_workloads.csv"


def load_workloads():
    rows = list(csv.DictReader(open(WORKLOADS, encoding="utf-8")))
    sets = {}
    for r in rows:
        c, s = bool(r["cuda_best_ms"]), bool(r["sycl_best_ms"])
        sets[r["name"]] = "matched" if c and s else ("cuda_only_ms" if c else ("sycl_only_ms" if s else "neither"))
    return sets


def sycl_name_map(src_dir: Path, tag: str = ""):
    m = {}
    for f in src_dir.glob("*.src"):
        text = f.read_text(errors="replace")
        for fn, h in re.findall(r"// Function: (\S+)\nclass TVM_SYCL_KERNEL_NAME\((h\d+),", text):
            m[h] = fn
    return m


def kernel_workload(name: str, target: str, hmap) -> str:
    if target == "sycl":
        h = re.search(r"(h\d+)_m", name)
        fn = hmap.get(h.group(1), name) if h else name
    else:
        fn = name.split("(")[0]
    fn = re.sub(r"_kernel(_\d+)?$", "", fn)
    return fn


def load_kernels(csv_path: Path):
    ks = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r.get("Name") or r.get("Kernel Name") or ""
            if not name or name.startswith("[CUDA mem"):
                continue
            ks.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name))
    ks.sort()
    return ks


def segments(ks, gap_ns=10_000_000):
    segs, cur = [], []
    for k in ks:
        if cur and k[0] - (cur[-1][0] + cur[-1][1]) > gap_ns:
            segs.append(cur)
            cur = []
        cur.append(k)
    if cur:
        segs.append(cur)
    return segs


def steady_state_inferences(ks, iters):
    """Split the trace into inferences of the timing loop (consecutive, each followed by a sync).

    The run ends with sleep-separated inferences (--nsys-iters); the last of them gives the launch
    sequence of one inference (length L, first kernel K0). The long continuous stretch before them
    holds warm-up and timed inferences back to back; it is cut at occurrences of K0 into chunks of
    L launches, and the last `iters` complete chunks are returned.
    """
    segs = segments(ks)
    ref = segs[-1]
    L, k0 = len(ref), ref[0][2]
    body = max(segs[:-1], key=len) if len(segs) > 1 else ref
    chunks, i = [], 0
    while i + L <= len(body):
        if body[i][2] == k0 and all(body[i + j][2] == ref[j][2] for j in range(L)):
            chunks.append(body[i:i + L])
            i += L
        else:
            i += 1
    return chunks[-iters:], L


def analyze(d: Path, run: str, iters: int, sets):
    target = "sycl" if run.startswith("sycl") else "cuda"
    batch = int(re.search(r"_b(\d+)$", run).group(1))
    ks = load_kernels(d / f"{run}_cuda_gpu_trace.csv")
    hmap = sycl_name_map(d / "src") if target == "sycl" else {}
    chunks, launches = steady_state_inferences(ks, iters)
    probe = json.load(open(d / f"{run}.json"))
    per = []
    for s in chunks:
        by_set, by_wl = defaultdict(float), defaultdict(float)
        for start, dur, kname in s:
            wl = kernel_workload(kname, target, hmap)
            by_wl[wl] += dur / 1e6
            by_set[sets.get(wl, "unmapped")] += dur / 1e6
        span = (s[-1][0] + s[-1][1] - s[0][0]) / 1e6
        ksum = sum(dur for _, dur, _ in s) / 1e6
        per.append({"launches": len(s), "kernel_ms": ksum, "span_ms": span, "idle_ms": span - ksum,
                    "by_set": dict(by_set), "by_workload": dict(by_wl)})
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    out = {"name": run, "target": target, "batch": batch, "iters": len(per), "wall_p50_ms": probe.get("latency_p50_ms"),
           "launches": launches, "kernel_ms": med([p["kernel_ms"] for p in per]),
           "span_ms": med([p["span_ms"] for p in per]), "idle_ms": med([p["idle_ms"] for p in per])}
    out["host_only_ms"] = out["wall_p50_ms"] - out["span_ms"]
    keys = set().union(*[p["by_set"] for p in per]) if per else set()
    out["by_set_ms"] = {k: med([p["by_set"].get(k, 0.0) for p in per]) for k in sorted(keys)}
    wkeys = set().union(*[p["by_workload"] for p in per]) if per else set()
    out["by_workload_ms"] = {k: med([p["by_workload"].get(k, 0.0) for p in per]) for k in sorted(wkeys)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="", help="comma list; default: every *_cuda_gpu_trace.csv")
    a = ap.parse_args()
    d = Path(a.dir)
    sets = load_workloads()
    names = a.names.split(",") if a.names else sorted(
        p.name[: -len("_cuda_gpu_trace.csv")] for p in d.glob("*_cuda_gpu_trace.csv"))
    res = {n: analyze(d, n, a.iters, sets) for n in names if (d / f"{n}_cuda_gpu_trace.csv").exists()}
    L = ["# E1: whole-model gap decomposition (median over traced inferences)\n",
         "| Target | B | wall p50 | launches | sum kernel | device span | inter-kernel idle | host-only | matched | sycl_only_ms | cuda_only_ms | neither |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in res.items():
        bs = r["by_set_ms"]
        L.append(f"| {r['name']} | {r['batch']} | {r['wall_p50_ms']:.3f} | {r['launches']:.0f} | {r['kernel_ms']:.3f} | "
                 f"{r['span_ms']:.3f} | {r['idle_ms']:.3f} | {r['host_only_ms']:.3f} | {bs.get('matched',0):.3f} | "
                 f"{bs.get('sycl_only_ms',0):.3f} | {bs.get('cuda_only_ms',0):.3f} | {bs.get('neither',0):.3f} |")
    pairs = [(f"cuda_b{b}", n) for n in res for b in (1, 16) if n.startswith("sycl") and n.endswith(f"_b{b}")]
    for cname, sname in pairs:
        c, s = res.get(cname), res.get(sname)
        if not (c and s):
            continue
        b = s["batch"]
        gap = s["wall_p50_ms"] - c["wall_p50_ms"]
        parts = {
            "kernel time, matched workloads": s["by_set_ms"].get("matched", 0) - c["by_set_ms"].get("matched", 0),
            "kernel time, other workloads": (s["kernel_ms"] - s["by_set_ms"].get("matched", 0)) - (c["kernel_ms"] - c["by_set_ms"].get("matched", 0)),
            "inter-kernel idle": s["idle_ms"] - c["idle_ms"],
            "host-only (outside device span)": s["host_only_ms"] - c["host_only_ms"],
        }
        L += ["", f"## {sname} - {cname} = {gap:.3f} ms", "| Component | ms | share |", "|---|---:|---:|"]
        for n, v in parts.items():
            L.append(f"| {n} | {v:.3f} | {100*v/gap:.1f}% |")
        L.append(f"| residual | {gap-sum(parts.values()):.3f} | |")
    Path(a.out).write_text("\n".join(L) + "\n\n```json\n" + json.dumps(res, indent=1) + "\n```\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
