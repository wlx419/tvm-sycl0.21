#!/usr/bin/env python3
"""Summarize a timing window: per configuration, the median of the per-process p50 values, the
pooled IQR of the raw per-iteration samples, and bootstrap 95% CIs for ratios against a reference
configuration (resampling iterations within each process, then taking the median of process p50s).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

import numpy as np


def load(dirname):
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(dirname, "*_run*.json"))):
        name = re.sub(r"_run\d+\.json$", "", os.path.basename(f))
        d = json.load(open(f))
        groups[name].append(d)
    return groups


def stat(runs):
    p50s = [r["latency_p50_ms"] for r in runs if r.get("latency_samples_ms")]
    samples = np.concatenate([np.asarray(r["latency_samples_ms"]) for r in runs if r.get("latency_samples_ms")]) \
        if p50s else np.array([])
    status = sorted({r.get("status") for r in runs})
    return {
        "n_proc": len(p50s), "status": status,
        "p50_median": statistics.median(p50s) if p50s else float("nan"),
        "p50_min": min(p50s) if p50s else float("nan"), "p50_max": max(p50s) if p50s else float("nan"),
        "q25": float(np.percentile(samples, 25)) if samples.size else float("nan"),
        "q75": float(np.percentile(samples, 75)) if samples.size else float("nan"),
        "compile_s": statistics.median([r["compile_s"] for r in runs if "compile_s" in r]) if runs else float("nan"),
        "max_abs_err": max((r.get("max_abs_err") or 0) for r in runs) if runs else float("nan"),
    }


def boot_median_p50(runs, rng, n=2000):
    arrs = [np.asarray(r["latency_samples_ms"]) for r in runs if r.get("latency_samples_ms")]
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median([np.median(rng.choice(a, size=a.size, replace=True)) for a in arrs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pairs", default="", help="comma list of name:reference for ratio CIs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = load(a.dir)
    rng = np.random.default_rng(20261012)
    L = ["| Configuration | procs | status | p50 median (ms) | p50 range over procs | IQR of samples | compile s | max abs err |",
         "|---|---:|---|---:|---|---|---:|---:|"]
    stats = {}
    for name in sorted(g):
        s = stat(g[name]); stats[name] = s
        L.append(f"| {name} | {s['n_proc']} | {','.join(s['status'])} | {s['p50_median']:.4f} | "
                 f"{s['p50_min']:.4f}-{s['p50_max']:.4f} | {s['q25']:.4f}-{s['q75']:.4f} | {s['compile_s']:.1f} | {s['max_abs_err']:.2e} |")
    if a.pairs:
        L += ["", "| Ratio | point | bootstrap 95% CI |", "|---|---:|---|"]
        boots = {}
        for pair in a.pairs.split(","):
            num, den = pair.split(":")
            if num not in g or den not in g:
                continue
            for k in (num, den):
                if k not in boots:
                    boots[k] = boot_median_p50(g[k], rng)
            ratio = boots[num] / boots[den]
            point = stats[num]["p50_median"] / stats[den]["p50_median"]
            lo, hi = np.percentile(ratio, [2.5, 97.5])
            L.append(f"| {num} / {den} | {point:.3f} | {lo:.3f}-{hi:.3f} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
