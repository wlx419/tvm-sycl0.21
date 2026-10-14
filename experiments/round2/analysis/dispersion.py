#!/usr/bin/env python3
"""Dispersion of the timing windows: for every configuration with several processes, the largest
deviation of a per-process p50 from the median of the per-process p50s, and the within-process IQR
of the samples relative to the p50. Grouped by model class and mode.

usage: dispersion.py <window dir> [<window dir> ...]
"""
import collections
import glob
import json
import os
import re
import statistics
import sys


def group_of(name):
    if "long_" in name:
        return None
    graph = "graph" in name
    if re.search(r"resnet|rn18|rn50|rn34", name):
        return "ResNet " + ("graphs" if graph else "eager")
    if re.search(r"mlp|cnn|transformer|vit", name):
        return "compact " + ("graphs" if graph else "eager")
    return None


def main():
    configs = collections.defaultdict(list)
    for d in sys.argv[1:]:
        for f in glob.glob(os.path.join(d, "*_run*.json")):
            j = json.load(open(f))
            if j.get("status") != "pass" or "latency_p50_ms" not in j:
                continue
            name = os.path.basename(d.rstrip("/")) + "/" + re.sub(r"_run\d+\.json$", "", os.path.basename(f))
            configs[name].append(j)
    rows = collections.defaultdict(list)
    for name, runs in configs.items():
        g = group_of(name)
        if g is None or len(runs) < 3:
            continue
        p50 = [r["latency_p50_ms"] for r in runs]
        med = statistics.median(p50)
        dev = max(abs(v - med) for v in p50) / med
        iqr = []
        for r in runs:
            s = sorted(r.get("latency_samples_ms") or [])
            if len(s) >= 8:
                q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
                iqr.append((q3 - q1) / r["latency_p50_ms"])
        rows[g].append((dev, statistics.median(iqr) if iqr else float("nan"), name))
    for g in sorted(rows):
        r = rows[g]
        devs = sorted(x[0] for x in r)
        iqrs = sorted(x[1] for x in r)
        worst = max(r)
        print(f"{g:16s} n={len(r):3d}  process dev: median {100 * statistics.median(devs):.2f}% "
              f"p90 {100 * devs[int(0.9 * (len(devs) - 1))]:.2f}% max {100 * devs[-1]:.2f}% ({worst[2]})  "
              f"within-process IQR/p50: median {100 * statistics.median(iqrs):.2f}% max {100 * iqrs[-1]:.2f}%")


if __name__ == "__main__":
    main()
