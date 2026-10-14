#!/usr/bin/env python3
"""Median p50 per configuration for window JSON files named <config>_<variant>[_runN].json.

usage: summarize_variants.py <dir> <variant> [<variant> ...]   (first variant is the reference)
"""
import collections
import glob
import json
import os
import re
import statistics
import sys


def p50(j):
    for k in ("latency_p50_ms", "p50_ms", "p50"):
        if k in j:
            return float(j[k])
    return None


def main():
    d, variants = sys.argv[1], sys.argv[2:]
    runs = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = re.sub(r"(_run\d+)?\.json$", "", os.path.basename(f))
        v = p50(json.load(open(f)))
        if v is not None:
            runs[name].append(v)
    pat = re.compile(r"^(.*)_(%s)$" % "|".join(map(re.escape, variants)))
    configs = sorted({pat.match(k).group(1) for k in runs if pat.match(k)})
    ref = variants[0]
    print(f"{'config':24s} " + " ".join(f"{v:>9s}" for v in variants) + "  " +
          " ".join(f"{v}/{ref}" for v in variants[1:]))
    for c in configs:
        m = {v: statistics.median(runs[f"{c}_{v}"]) for v in variants if runs.get(f"{c}_{v}")}
        cells = " ".join(f"{m[v]:9.4f}" if v in m else f"{'-':>9s}" for v in variants)
        ratios = " ".join(f"{m[v] / m[ref]:.3f}" if v in m and ref in m else "-" for v in variants[1:])
        n = "/".join(str(len(runs.get(f"{c}_{v}", []))) for v in variants)
        print(f"{c:24s} {cells}  {ratios}  n={n}")


if __name__ == "__main__":
    main()
