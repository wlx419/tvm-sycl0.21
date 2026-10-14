#!/usr/bin/env python3
"""Tabulate DCU sweeps (p50 ms per model and batch) across configurations.

usage: tabulate_dcu.py <dir with results/ and results_d1/> [--out md]
Columns: published 2026-09-04 SYCL and ROCm (from results/remote_dcu_20260904), then every
compact_*/resnet_* JSON found under results/ (name) and results_d1/ (d1:name).
"""
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["mlp", "cnn", "transformer", "vit", "resnet18", "resnet34", "resnet50"]


def load(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {(r.get("model"), r.get("batch")): (r.get("status"), r.get("latency_p50_ms"))
            for r in d.get("results", [])}


def main():
    base = Path(sys.argv[1])
    out = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else None
    sep = ROOT / "results/remote_dcu_20260904/results"
    cfgs = {"pub_0904": {**load(sep / "compact_sycl.json"), **load(sep / "resnet_sycl.json")},
            "rocm_0904": {**load(sep / "compact_rocm.json"), **load(sep / "resnet_rocm.json")}}
    for sub, prefix in (("results", ""), ("results_d1", "d1:")):
        for f in sorted(glob.glob(str(base / sub / "*.json"))):
            stem = Path(f).stem
            name = prefix + stem.split("_", 1)[1]
            cfgs.setdefault(name, {}).update(load(f))
    keys = sorted({k for v in cfgs.values() for k in v if k[0] in ORDER},
                  key=lambda k: (ORDER.index(k[0]), k[1]))
    cols = list(cfgs)
    lines = ["| model | B | " + " | ".join(cols) + " |", "|---|---:|" + "---:|" * len(cols)]
    for k in keys:
        cells = []
        for c in cols:
            s, p = cfgs[c].get(k, (None, None))
            cells.append("-" if p is None else f"{p:.3f}" + ("" if s == "pass" else "*"))
        lines.append(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("p50 in ms; * = not passing the tolerance check (ROCm Transformer/ViT produce nonfinite output).")
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n")


if __name__ == "__main__":
    main()
