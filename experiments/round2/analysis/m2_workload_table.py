#!/usr/bin/env python3
"""The 19 ResNet-18 workloads of the published paired MetaSchedule configuration
with weight (calls per inference), schedule source on each side, and kernel time per inference from
the whole-model Nsight traces of the published configurations (window 1, E1).

usage: m2_workload_table.py <window1 e1_summary.md> (JSON block at the end)
"""
import json
import os
import sys

# (workload, weight, CUDA source, SYCL source); MS = MetaSchedule record, DLight = fallback
ROWS = [
    ("fused_conv2d_add_relu", "7x7 conv", 1, "MS", "MS"),
    ("max_pool2d", "pool", 1, "MS", "MS"),
    ("fused_conv2d1_add1_relu1", "3x3 conv", 2, "MS", "MS"),
    ("fused_conv2d1_add1_add2_relu1", "3x3 conv", 2, "MS", "MS"),
    ("fused_conv2d3_add3_relu2", "3x3 conv", 1, "MS", "MS"),
    ("fused_conv2d4_add3", "1x1 conv", 1, "MS", "MS"),
    ("fused_conv2d7_add5", "1x1 conv", 1, "MS", "MS"),
    ("fused_conv2d8_add7_relu4", "3x3 conv", 1, "MS", "MS"),
    ("fused_conv2d9_add7_relu4", "3x3 conv", 1, "MS", "MS"),
    ("fused_conv2d9_add7_add8_relu4", "3x3 conv", 2, "MS", "MS"),
    ("fused_conv2d10_add7", "1x1 conv", 1, "MS", "MS"),
    ("adaptive_avg_pool2d", "pool", 1, "MS", "MS"),
    ("fused_conv2d2_add3_relu2", "3x3 conv", 1, "MS", "DLight"),
    ("fused_conv2d6_add5_relu3", "3x3 conv", 1, "MS", "DLight"),
    ("fused_conv2d6_add5_add6_relu3", "3x3 conv", 2, "MS", "DLight"),
    ("fused_conv2d3_add3_add4_relu2", "3x3 conv", 2, "DLight", "MS"),
    ("fused_conv2d5_add5_relu3", "3x3 conv", 1, "DLight", "MS"),
    ("fused_matmul_add9", "dense", 1, "DLight", "MS"),
    ("reshape4", "reshape", 1, "DLight", "DLight"),
]


def main():
    text = open(sys.argv[1], encoding="utf-8").read()
    data = json.loads(text[text.index("```json") + 7:text.rindex("```")])
    c, s = data["cuda_b1"], data["sycl_b1"]
    cw, sw = c["by_workload_ms"], s["by_workload_ms"]
    print("| Workload | Kind | Calls | CUDA schedule | SYCL schedule | CUDA (ms) | SYCL (ms) | SYCL - CUDA |")
    print("|---|---|---:|---|---|---:|---:|---:|")
    sets = {}
    for name, kind, w, cs, ss in ROWS:
        a, b = cw.get(name), sw.get(name)
        key = "both MetaSchedule" if (cs, ss) == ("MS", "MS") else \
              "SYCL fell back to DLight" if ss == "DLight" and cs == "MS" else \
              "CUDA fell back to DLight" if cs == "DLight" and ss == "MS" else "no kernel"
        t = sets.setdefault(key, [0, 0.0, 0.0])
        t[0] += w
        if a is None:
            print(f"| `{name}` | {kind} | {w} | {cs} | {ss} | - | - | - |")
            continue
        t[1] += a
        t[2] += b
        print(f"| `{name}` | {kind} | {w} | {cs} | {ss} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    print()
    print("| Set | Calls | CUDA (ms) | SYCL (ms) | SYCL - CUDA |")
    print("|---|---:|---:|---:|---:|")
    for k in ("both MetaSchedule", "SYCL fell back to DLight", "CUDA fell back to DLight", "no kernel"):
        n, a, b = sets[k]
        print(f"| {k} | {n} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    print(f"| kernel time, all | 24 | {c['kernel_ms']:.3f} | {s['kernel_ms']:.3f} | {s['kernel_ms'] - c['kernel_ms']:+.3f} |")
    print(f"| idle between kernels | | {c['idle_ms']:.3f} | {s['idle_ms']:.3f} | {s['idle_ms'] - c['idle_ms']:+.3f} |")
    print(f"| host time outside the device span | | {c['host_only_ms']:.3f} | {s['host_only_ms']:.3f} | {s['host_only_ms'] - c['host_only_ms']:+.3f} |")
    print(f"| wall time per traced inference | | {c['wall_p50_ms']:.3f} | {s['wall_p50_ms']:.3f} | {s['wall_p50_ms'] - c['wall_p50_ms']:+.3f} |")


if __name__ == "__main__":
    main()
