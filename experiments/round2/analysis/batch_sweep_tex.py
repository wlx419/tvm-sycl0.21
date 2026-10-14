#!/usr/bin/env python3
"""LaTeX rows of the seven-model batch-sweep table from window JSON files.

usage: batch_sweep_tex.py <eager dir> <graph dir>
Eager columns come from <eager dir>/sweep_<model>_b<B>_{cuda,sycl}_run*.json, graph columns from
<graph dir>/sweep_<model>_b<B>_{cuda,sycl}_graph_run*.json; p50 = median over processes of the
per-process p50. Prints the rows, the ratio ranges, and the number of passing runs.
"""
import glob
import json
import os
import statistics
import sys

ROWS = [("resnet18", "ResNet-18", "$(B,3,224,224)$"), ("resnet34", "ResNet-34", "$(B,3,224,224)$"),
        ("resnet50", "ResNet-50", "$(B,3,224,224)$"), ("mlp", "MLP", "$(B,784)$"),
        ("cnn", "CNN", "$(B,3,32,32)$"), ("transformer", "Transformer", "$(B,16,64)$"),
        ("vit", "ViT-style", "$(B,784)$")]
# SYCL/CUDA ratios of the earlier release of this artifact (unchanged column)
EARLIER = {"resnet18": (1.76, 1.88, 1.93), "resnet34": (1.65, 2.01, 2.02),
           "resnet50": (1.89, 2.11, 2.36), "mlp": (3.03, 2.81, 2.85), "cnn": (3.55, 3.86, 3.80),
           "transformer": (3.90, 4.02, 3.94), "vit": (3.25, 5.07, 3.16)}


def runs(d, name):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, f"{name}_run*.json")))]


def med(rs):
    v = [r["latency_p50_ms"] for r in rs if r.get("status") == "pass"]
    return statistics.median(v) if len(v) == len(rs) and v else None


def main():
    eager_dir, graph_dir = sys.argv[1], sys.argv[2]
    npass = ntotal = 0
    eager_r, graph_r = {"resnet": [], "compact": []}, {"compact": []}
    for key, label, shape in ROWS:
        nd = 3 if key.startswith("resnet") else 4
        for i, b in enumerate((1, 8, 16)):
            groups = {k: runs(eager_dir, f"sweep_{key}_b{b}_{k}") for k in ("cuda", "sycl")}
            if not key.startswith("resnet"):
                groups.update({k + "_graph": runs(graph_dir, f"sweep_{key}_b{b}_{k}_graph") for k in ("cuda", "sycl")})
            for rs in groups.values():
                ntotal += len(rs)
                npass += sum(r.get("status") == "pass" for r in rs)
            c, s = med(groups["cuda"]), med(groups["sycl"])
            r = s / c
            eager_r["resnet" if key.startswith("resnet") else "compact"].append(r)
            cells = [f"{c:.{nd}f}", f"{s:.{nd}f}", f"{r:.2f}$\\times$"]
            if key.startswith("resnet"):
                cells += ["", "", ""]
            else:
                cg, sg = med(groups["cuda_graph"]), med(groups["sycl_graph"])
                graph_r["compact"].append(sg / cg)
                cells += [f"{cg:.{nd}f}", f"{sg:.{nd}f}", f"{sg / cg:.2f}$\\times$"]
            head = f"{label} & {shape}" if i == 0 else " & "
            print(f"{head} & {b} & " + " & ".join(cells) + f" & {EARLIER[key][i]:.2f}$\\times$ \\\\")
    print(f"% ResNet eager {min(eager_r['resnet']):.2f}-{max(eager_r['resnet']):.2f}; compact eager "
          f"{min(eager_r['compact']):.2f}-{max(eager_r['compact']):.2f}; compact graphs "
          f"{min(graph_r['compact']):.2f}-{max(graph_r['compact']):.2f}; pass {npass}/{ntotal}")


if __name__ == "__main__":
    main()
