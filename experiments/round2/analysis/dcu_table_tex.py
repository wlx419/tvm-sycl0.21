#!/usr/bin/env python3
"""LaTeX rows for the final DCU table from results/dcu_final_20261013/results/*.json."""
import json
import sys
from pathlib import Path

ORDER = [("resnet18", "ResNet-18", "$(B,3,224,224)$"), ("resnet34", "ResNet-34", "$(B,3,224,224)$"),
         ("resnet50", "ResNet-50", "$(B,3,224,224)$"), ("mlp", "MLP", "$(B,784)$"),
         ("cnn", "CNN", "$(B,3,32,32)$"), ("transformer", "Transformer", "$(B,16,64)$"),
         ("vit", "ViT-style", "$(B,784)$")]


def load(d, name):
    out = {}
    for kind in ("compact", "resnet"):
        p = d / f"{kind}_{name}.json"
        if p.exists():
            for r in json.load(open(p)).get("results", []):
                out[(r["model"], r["batch"])] = r
    return out


def sci(x):
    if x is None:
        return "--"
    m, e = f"{x:.2e}".split("e")
    return f"${m}{{\\times}}10^{{{int(e)}}}$"


def main():
    d = Path(sys.argv[1])
    pub, r2, rocm = load(d, "pub"), load(d, "r2"), load(d, "rocm")
    for key, label, shape in ORDER:
        for i, b in enumerate((1, 8, 16)):
            p, s, h = pub.get((key, b), {}), r2.get((key, b), {}), rocm.get((key, b), {})
            hp = h.get("latency_p50_ms") if h.get("status") == "pass" else None
            pp, sp = p.get("latency_p50_ms"), s.get("latency_p50_ms")
            hcell = f"{hp:.3f}" if hp else "n/a"
            r_new = f"{sp / hp:.2f}$\\times$" if hp and sp else "--"
            r_old = f"{pp / hp:.2f}$\\times$" if hp and pp else "--"
            worst = max([x.get("max_abs_err") or 0 for x in (p, s)]) or None
            ok = all(x.get("status") == "pass" for x in (p, s))
            first = f"{label} & {shape}" if i == 0 else " & "
            print(f"{first} & {b} & {hcell} & {pp:.3f} & {sp:.3f} & {r_old} & {r_new} & {sci(worst)}{'' if ok else ' (fail)'} \\\\")


if __name__ == "__main__":
    main()
