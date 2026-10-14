#!/usr/bin/env python3
"""Per-workload schedule source, weight and record-level contribution (existing data only)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/remote_4090_20260813/paired_resnet18_metaschedule_formal1024/record_analysis/paired_metaschedule_workloads.csv"
OUT = ROOT / "revision_round2/analysis/m2_workloads.md"

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
L = ["# ResNet-18 B1 paired MetaSchedule: all 19 extracted workloads\n",
     "Source: `" + str(SRC.relative_to(ROOT)).replace("\\", "/") + "`.\n",
     "Weight = MetaSchedule `ExtractedTask.weight`, the number of call sites of the deduplicated PrimFunc in the",
     "Relax `main` function, i.e. invocations per inference. Each of these PrimFuncs lowers to one device kernel,",
     "so the weight is also the expected launch count (to be confirmed by the whole-model Nsight trace, E1).",
     "Schedule source: MS = best finite MetaSchedule record applied; DLight = no finite record, DLight fallback.\n",
     "| Task | Workload | Group | Weight | CUDA src | SYCL src | CUDA best ms | SYCL best ms | SYCL/CUDA | Weighted CUDA ms | Weighted SYCL ms | Set |",
     "|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
agg = {}
for r in rows:
    w = int(r["weight"])
    c = float(r["cuda_best_ms"]) if r["cuda_best_ms"] else None
    s = float(r["sycl_best_ms"]) if r["sycl_best_ms"] else None
    cs, ss = ("MS" if c else "DLight"), ("MS" if s else "DLight")
    st = "matched" if c and s else ("CUDA-only MS" if c else ("SYCL-only MS" if s else "neither"))
    a = agg.setdefault(st, {"n": 0, "w": 0, "wc": 0.0, "ws": 0.0})
    a["n"] += 1; a["w"] += w; a["wc"] += w * (c or 0); a["ws"] += w * (s or 0)
    L.append(f"| {r['task_id']} | {r['name']} | {r['operator_group']} | {w} | {cs} | {ss} | "
             f"{c if c is None else f'{c:.4f}'} | {s if s is None else f'{s:.4f}'} | "
             f"{(f'{s/c:.2f}' if c and s else '-')} | {(f'{w*c:.4f}' if c else '-')} | {(f'{w*s:.4f}' if s else '-')} | {st} |")
L += ["", "## Summary by set\n", "| Set | Workloads | Calls per inference | Weighted CUDA ms (finite only) | Weighted SYCL ms (finite only) |",
      "|---|---:|---:|---:|---:|"]
for k in ["matched", "CUDA-only MS", "SYCL-only MS", "neither"]:
    a = agg.get(k)
    if a:
        L.append(f"| {k} | {a['n']} | {a['w']} | {a['wc']:.4f} | {a['ws']:.4f} |")
tot_w = sum(a["w"] for a in agg.values())
m = agg["matched"]
L += ["", f"Total calls per inference: {tot_w}. Matched set: {m['n']} workloads, {m['w']} calls.",
      f"Matched record-level gap: {m['ws']-m['wc']:.4f} ms = {100*(m['ws']-m['wc'])/(2.038-0.782):.1f}% of the 1.256 ms complete-model gap;",
      f"matched CUDA records cover {100*m['wc']/0.782:.1f}% of CUDA model time and matched SYCL records {100*m['ws']/2.038:.1f}% of SYCL model time.",
      f"Remainder outside the matched records: {1.256-(m['ws']-m['wc']):.3f} ms, spread over {tot_w-m['w']} non-matched calls plus all non-kernel time.",
      "Calls whose program differs in source between targets (one MS, one DLight): "
      + str(agg.get('CUDA-only MS', {}).get('w', 0) + agg.get('SYCL-only MS', {}).get('w', 0)) + "."]
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
