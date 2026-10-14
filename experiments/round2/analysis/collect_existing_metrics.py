#!/usr/bin/env python3
"""Collect dispersion, compile-time and tuning-time metrics from existing raw records.

No new measurements: every number comes from files already in results/.
Outputs (under revision_round2/analysis/):
  existing_metrics.json   machine-readable
  existing_metrics.md     summary tables
"""
from __future__ import annotations

import csv
import io
import json
import re
import statistics
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "revision_round2" / "analysis"

MODELS = ["resnet18", "resnet34", "resnet50", "mlp", "cnn", "transformer", "vit"]
BATCHES = [1, 8, 16]


def read_tar_csvs(tar_path: Path, pattern: str) -> dict[str, list[dict]]:
    out = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and re.search(pattern, m.name):
                data = tf.extractfile(m).read().decode("utf-8")
                out[m.name] = list(csv.DictReader(io.StringIO(data)))
    return out


def f(x):
    return float(x) if x not in (None, "") else None


def row_metrics(r: dict) -> dict:
    return {
        "p50": f(r.get("latency_p50_ms")),
        "p90": f(r.get("latency_p90_ms")),
        "p99": f(r.get("latency_p99_ms")),
        "mean": f(r.get("latency_mean_ms")),
        "compile_s": f(r.get("compile_s")),
        "status": r.get("status"),
        "max_abs": f(r.get("max_abs_err")),
    }


# ---------------------------------------------------------------- Table 9 (4090 paired sweep)
def table9():
    out = {}
    for tar in [RES / "remote_4090_20260810" / "unified_resnet_formal_20260810.tar.gz",
                RES / "remote_4090_20260810" / "unified_compact_formal_20260810.tar.gz"]:
        for name, rows in read_tar_csvs(tar, r"/sweep/(cuda|sycl)/[a-z0-9]+_b\d+\.csv$").items():
            m = re.search(r"/sweep/(cuda|sycl)/([a-z0-9]+)_b(\d+)\.csv$", name)
            backend, model, b = m.group(1), m.group(2), int(m.group(3))
            out.setdefault(model, {}).setdefault(b, {})[backend] = row_metrics(rows[0])
    return out


# ---------------------------------------------------------------- Table 10 (DCU SYCL sweep)
def table10():
    base = RES / "remote_dcu_20260819" / "artifact" / "results"
    out = {}
    for fn in ["dcu_compact_models_b1_b8_b16_device0_isolated_final_20260819.csv",
               "dcu_resnet18_34_b1_b8_b16_device0_onnx_20260819.csv",
               "dcu_resnet50_b1_b8_b16_device1_onnx_20260819.csv"]:
        for r in csv.DictReader(open(base / fn, encoding="utf-8")):
            out.setdefault(r["model"], {})[int(r["batch"])] = row_metrics(r) | {
                "target_str": r.get("resolved_target", "")}
    return out


# ---------------------------------------------------------------- Table 11 (DCU HIP vs SYCL)
def table11():
    base = RES / "remote_dcu_20260904" / "results"
    out = {}
    for fam in ["compact", "resnet"]:
        for backend in ["rocm", "sycl"]:
            for r in csv.DictReader(open(base / f"{fam}_{backend}.csv", encoding="utf-8")):
                out.setdefault(r["model"], {}).setdefault(int(r["batch"]), {})[backend] = (
                    row_metrics(r) | {"target_str": r.get("resolved_target", "")})
    return out


# ---------------------------------------------------------------- Table 12 (oneDNN vs TVM-SYCL)
def table12():
    base = RES / "remote_4090_20260811" / "onednn_nvidia_comparison_20260811"
    out = {}
    for run in range(1, 6):
        for r in csv.DictReader(open(base / f"onednn_cuda_vendor_run{run}.csv", encoding="utf-8")):
            d = out.setdefault(r["op"], {"onednn": [], "tvm_sycl": [], "shape": r["shape"]})
            d["onednn"].append({"p50": f(r["p50_ms"]), "p90": f(r["p90_ms"])})
        for op in ["matmul", "row_sum", "softmax", "conv2d_nhwc"]:
            p = base / f"tvm_sycl_run{run}_{op}.json"
            if p.exists():
                j = json.load(open(p, encoding="utf-8"))
                key = op if op in out else op
                out.setdefault(key, {"onednn": [], "tvm_sycl": [], "shape": j.get("shape")})
                out[key]["tvm_sycl"].append({"p50": j["latency_p50_ms"], "p90": j["latency_p90_ms"],
                                             "p99": j.get("latency_p99_ms"), "wg_size": j.get("wg_size"),
                                             "build_s": j.get("build_s")})
    return out


# ---------------------------------------------------------------- Table 4 (paired ResNet-18 MS)
def table4():
    base = RES / "remote_4090_20260813" / "paired_resnet18_metaschedule_formal1024"
    out = {}
    for backend in ["cuda", "sycl"]:
        runs = []
        for i in (1, 2, 3):
            j = json.load(open(base / "apply" / backend / f"run{i}.json", encoding="utf-8"))
            r = j["results"][0]
            runs.append({"p50": r["latency_p50_ms"], "p90": r["latency_p90_ms"], "p99": r["latency_p99_ms"],
                         "mean": r["latency_mean_ms"], "compile_s": r["compile_s"]})
        out[backend] = runs
    return out


# ---------------------------------------------------------------- tuning wall time from logs
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def log_span(text: str):
    stamps = [datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
              for line in text.splitlines() if (m := TS.match(line))]
    return (stamps[0], stamps[-1]) if stamps else (None, None)


def tuning_times():
    out = {}
    base = RES / "remote_4090_20260813" / "paired_resnet18_metaschedule_formal1024" / "tune"
    for backend in ["cuda", "sycl"]:
        p = base / backend / "logs" / "tvm.meta_schedule.logging.task_scheduler.log"
        a, b = log_span(p.read_text(encoding="utf-8", errors="replace"))
        out[f"paired_resnet18_{backend}"] = (b - a).total_seconds() if a else None
    # Table 8 coverage campaign (SYCL target, 4090)
    for tar in [RES / "remote_4090_20260810" / "unified_resnet_formal_20260810.tar.gz",
                RES / "remote_4090_20260810" / "unified_compact_formal_20260810.tar.gz"]:
        with tarfile.open(tar, "r:gz") as tf:
            for m in tf.getmembers():
                mm = re.search(r"/coverage/([a-z0-9]+)/logs/tvm\.meta_schedule\.logging\.task_scheduler\.log$", m.name)
                if mm:
                    a, b = log_span(tf.extractfile(m).read().decode("utf-8", errors="replace"))
                    out[f"coverage_{mm.group(1)}_sycl"] = (b - a).total_seconds() if a else None
    return out


def main():
    data = {"table4": table4(), "table9": table9(), "table10": table10(), "table11": table11(),
            "table12": table12(), "tuning_s": tuning_times()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "existing_metrics.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

    L = []
    w = L.append
    w("# Dispersion, compile time and tuning time from existing records\n")
    w("Generated by `revision_round2/scripts/collect_existing_metrics.py`. No new measurements.\n")
    w("Only p50/p90/p99 and mean were stored per run; per-iteration samples were not kept, so")
    w("confidence intervals require re-measurement.\n")

    w("## Table 4: paired ResNet-18 B1 MetaSchedule (three database-reuse processes, 10 warm-up + 50)\n")
    w("| Target | p50 per process (ms) | p90 per process | p99 per process | compile s per process |")
    w("|---|---|---|---|---|")
    for bk, runs in data["table4"].items():
        w(f"| {bk.upper()} | {' / '.join(f'{r['p50']:.3f}' for r in runs)} | "
          f"{' / '.join(f'{r['p90']:.3f}' for r in runs)} | {' / '.join(f'{r['p99']:.3f}' for r in runs)} | "
          f"{' / '.join(f'{r['compile_s']:.0f}' for r in runs)} |")
    t = data["tuning_s"]
    w(f"\nTuning wall time (1,024 trials, from task-scheduler log timestamps): "
      f"CUDA {t['paired_resnet18_cuda']/60:.1f} min, SYCL {t['paired_resnet18_sycl']/60:.1f} min.\n")

    w("## Table 9: 4090 paired default-pipeline sweep (5 warm-up + 30)\n")
    w("| Model | B | CUDA p50 | CUDA p90 | CUDA p99 | SYCL p50 | SYCL p90 | SYCL p99 | SYCL p90/p50 | compile s CUDA | compile s SYCL |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for b in BATCHES:
            c = data["table9"].get(m, {}).get(b, {}).get("cuda")
            s = data["table9"].get(m, {}).get(b, {}).get("sycl")
            if not c or not s:
                continue
            w(f"| {m} | {b} | {c['p50']:.4f} | {c['p90']:.4f} | {c['p99']:.4f} | {s['p50']:.4f} | {s['p90']:.4f} | "
              f"{s['p99']:.4f} | {s['p90']/s['p50']:.2f} | {c['compile_s']:.1f} | {s['compile_s']:.1f} |")

    w("\n## Table 10: DCU SYCL sweep (5 warm-up + 30)\n")
    w("| Model | B | p50 | p90 | p99 | p90/p50 | compile s |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for b in BATCHES:
            r = data["table10"].get(m, {}).get(b)
            if r:
                w(f"| {m} | {b} | {r['p50']:.3f} | {r['p90']:.3f} | {r['p99']:.3f} | {r['p90']/r['p50']:.2f} | {r['compile_s']:.1f} |")

    w("\n## Table 11: DCU paired TVM-HIP / SYCL session (5 warm-up + 30)\n")
    w("| Model | B | HIP p50 | HIP p90 | SYCL p50 | SYCL p90 | SYCL/HIP | SYCL here vs Table 10 | compile s HIP | compile s SYCL |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODELS:
        for b in BATCHES:
            d = data["table11"].get(m, {}).get(b, {})
            h, s = d.get("rocm"), d.get("sycl")
            t10 = data["table10"].get(m, {}).get(b)
            if not s:
                continue
            drift = f"{100*(s['p50']-t10['p50'])/t10['p50']:+.1f}%" if t10 else "-"
            hip_ok = h and h.get("status") == "pass"
            w(f"| {m} | {b} | {h['p50']:.3f} | {h['p90']:.3f} | {s['p50']:.3f} | {s['p90']:.3f} | "
              f"{s['p50']/h['p50']:.2f} | {drift} | {h['compile_s']:.1f} | {s['compile_s']:.1f} |" if hip_ok else
              f"| {m} | {b} | n/a ({h['status'] if h else 'missing'}) | | {s['p50']:.3f} | {s['p90']:.3f} | | {drift} | "
              f"{(h or {}).get('compile_s') or 0:.1f} | {s['compile_s']:.1f} |")

    w("\n## Table 12: oneDNN vs TVM-SYCL, five processes each (10 warm-up + 30)\n")
    w("| Op | Shape | oneDNN p50 range over 5 processes | median | TVM-SYCL p50 range | median | TVM-SYCL wg_size | TVM-SYCL max p99/p50 |")
    w("|---|---|---|---:|---|---:|---:|---:|")
    for op, d in data["table12"].items():
        o = [x["p50"] for x in d["onednn"]]
        t_ = [x["p50"] for x in d["tvm_sycl"]]
        if not o or not t_:
            continue
        tail = max((x.get("p99") or 0) / x["p50"] for x in d["tvm_sycl"])
        w(f"| {op} | {d['shape']} | {min(o):.4f}-{max(o):.4f} | {statistics.median(o):.4f} | "
          f"{min(t_):.4f}-{max(t_):.4f} | {statistics.median(t_):.4f} | {d['tvm_sycl'][0].get('wg_size')} | {tail:.2f} |")

    w("\n## Tuning wall time per model (Table 8 coverage campaign, SYCL target, 1,024 trials)\n")
    w("| Campaign | minutes |")
    w("|---|---:|")
    for k, v in sorted(t.items()):
        if v is not None:
            w(f"| {k} | {v/60:.1f} |")
    (OUT / "existing_metrics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
