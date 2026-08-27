#!/usr/bin/env python3
"""Summarize the formal Relax/MetaSchedule ResNet experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


MODELS = ("resnet18", "resnet34", "resnet50")
BATCHES = (1, 8, 16)
TARGETS = ("cuda", "sycl")
FAIL_LATENCY_S = 1e9


def read_json_lines(path: Path) -> Iterable[object]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_database(model: str, work_dir: Path) -> Dict[str, object]:
    workloads = list(read_json_lines(work_dir / "database_workload.json"))
    records = list(read_json_lines(work_dir / "database_tuning_record.json"))
    successes: Dict[int, List[float]] = {}
    failed = 0
    for record in records:
        workload_id = int(record[0])
        try:
            run_secs = [float(value) for value in record[1][1]]
        except (IndexError, TypeError, ValueError):
            run_secs = []
        finite = [value for value in run_secs if math.isfinite(value) and value < FAIL_LATENCY_S]
        if finite:
            successes.setdefault(workload_id, []).extend(finite)
        else:
            failed += 1

    best_ms = [min(values) * 1000.0 for values in successes.values()]
    workload_count = len(workloads)
    covered_count = len(successes)
    return {
        "model": model,
        "stack": "TVM v0.21 Relax/MetaSchedule",
        "workloads": workload_count,
        "workloads_with_success": covered_count,
        "fallback_workloads": workload_count - covered_count,
        "trials": len(records),
        "failed_trials": failed,
        "failure_rate": failed / len(records) if records else 0.0,
        "sum_best_ms": sum(best_ms),
        "all_workloads_have_success": workload_count > 0 and covered_count == workload_count,
        "work_dir": str(work_dir),
    }


def load_probe_result(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"status": "missing", "result_path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if len(rows) != 1:
        return {
            "status": "invalid_result_count",
            "result_count": len(rows),
            "result_path": str(path),
        }
    row = dict(rows[0])
    row["result_path"] = str(path)
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    coverage = [summarize_database(model, root / "coverage" / model) for model in MODELS]
    for row in coverage:
        execution = load_probe_result(root / "coverage" / str(row["model"]) / "probe.json")
        row["execution_status"] = execution.get("status")
        row["max_abs_err"] = execution.get("max_abs_err")
        row["latency_p50_ms"] = execution.get("latency_p50_ms")

    sweep = []
    for target in TARGETS:
        for model in MODELS:
            for batch in BATCHES:
                row = load_probe_result(root / "sweep" / target / f"{model}_b{batch}.json")
                row.setdefault("target", target)
                row.setdefault("model", model)
                row.setdefault("batch", batch)
                sweep.append(row)

    pairs = []
    by_key = {(str(row["model"]), int(row["batch"]), str(row["target"])): row for row in sweep}
    for model in MODELS:
        for batch in BATCHES:
            cuda = by_key[(model, batch, "cuda")]
            sycl = by_key[(model, batch, "sycl")]
            cuda_ms = cuda.get("latency_p50_ms")
            sycl_ms = sycl.get("latency_p50_ms")
            pair_pass = cuda.get("status") == "pass" and sycl.get("status") == "pass"
            pairs.append(
                {
                    "model": model,
                    "batch": batch,
                    "cuda_status": cuda.get("status"),
                    "sycl_status": sycl.get("status"),
                    "cuda_p50_ms": cuda_ms,
                    "sycl_p50_ms": sycl_ms,
                    "sycl_over_cuda": (
                        float(sycl_ms) / float(cuda_ms)
                        if isinstance(cuda_ms, (int, float))
                        and isinstance(sycl_ms, (int, float))
                        and cuda_ms > 0
                        else None
                    ),
                    "cuda_max_abs_err": cuda.get("max_abs_err"),
                    "sycl_max_abs_err": sycl.get("max_abs_err"),
                    "pair_pass": pair_pass,
                }
            )

    metaschedule_full_coverage = all(row["all_workloads_have_success"] for row in coverage)
    coverage_pass = all(
        row["execution_status"] == "pass"
        and row["workloads"] > 0
        and row["workloads_with_success"] > 0
        and row["workloads_with_success"] + row["fallback_workloads"] == row["workloads"]
        for row in coverage
    )
    sweep_pass = all(row["pair_pass"] for row in pairs)
    summary = {
        "protocol": {
            "models": list(MODELS),
            "batches": list(BATCHES),
            "targets": list(TARGETS),
            "execution_stack": "TVM v0.21 Relax VM default generic-GPU pipeline",
            "coverage_stack": "TVM v0.21 Relax/MetaSchedule with DLight fallback",
            "reference": "PyTorch CPU with identical parameters and input",
        },
        "coverage": coverage,
        "sweep": sweep,
        "pairs": pairs,
        "checks": {
            "coverage_pass": coverage_pass,
            "metaschedule_full_coverage": metaschedule_full_coverage,
            "sweep_pass": sweep_pass,
            "all_pass": coverage_pass and sweep_pass,
        },
    }

    root.mkdir(parents=True, exist_ok=True)
    (root / "unified_resnet_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(root / "unified_resnet_coverage.csv", coverage)
    write_csv(root / "unified_resnet_sweep.csv", pairs)

    lines = [
        "# Unified ResNet Relax/MetaSchedule Results",
        "",
        f"MetaSchedule plus DLight schedule coverage: {'pass' if coverage_pass else 'FAIL'}.",
        f"Every workload has a measured MetaSchedule record: {metaschedule_full_coverage}.",
        f"Paired execution check: {'pass' if sweep_pass else 'FAIL'}.",
        "",
        "## MetaSchedule Coverage",
        "",
        "| Model | MS measured | DLight fallback | Records | Failed | Sum best ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        lines.append(
            f"| {row['model']} | {row['workloads_with_success']}/{row['workloads']} | "
            f"{row['fallback_workloads']} | {row['trials']} | {row['failed_trials']} | "
            f"{row['sum_best_ms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Default-Pipeline Execution",
            "",
            "| Model | Batch | CUDA p50 ms | SYCL p50 ms | Ratio | Pass |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in pairs:
        ratio = row["sycl_over_cuda"]
        ratio_text = f"{ratio:.3f}x" if isinstance(ratio, float) else "NA"
        cuda_text = f"{row['cuda_p50_ms']:.6f}" if isinstance(row["cuda_p50_ms"], float) else "NA"
        sycl_text = f"{row['sycl_p50_ms']:.6f}" if isinstance(row["sycl_p50_ms"], float) else "NA"
        lines.append(
            f"| {row['model']} | {row['batch']} | {cuda_text} | {sycl_text} | "
            f"{ratio_text} | {row['pair_pass']} |"
        )
    (root / "unified_resnet_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["checks"], sort_keys=True))
    if not args.allow_partial and not summary["checks"]["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
