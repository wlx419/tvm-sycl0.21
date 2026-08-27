#!/usr/bin/env python3
"""Summarize the formal Relax/MetaSchedule compact-model matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List


MODELS = ("mlp", "cnn", "transformer", "vit")
BATCHES = (1, 8, 16)
TARGETS = ("cuda", "sycl")
FAIL_LATENCY_S = 1e9


def read_json_lines(path: Path) -> Iterable[object]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_database(model: str, work_dir: Path) -> Dict[str, object]:
    workloads = list(read_json_lines(work_dir / "database_workload.json"))
    records = list(read_json_lines(work_dir / "database_tuning_record.json"))
    successes: Dict[int, List[float]] = {}
    failed = 0
    for record in records:
        try:
            values = [float(value) for value in record[1][1]]
        except (IndexError, TypeError, ValueError):
            values = []
        finite = [value for value in values if math.isfinite(value) and value < FAIL_LATENCY_S]
        if finite:
            successes.setdefault(int(record[0]), []).extend(finite)
        else:
            failed += 1
    return {
        "model": model,
        "workloads": len(workloads),
        "workloads_with_success": len(successes),
        "fallback_workloads": len(workloads) - len(successes),
        "trials": len(records),
        "failed_trials": failed,
        "failure_rate": failed / len(records) if records else 0.0,
        "sum_best_ms": sum(min(values) * 1000.0 for values in successes.values()),
        "all_workloads_have_success": bool(workloads) and len(successes) == len(workloads),
    }


def load_result(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"status": "missing"}
    rows = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    return dict(rows[0]) if len(rows) == 1 else {"status": "invalid_result_count"}


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    coverage = [summarize_database(model, root / "coverage" / model) for model in MODELS]
    for row in coverage:
        execution = load_result(root / "coverage" / str(row["model"]) / "probe.json")
        row["execution_status"] = execution.get("status")
        row["max_abs_err"] = execution.get("max_abs_err")
        row["latency_p50_ms"] = execution.get("latency_p50_ms")
    pairs = []
    for model in MODELS:
        for batch in BATCHES:
            cuda = load_result(root / "sweep" / "cuda" / f"{model}_b{batch}.json")
            sycl = load_result(root / "sweep" / "sycl" / f"{model}_b{batch}.json")
            cuda_ms = cuda.get("latency_p50_ms")
            sycl_ms = sycl.get("latency_p50_ms")
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
                    "pair_pass": cuda.get("status") == "pass" and sycl.get("status") == "pass",
                }
            )

    checks = {
        "coverage_pass": all(
            row["execution_status"] == "pass"
            and row["workloads"] > 0
            and row["workloads_with_success"] > 0
            and row["workloads_with_success"] + row["fallback_workloads"] == row["workloads"]
            for row in coverage
        ),
        "metaschedule_full_coverage": all(
            row["all_workloads_have_success"] for row in coverage
        ),
        "sweep_pass": all(row["pair_pass"] for row in pairs),
    }
    checks["all_pass"] = checks["coverage_pass"] and checks["sweep_pass"]
    summary = {"coverage": coverage, "pairs": pairs, "checks": checks}
    (root / "unified_compact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(root / "unified_compact_coverage.csv", coverage)
    write_csv(root / "unified_compact_sweep.csv", pairs)
    print(json.dumps(checks, sort_keys=True))
    if not checks["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
