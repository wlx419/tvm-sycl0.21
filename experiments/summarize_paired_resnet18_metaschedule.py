#!/usr/bin/env python3
"""Summarize paired CUDA/SYCL Relax/MetaSchedule ResNet-18 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


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


def summarize_database(work_dir: Path) -> Dict[str, object]:
    workloads = list(read_json_lines(work_dir / "database_workload.json"))
    records = list(read_json_lines(work_dir / "database_tuning_record.json"))
    workload_keys = {
        index: str(workload[0]) if isinstance(workload, list) and workload else str(index)
        for index, workload in enumerate(workloads)
    }
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

    best_by_key = {
        workload_keys[index]: min(values) * 1000.0
        for index, values in successes.items()
        if index in workload_keys
    }
    return {
        "workloads": len(workloads),
        "workload_keys": sorted(workload_keys.values()),
        "workloads_with_success": len(successes),
        "fallback_workloads": len(workloads) - len(successes),
        "trials": len(records),
        "failed_trials": failed,
        "failure_rate": failed / len(records) if records else 0.0,
        "sum_best_ms": sum(best_by_key.values()),
        "best_by_key": best_by_key,
        "all_workloads_have_success": bool(workloads) and len(successes) == len(workloads),
        "work_dir": str(work_dir),
    }


def load_probe(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    if len(rows) != 1:
        raise ValueError(f"Expected one result in {path}, found {len(rows)}")
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
    parser.add_argument("--apply-runs", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    root = args.root.resolve()
    databases = {
        target: summarize_database(root / "tune" / target) for target in TARGETS
    }
    tuning_results = {
        target: load_probe(root / "tune" / target / "probe.json") for target in TARGETS
    }

    apply_rows = []
    by_target: Dict[str, List[Dict[str, object]]] = {target: [] for target in TARGETS}
    for target in TARGETS:
        for run in range(1, args.apply_runs + 1):
            row = load_probe(root / "apply" / target / f"run{run}.json")
            row["run"] = run
            apply_rows.append(row)
            by_target[target].append(row)

    execution = {}
    for target, rows in by_target.items():
        p50_values = [float(row["latency_p50_ms"]) for row in rows]
        execution[target] = {
            "runs": len(rows),
            "all_pass": all(row.get("status") == "pass" for row in rows),
            "p50_ms_by_run": p50_values,
            "median_p50_ms": statistics.median(p50_values),
            "min_p50_ms": min(p50_values),
            "max_p50_ms": max(p50_values),
            "max_abs_error": max(float(row.get("max_abs_err", 0.0)) for row in rows),
        }

    cuda_keys = set(databases["cuda"]["workload_keys"])
    sycl_keys = set(databases["sycl"]["workload_keys"])
    cuda_best = databases["cuda"]["best_by_key"]
    sycl_best = databases["sycl"]["best_by_key"]
    matched_rows = []
    for key in sorted(set(cuda_best) & set(sycl_best)):
        cuda_ms = float(cuda_best[key])
        sycl_ms = float(sycl_best[key])
        matched_rows.append(
            {
                "workload_key": key,
                "cuda_best_ms": cuda_ms,
                "sycl_best_ms": sycl_ms,
                "sycl_over_cuda": sycl_ms / cuda_ms,
                "gap_ms": sycl_ms - cuda_ms,
            }
        )
    matched_rows.sort(key=lambda row: float(row["gap_ms"]), reverse=True)

    cuda_p50 = execution["cuda"]["median_p50_ms"]
    sycl_p50 = execution["sycl"]["median_p50_ms"]
    summary = {
        "protocol": {
            "model": "torchvision ResNet-18 with folded batch normalization",
            "input_shape": f"(1, 3, {args.image_size}, {args.image_size})",
            "stack": "TVM v0.21 Relax/MetaSchedule with DLight fallback",
            "targets": list(TARGETS),
            "equal_global_trial_budget": args.trials,
            "apply_only_runs_per_target": args.apply_runs,
            "reported_latency": "median of independent synchronized p50 runs",
        },
        "databases": databases,
        "tuning_results": tuning_results,
        "execution": execution,
        "comparison": {
            "workload_sets_identical": cuda_keys == sycl_keys,
            "cuda_workloads": len(cuda_keys),
            "sycl_workloads": len(sycl_keys),
            "matched_workloads_with_success": len(matched_rows),
            "cuda_median_p50_ms": cuda_p50,
            "sycl_median_p50_ms": sycl_p50,
            "sycl_over_cuda_complete_model": sycl_p50 / cuda_p50,
        },
        "matched_workloads": matched_rows,
        "checks": {
            "all_apply_runs_pass": all(item["all_pass"] for item in execution.values()),
            "workload_sets_identical": cuda_keys == sycl_keys,
            "both_databases_nonempty": all(item["trials"] > 0 for item in databases.values()),
        },
    }

    (root / "paired_metaschedule_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(root / "paired_metaschedule_apply_runs.csv", apply_rows)
    write_csv(root / "paired_metaschedule_matched_workloads.csv", matched_rows)

    ratio = summary["comparison"]["sycl_over_cuda_complete_model"]
    lines = [
        "# Paired ResNet-18 Relax/MetaSchedule Results",
        "",
        f"Equal global budget: {args.trials} trials per target.",
        f"Workload sets identical: {cuda_keys == sycl_keys}.",
        f"All apply-only runs pass: {summary['checks']['all_apply_runs_pass']}.",
        "",
        "| Target | MS measured | Fallback | Trials | Failed | Median p50 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        db = databases[target]
        ex = execution[target]
        lines.append(
            f"| {target.upper()} | {db['workloads_with_success']}/{db['workloads']} | "
            f"{db['fallback_workloads']} | {db['trials']} | {db['failed_trials']} | "
            f"{ex['median_p50_ms']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Complete-model SYCL/CUDA ratio: **{ratio:.3f}x**.",
            "",
            "The complete-model ratio is primary. The matched-workload database table is "
            "unweighted diagnostic evidence because graph call multiplicities differ.",
        ]
    )
    (root / "paired_metaschedule_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["checks"], sort_keys=True))
    if not all(summary["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
