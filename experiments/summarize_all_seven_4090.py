#!/usr/bin/env python3
"""Merge the two formal RTX 4090 runs into one seven-model summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


MODELS: Sequence[Tuple[str, str, str]] = (
    ("ResNet-18", "resnet18", "resnet"),
    ("ResNet-34", "resnet34", "resnet"),
    ("ResNet-50", "resnet50", "resnet"),
    ("MLP", "mlp", "compact"),
    ("CNN", "cnn", "compact"),
    ("Transformer", "transformer", "compact"),
    ("ViT-style", "vit", "compact"),
)
BATCHES = (1, 8, 16)
TARGETS = ("cuda", "sycl")
FAIL_LATENCY_S = 1e9


def read_lines(path: Path) -> List[object]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize_database(label: str, key: str, root: Path) -> Dict[str, object]:
    work_dir = root / "coverage" / key
    workloads = read_lines(work_dir / "database_workload.json")
    records = read_lines(work_dir / "database_tuning_record.json")
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
        "model": label,
        "model_key": key,
        "stack": "TVM v0.21 Relax/MetaSchedule",
        "workloads": len(workloads),
        "workloads_with_success": len(successes),
        "fallback_workloads": len(workloads) - len(successes),
        "trials": len(records),
        "failed_trials": failed,
        "failure_rate": failed / len(records) if records else None,
        "sum_best_ms": sum(min(values) * 1000.0 for values in successes.values()),
        "all_workloads_have_success": bool(workloads) and len(successes) == len(workloads),
    }


def load_probe(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if len(rows) != 1:
        return {"status": "invalid_result_count", "path": str(path)}
    result = dict(rows[0])
    result["_protocol"] = payload.get("protocol", {})
    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet-root", type=Path, required=True)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    roots = {"resnet": args.resnet_root.resolve(), "compact": args.compact_root.resolve()}

    coverage = [summarize_database(label, key, roots[group]) for label, key, group in MODELS]
    for row, (_, key, group) in zip(coverage, MODELS):
        execution = load_probe(roots[group] / "coverage" / key / "probe.json")
        row["execution_status"] = execution.get("status")
        row["max_abs_err"] = execution.get("max_abs_err")
        row["latency_p50_ms"] = execution.get("latency_p50_ms")
    pairs = []
    for label, key, group in MODELS:
        for batch in BATCHES:
            cuda = load_probe(roots[group] / "sweep" / "cuda" / f"{key}_b{batch}.json")
            sycl = load_probe(roots[group] / "sweep" / "sycl" / f"{key}_b{batch}.json")
            cuda_ms = cuda.get("latency_p50_ms")
            sycl_ms = sycl.get("latency_p50_ms")
            cuda_protocol = cuda.get("_protocol", {})
            sycl_protocol = sycl.get("_protocol", {})
            shared_case = (
                cuda.get("model") == key
                and sycl.get("model") == key
                and cuda.get("batch") == batch
                and sycl.get("batch") == batch
                and cuda.get("target") == "cuda"
                and sycl.get("target") == "sycl"
                and cuda.get("input_shape") == sycl.get("input_shape")
                and cuda.get("atol") == sycl.get("atol") == 2e-4
                and cuda.get("rtol") == sycl.get("rtol") == 2e-3
            )
            if group == "compact":
                shared_case = shared_case and (
                    cuda.get("seed_target_alias") == sycl.get("seed_target_alias") == "sycl"
                    and cuda_protocol.get("seed") == sycl_protocol.get("seed") == 123
                )
            else:
                shared_case = shared_case and (
                    cuda_protocol.get("seed") == sycl_protocol.get("seed") == 321
                    and cuda_protocol.get("image_size")
                    == sycl_protocol.get("image_size")
                    == 224
                    and cuda_protocol.get("num_classes")
                    == sycl_protocol.get("num_classes")
                    == 1000
                )
            pairs.append(
                {
                    "model": label,
                    "model_key": key,
                    "batch": batch,
                    "input_shape": sycl.get("input_shape") or cuda.get("input_shape"),
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
                    "worst_max_abs_err": max(
                        float(cuda.get("max_abs_err", 0.0)), float(sycl.get("max_abs_err", 0.0))
                    ),
                    "paired_case_match": shared_case,
                    "pair_pass": cuda.get("status") == "pass" and sycl.get("status") == "pass",
                }
            )

    checks = {
        "seven_model_coverage_pass": all(
            row["execution_status"] == "pass"
            and row["workloads"] > 0
            and row["workloads_with_success"] > 0
            and row["workloads_with_success"] + row["fallback_workloads"] == row["workloads"]
            for row in coverage
        ),
        "seven_model_full_metaschedule_coverage": all(
            row["all_workloads_have_success"] for row in coverage
        ),
        "all_42_backend_runs_pass": all(row["pair_pass"] for row in pairs),
        "all_21_pairs_use_matching_cases": all(row["paired_case_match"] for row in pairs),
    }
    checks["all_pass"] = (
        checks["seven_model_coverage_pass"]
        and checks["all_42_backend_runs_pass"]
        and checks["all_21_pairs_use_matching_cases"]
    )
    summary = {
        "protocol": {
            "stack": "TVM v0.21 Relax/MetaSchedule",
            "models": 7,
            "batches": list(BATCHES),
            "backend_runs": len(MODELS) * len(BATCHES) * len(TARGETS),
            "warmup": 5,
            "measure": 30,
            "tolerance": {"atol": 2e-4, "rtol": 2e-3},
        },
        "coverage": coverage,
        "pairs": pairs,
        "checks": checks,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "all_seven_4090_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.out_dir / "all_seven_4090_coverage.csv", coverage)
    write_csv(args.out_dir / "all_seven_4090_sweep.csv", pairs)

    lines = [
        "# Seven-model TVM v0.21 Relax/MetaSchedule Results",
        "",
        f"Coverage: {'pass' if checks['seven_model_coverage_pass'] else 'FAIL'}.",
        f"All 42 backend runs: {'pass' if checks['all_42_backend_runs_pass'] else 'FAIL'}.",
        "",
        "## MetaSchedule Coverage",
        "",
        "| Model | MS measured | DLight fallback | Records | Failed | Failure rate | Sum best ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in coverage:
        failure_rate = row["failure_rate"]
        rate = f"{100.0 * failure_rate:.1f}%" if isinstance(failure_rate, float) else "NA"
        lines.append(
            f"| {row['model']} | {row['workloads_with_success']}/{row['workloads']} | "
            f"{row['fallback_workloads']} | {row['trials']} | {row['failed_trials']} | {rate} | "
            f"{row['sum_best_ms']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired Execution",
            "",
            "| Model | B | CUDA p50 ms | SYCL p50 ms | Ratio | Worst max abs | Pass |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in pairs:
        ratio = row["sycl_over_cuda"]
        lines.append(
            f"| {row['model']} | {row['batch']} | {row['cuda_p50_ms']:.6f} | "
            f"{row['sycl_p50_ms']:.6f} | {ratio:.3f}x | {row['worst_max_abs_err']:.3e} | "
            f"{row['pair_pass']} |"
        )
    (args.out_dir / "all_seven_4090_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, sort_keys=True))
    if not checks["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
