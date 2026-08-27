#!/usr/bin/env python3
"""Analyze paired CUDA/SYCL MetaSchedule databases for one Relax model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import tvm
from tvm import meta_schedule as ms


def mean_run_secs(record) -> float | None:
    if not record.run_secs:
        return None
    values = [float(value) for value in record.run_secs]
    if not values or not all(math.isfinite(value) and value < 1e9 for value in values):
        return None
    return sum(values) / len(values)


def workload_key(mod) -> str:
    return str(int(tvm.ir.structural_hash(mod)) & ((1 << 64) - 1))


def load_workloads(work_dir: Path) -> tuple[list[str], dict[str, object]]:
    keys = []
    modules = {}
    for line in (work_dir / "database_workload.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = row[0]
            key = str(key)
            keys.append(key)
            modules[key] = ms.database.Workload.from_json(row).mod
    return keys, modules


def parse_task_table(log_path: Path) -> dict[int, dict[str, object]]:
    tasks: dict[int, dict[str, object]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) != 9 or not columns[0].isdigit():
            continue
        try:
            tasks[int(columns[0])] = {
                "name": columns[1],
                "flop": int(columns[2]),
                "weight": int(columns[3]),
                "trials_reported": int(columns[7]),
            }
        except ValueError:
            continue
    return tasks


def buffer_shapes(mod) -> list[tuple[int, ...]]:
    func = mod["main"]
    shapes = []
    for param in func.params:
        buf = func.buffer_map.get(param)
        if buf is None:
            continue
        try:
            shapes.append(tuple(int(value) for value in buf.shape))
        except (TypeError, ValueError):
            pass
    return shapes


def operator_group(name: str, mod) -> str:
    lower = name.lower()
    if "conv2d" in lower:
        filters = [
            shape
            for shape in buffer_shapes(mod)
            if len(shape) == 4
            and shape[0] > 1
            and shape[1] > 1
            and shape[-1] in (1, 3, 7)
            and shape[-2] == shape[-1]
        ]
        if filters:
            kernel = filters[0][-1]
            return f"Conv2d {kernel}x{kernel}"
        return "Conv2d"
    if "matmul" in lower:
        return "Dense/matmul"
    if "pool" in lower:
        return "Pooling"
    if "reshape" in lower:
        return "Reshape"
    return "Elementwise/reduction"


def load_target(work_dir: Path) -> dict[str, object]:
    database = ms.database.JSONDatabase(
        path_workload=str(work_dir / "database_workload.json"),
        path_tuning_record=str(work_dir / "database_tuning_record.json"),
    )
    records = list(database.get_all_tuning_records())
    grouped = defaultdict(list)
    for record in records:
        grouped[workload_key(record.workload.mod)].append(record)

    best = {}
    for key, candidates in grouped.items():
        finite = [(mean_run_secs(record), record) for record in candidates]
        finite = [(value, record) for value, record in finite if value is not None]
        if finite:
            best[key] = min(finite, key=lambda item: item[0])

    keys, modules = load_workloads(work_dir)
    return {
        "database": database,
        "records": records,
        "grouped": grouped,
        "best": best,
        "keys": keys,
        "modules": modules,
        "tasks": parse_task_table(work_dir / "probe.log"),
    }


def write_scheduled_tir(out_dir: Path, target_name: str, key: str, name: str, record) -> str:
    candidate = record.as_measure_candidate()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    path = out_dir / "scheduled_tir" / f"{int(key):020d}_{slug}_{target_name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate.sch.mod.script() + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Experiment root containing tune/cuda and tune/sycl")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir or root / "record_analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: load_target(root / "tune" / name) for name in ("cuda", "sycl")}

    if targets["cuda"]["keys"] != targets["sycl"]["keys"]:
        raise RuntimeError("CUDA and SYCL workload order differs")

    rows = []
    for task_id, key in enumerate(targets["cuda"]["keys"]):
        cuda_task = targets["cuda"]["tasks"].get(task_id, {})
        sycl_task = targets["sycl"]["tasks"].get(task_id, {})
        name = str(cuda_task.get("name") or sycl_task.get("name") or f"task_{task_id}")
        weight = int(cuda_task.get("weight") or sycl_task.get("weight") or 1)
        cuda_records = targets["cuda"]["grouped"].get(key, [])
        sycl_records = targets["sycl"]["grouped"].get(key, [])
        exemplar_mod = targets["cuda"]["modules"][key]
        cuda_best = targets["cuda"]["best"].get(key)
        sycl_best = targets["sycl"]["best"].get(key)
        cuda_ms = cuda_best[0] * 1e3 if cuda_best else None
        sycl_ms = sycl_best[0] * 1e3 if sycl_best else None
        row = {
            "task_id": task_id,
            "workload_key": key,
            "name": name,
            "operator_group": operator_group(name, exemplar_mod),
            "weight": weight,
            "cuda_trials": len(cuda_records),
            "cuda_failed": sum(mean_run_secs(record) is None for record in cuda_records),
            "cuda_best_ms": cuda_ms,
            "sycl_trials": len(sycl_records),
            "sycl_failed": sum(mean_run_secs(record) is None for record in sycl_records),
            "sycl_best_ms": sycl_ms,
            "sycl_over_cuda": sycl_ms / cuda_ms if cuda_ms and sycl_ms else None,
            "weighted_gap_ms": weight * (sycl_ms - cuda_ms) if cuda_ms and sycl_ms else None,
        }
        for target_name, best_record in (("cuda", cuda_best), ("sycl", sycl_best)):
            row[f"{target_name}_scheduled_tir"] = (
                write_scheduled_tir(out_dir, target_name, key, name, best_record[1])
                if best_record
                else ""
            )
        rows.append(row)

    fields = list(rows[0])
    with (out_dir / "paired_metaschedule_workloads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paired = [row for row in rows if row["sycl_over_cuda"] is not None]
    total_cuda_weighted = sum(
        row["weight"] * row["cuda_best_ms"] for row in rows if row["cuda_best_ms"] is not None
    )
    total_sycl_weighted = sum(
        row["weight"] * row["sycl_best_ms"] for row in rows if row["sycl_best_ms"] is not None
    )
    groups = {}
    for group in sorted({row["operator_group"] for row in paired}):
        selected = [row for row in paired if row["operator_group"] == group]
        cuda_sum = sum(row["weight"] * row["cuda_best_ms"] for row in selected)
        sycl_sum = sum(row["weight"] * row["sycl_best_ms"] for row in selected)
        groups[group] = {
            "workloads": len(selected),
            "weighted_cuda_ms": cuda_sum,
            "weighted_sycl_ms": sycl_sum,
            "sycl_over_cuda": sycl_sum / cuda_sum if cuda_sum else None,
            "weighted_gap_ms": sycl_sum - cuda_sum,
        }

    summary = {
        "workloads": len(rows),
        "paired_successful_workloads": len(paired),
        "cuda_workloads_with_success": sum(row["cuda_best_ms"] is not None for row in rows),
        "sycl_workloads_with_success": sum(row["sycl_best_ms"] is not None for row in rows),
        "cuda_trials": sum(row["cuda_trials"] for row in rows),
        "cuda_failed": sum(row["cuda_failed"] for row in rows),
        "sycl_trials": sum(row["sycl_trials"] for row in rows),
        "sycl_failed": sum(row["sycl_failed"] for row in rows),
        "matched_weighted_cuda_ms": sum(row["weight"] * row["cuda_best_ms"] for row in paired),
        "matched_weighted_sycl_ms": sum(row["weight"] * row["sycl_best_ms"] for row in paired),
        "all_finite_weighted_cuda_ms": total_cuda_weighted,
        "all_finite_weighted_sycl_ms": total_sycl_weighted,
        "groups": groups,
        "lowest_ratio_workload": min(paired, key=lambda row: row["sycl_over_cuda"]) if paired else None,
        "highest_ratio_workload": max(paired, key=lambda row: row["sycl_over_cuda"]) if paired else None,
    }
    cuda_sum = summary["matched_weighted_cuda_ms"]
    summary["matched_weighted_sycl_over_cuda"] = (
        summary["matched_weighted_sycl_ms"] / cuda_sum if cuda_sum else None
    )
    matched_gap = summary["matched_weighted_sycl_ms"] - summary["matched_weighted_cuda_ms"]
    for group in summary["groups"].values():
        group["share_of_matched_gap"] = group["weighted_gap_ms"] / matched_gap if matched_gap else None
    (out_dir / "paired_metaschedule_record_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
