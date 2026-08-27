#!/usr/bin/env python3
"""Combine paired MetaSchedule wall timing, Nsight timing, and launch geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def section_rows(lines: list[str], header_prefix: str) -> list[dict[str, str]]:
    start = next(index for index, line in enumerate(lines) if line.startswith(header_prefix))
    block = []
    for line in lines[start:]:
        if not line.strip() and block:
            break
        if line.strip():
            block.append(line)
    return list(csv.DictReader(block))


def parse_nsys(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    kernel_rows = section_rows(lines, "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns)")
    geometry_rows = [
        row
        for row in section_rows(lines, "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns)")
        if "GridXYZ" in row
    ]
    # The first matching header is the kernel summary. Find the geometry header explicitly.
    geometry_start = next(index for index, line in enumerate(lines) if "GridXYZ,BlockXYZ,Name" in line)
    geometry_block = []
    for line in lines[geometry_start:]:
        if not line.strip() and geometry_block:
            break
        if line.strip():
            geometry_block.append(line)
    geometry_rows = list(csv.DictReader(geometry_block))
    api_rows = section_rows(lines, "Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns)")
    kernel = kernel_rows[0]
    geometry = geometry_rows[0]
    launch = next(row for row in api_rows if row["Name"] == "cuLaunchKernel")
    return {
        "kernel_avg_us": float(kernel["Avg (ns)"]) / 1e3,
        "kernel_median_us": float(kernel["Med (ns)"]) / 1e3,
        "kernel_instances": int(kernel["Instances"]),
        "launch_api_avg_us": float(launch["Avg (ns)"]) / 1e3,
        "launch_api_median_us": float(launch["Med (ns)"]) / 1e3,
        "launch_api_calls": int(launch["Num Calls"]),
        "grid_xyz": " ".join(geometry["GridXYZ"].split()),
        "block_xyz": " ".join(geometry["BlockXYZ"].split()),
        "kernel_name": geometry["Name"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    pairs = []
    for profile_path in sorted((root / "operator_profiles").glob("*/profile_summary.json")):
        summary = json.loads(profile_path.read_text(encoding="utf-8"))
        key = summary["workload_key"]
        by_target = {}
        for target in ("cuda", "sycl"):
            row = {
                "workload_key": key,
                "target": target,
                **summary[target],
                **parse_nsys(profile_path.parent / "nsys" / f"{target}_stats.csv"),
            }
            rows.append(row)
            by_target[target] = row
        cuda = by_target["cuda"]
        sycl = by_target["sycl"]
        pairs.append(
            {
                "workload_key": key,
                "wall_p50_ratio": sycl["wall_p50_ms"] / cuda["wall_p50_ms"],
                "device_event_p50_ratio": sycl["device_event_p50_ms"]
                / cuda["device_event_p50_ms"],
                "nsys_kernel_median_ratio": sycl["kernel_median_us"] / cuda["kernel_median_us"],
                "launch_api_median_ratio": sycl["launch_api_median_us"]
                / cuda["launch_api_median_us"],
                "cuda_grid_xyz": cuda["grid_xyz"],
                "sycl_grid_xyz": sycl["grid_xyz"],
                "cuda_block_xyz": cuda["block_xyz"],
                "sycl_block_xyz": sycl["block_xyz"],
                "cuda_barrier_tokens": cuda["barrier_tokens"],
                "sycl_barrier_tokens": sycl["barrier_tokens"],
                "outputs_match": summary["output_comparison"]["allclose"],
            }
        )
    if not rows:
        raise RuntimeError(f"No operator profiles under {root}")
    write_csv(root / "paired_metaschedule_profile_metrics.csv", rows)
    write_csv(root / "paired_metaschedule_profile_pairs.csv", pairs)
    result = {"profiles": rows, "pairs": pairs}
    (root / "paired_metaschedule_profile_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
