"""Summarize paired TVM-SYCL and oneDNN NVIDIA operator runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


OP_ORDER = ("matmul", "row_sum", "softmax", "conv2d_nhwc")


def summarize(values: Iterable[float]) -> Dict[str, float]:
    xs = list(values)
    if not xs:
        raise ValueError("no latency samples")
    return {
        "median_ms": statistics.median(xs),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def load_onednn(input_dir: Path) -> Dict[str, List[dict]]:
    rows: Dict[str, List[dict]] = {op: [] for op in OP_ORDER}
    for path in sorted(input_dir.glob("onednn_cuda_vendor_run*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows[row["op"]].append(row)
    return rows


def load_tvm(input_dir: Path) -> Dict[str, List[dict]]:
    rows: Dict[str, List[dict]] = {op: [] for op in OP_ORDER}
    for path in sorted(input_dir.glob("tvm_sycl_run*_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[row["op"]].append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    onednn = load_onednn(args.input_dir)
    tvm = load_tvm(args.input_dir)
    output = []
    for op in OP_ORDER:
        one_rows = onednn[op]
        tvm_rows = tvm[op]
        if not one_rows or not tvm_rows:
            raise ValueError(f"missing paired runs for {op}")
        if any(row["status"] != "pass" for row in one_rows + tvm_rows):
            raise ValueError(f"non-passing result for {op}")

        one_stats = summarize(float(row["p50_ms"]) for row in one_rows)
        tvm_stats = summarize(float(row["latency_p50_ms"]) for row in tvm_rows)
        output.append(
            {
                "op": op,
                "shape": one_rows[0]["shape"],
                "runs": len(one_rows),
                "tvm_sycl_median_ms": tvm_stats["median_ms"],
                "tvm_sycl_min_ms": tvm_stats["min_ms"],
                "tvm_sycl_max_ms": tvm_stats["max_ms"],
                "onednn_cuda_median_ms": one_stats["median_ms"],
                "onednn_cuda_min_ms": one_stats["min_ms"],
                "onednn_cuda_max_ms": one_stats["max_ms"],
                "onednn_over_tvm": one_stats["median_ms"] / tvm_stats["median_ms"],
                "onednn_max_abs_err": max(float(row["max_abs"]) for row in one_rows),
                "tvm_max_abs_err": max(float(row["max_abs_err"]) for row in tvm_rows),
            }
        )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    with args.output_stem.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
