#!/usr/bin/env python3
"""Rebuild and profile one workload from paired MetaSchedule databases."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import meta_schedule as ms


def key_for_mod(mod) -> str:
    return str(int(tvm.ir.structural_hash(mod)) & ((1 << 64) - 1))


def mean_run_secs(record) -> float | None:
    if not record.run_secs:
        return None
    values = [float(value) for value in record.run_secs]
    if not values or not all(math.isfinite(value) and value < 1e9 for value in values):
        return None
    return statistics.mean(values)


def best_record(work_dir: Path, workload_key: str):
    database = ms.database.JSONDatabase(
        path_workload=str(work_dir / "database_workload.json"),
        path_tuning_record=str(work_dir / "database_tuning_record.json"),
    )
    candidates = []
    for record in database.get_all_tuning_records():
        if key_for_mod(record.workload.mod) != workload_key:
            continue
        value = mean_run_secs(record)
        if value is not None:
            candidates.append((value, record))
    if not candidates:
        raise RuntimeError(f"No finite record for workload {workload_key} in {work_dir}")
    return min(candidates, key=lambda item: item[0])


def device_for(target_name: str):
    return tvm.device("oneapi" if target_name == "sycl" else "cuda", 0)


def module_source(module) -> str:
    sources = []
    stack = list(module.imported_modules)
    while stack:
        current = stack.pop()
        try:
            source = current.get_source()
        except (AttributeError, tvm.error.TVMError):
            source = ""
        if source:
            sources.append(source)
        stack.extend(current.imported_modules)
    return "\n\n".join(sources)


def make_host_args(args_info, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    arrays = []
    for info in args_info:
        shape = tuple(int(value) for value in info.shape)
        dtype = np.dtype(str(info.dtype))
        if np.issubdtype(dtype, np.floating):
            value = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 7, size=shape, dtype=dtype)
        else:
            raise TypeError(f"Unsupported argument dtype: {dtype}")
        arrays.append(value)
    return arrays


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype="float64"), q))


def run_target(
    target_name: str,
    work_dir: Path,
    workload_key: str,
    host_args: list[np.ndarray] | None,
    warmup: int,
    measure: int,
    output_dir: Path,
) -> tuple[dict[str, object], list[np.ndarray], list[np.ndarray]]:
    log_best_s, record = best_record(work_dir, workload_key)
    candidate = record.as_measure_candidate()
    if host_args is None:
        host_args = make_host_args(candidate.args_info, seed=20260813)

    scheduled_tir = candidate.sch.mod.script()
    module = tvm.build(candidate.sch.mod, target=record.target)
    dev = device_for(target_name)
    arrays = [tvm.nd.array(value.copy(), dev) for value in host_args]
    function = module["main"]

    for _ in range(warmup):
        function(*arrays)
    dev.sync()

    wall_ms = []
    for _ in range(measure):
        started = time.perf_counter_ns()
        function(*arrays)
        dev.sync()
        wall_ms.append((time.perf_counter_ns() - started) / 1e6)

    evaluator = module.time_evaluator("main", dev, number=1, repeat=measure)
    event_ms = [float(value) * 1e3 for value in evaluator(*arrays).results]
    dev.sync()
    outputs = [array.numpy() for array in arrays]
    source = module_source(module)

    (output_dir / f"{target_name}_scheduled_tir.py").write_text(scheduled_tir + "\n", encoding="utf-8")
    suffix = "cu" if target_name == "cuda" else "cpp"
    (output_dir / f"{target_name}_generated.{suffix}").write_text(source + "\n", encoding="utf-8")
    result = {
        "target": target_name,
        "resolved_target": str(record.target),
        "log_best_ms": log_best_s * 1e3,
        "wall_p50_ms": percentile(wall_ms, 50),
        "wall_p90_ms": percentile(wall_ms, 90),
        "device_event_p50_ms": percentile(event_ms, 50),
        "device_event_p90_ms": percentile(event_ms, 90),
        "barrier_tokens": source.count("group_barrier") + source.count("__syncthreads"),
        "generated_source_bytes": len(source.encode("utf-8")),
        "argument_shapes": [list(value.shape) for value in host_args],
        "argument_dtypes": [str(value.dtype) for value in host_args],
    }
    return result, host_args, outputs


def compare_outputs(cuda_outputs: list[np.ndarray], sycl_outputs: list[np.ndarray]) -> dict[str, object]:
    rows = []
    all_close = True
    for index, (cuda_value, sycl_value) in enumerate(zip(cuda_outputs, sycl_outputs)):
        diff = np.abs(cuda_value.astype("float64") - sycl_value.astype("float64"))
        scale = np.maximum(np.abs(cuda_value.astype("float64")), 1e-12)
        close = bool(np.allclose(cuda_value, sycl_value, atol=2e-4, rtol=2e-3))
        all_close = all_close and close
        rows.append(
            {
                "argument": index,
                "allclose": close,
                "max_abs_error": float(diff.max(initial=0.0)),
                "max_rel_error": float((diff / scale).max(initial=0.0)),
            }
        )
    return {"allclose": all_close, "arguments": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Experiment root containing tune/cuda and tune/sycl")
    parser.add_argument("workload_key")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--only-target", choices=("cuda", "sycl"))
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (args.out_dir or root / "operator_profiles" / args.workload_key).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.only_target:
        result, _, _ = run_target(
            args.only_target,
            root / "tune" / args.only_target,
            args.workload_key,
            None,
            args.warmup,
            args.measure,
            output_dir,
        )
        path = output_dir / f"{args.only_target}_profile_summary.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    cuda, host_args, cuda_outputs = run_target(
        "cuda",
        root / "tune" / "cuda",
        args.workload_key,
        None,
        args.warmup,
        args.measure,
        output_dir,
    )
    sycl, _, sycl_outputs = run_target(
        "sycl",
        root / "tune" / "sycl",
        args.workload_key,
        host_args,
        args.warmup,
        args.measure,
        output_dir,
    )
    result = {
        "workload_key": args.workload_key,
        "protocol": {
            "warmup": args.warmup,
            "measure": args.measure,
            "wall": "host perf_counter around one launch followed by device synchronization",
            "device_event": "TVM target timer; the SYCL path brackets dispatch with profiling events",
        },
        "cuda": cuda,
        "sycl": sycl,
        "ratios": {
            "log_best": sycl["log_best_ms"] / cuda["log_best_ms"],
            "wall_p50": sycl["wall_p50_ms"] / cuda["wall_p50_ms"],
            "device_event_p50": sycl["device_event_p50_ms"] / cuda["device_event_p50_ms"],
        },
        "output_comparison": compare_outputs(cuda_outputs, sycl_outputs),
    }
    (output_dir / "profile_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["output_comparison"]["allclose"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
