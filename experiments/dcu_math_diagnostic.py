"""Diagnose non-finite LayerNorm and GELU results on the DCU SYCL backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import tvm

from tvm_sycl_extended_probe import (
    build_layer_norm_mean,
    build_layer_norm_normalize,
    build_layer_norm_var,
)


def build_elementwise(name: str, expr: str, length: int) -> tuple[Any, Any]:
    src = f"""@T.prim_func
def {name}(A: T.Buffer(({length},), "float32"),
           Out: T.Buffer(({length},), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding(({length} + 255) // 256, thread="blockIdx.x"):
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            i = bx * 256 + tx
            if i < {length}:
                x = A[i]
                Out[i] = {expr}
"""
    prim_func = tvm.script.from_source(src).with_attr("global_symbol", name)
    module = tvm.build(tvm.IRModule({name: prim_func}), target="sycl")
    return module, module[name]


def build_explicit_layer_norm(rows: int, cols: int) -> tuple[Any, Any]:
    total = rows * cols
    blocks = (total + 255) // 256
    src = f"""@T.prim_func
def explicit_layer_norm(A: T.Buffer(({rows}, {cols}), "float32"),
                        Mean: T.Buffer(({rows},), "float32"),
                        Var: T.Buffer(({rows},), "float32"),
                        Out: T.Buffer(({rows}, {cols}), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding({blocks}, thread="blockIdx.x"):
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            idx = bx * 256 + tx
            if idx < {total}:
                i = idx // {cols}
                j = idx - i * {cols}
                denom = T.call_pure_extern(
                    "sycl::sqrt", Var[i] + T.float32(0.00001), dtype="float32"
                )
                Out[i, j] = (A[i, j] - Mean[i]) / denom
"""
    prim_func = tvm.script.from_source(src).with_attr("global_symbol", "explicit_layer_norm")
    module = tvm.build(tvm.IRModule({"explicit_layer_norm": prim_func}), target="sycl")
    return module, module["explicit_layer_norm"]


def summarize(actual: np.ndarray, expected: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(actual)
    finite_count = int(np.count_nonzero(finite))
    result: Dict[str, Any] = {
        "size": int(actual.size),
        "finite": finite_count,
        "nan": int(np.count_nonzero(np.isnan(actual))),
        "pos_inf": int(np.count_nonzero(np.isposinf(actual))),
        "neg_inf": int(np.count_nonzero(np.isneginf(actual))),
    }
    nonfinite_indices = np.flatnonzero(~finite)
    if nonfinite_indices.size:
        result["nonfinite_indices_first64"] = nonfinite_indices[:64].tolist()
        result["nonfinite_local_ids"] = np.unique(nonfinite_indices % 256).tolist()
    if finite_count:
        diff = np.abs(actual[finite] - expected[finite])
        denom = np.maximum(np.abs(expected[finite]), np.float32(1e-7))
        result["max_abs_err_finite"] = float(np.max(diff))
        result["max_rel_err_finite"] = float(np.max(diff / denom))
        result["actual_min_finite"] = float(np.min(actual[finite]))
        result["actual_max_finite"] = float(np.max(actual[finite]))
    return result


def run_unary(
    dev: Any,
    name: str,
    expression: str,
    values: np.ndarray,
    expected: np.ndarray,
) -> Dict[str, Any]:
    module, function = build_elementwise(name, expression, values.size)
    out = tvm.nd.array(np.zeros_like(values), dev)
    function(tvm.nd.array(values, dev), out)
    dev.sync()
    actual = out.numpy()
    result = summarize(actual, expected)
    result["input"] = values.tolist()
    result["actual"] = actual.tolist()
    result["expected"] = expected.tolist()
    result["generated_math_lines"] = generated_math_lines(module)
    return result


def generated_math_lines(module: Any) -> list[str]:
    sources: Iterable[str] = []
    imported = getattr(module, "imported_modules", [])
    if imported:
        sources = [submodule.get_source() for submodule in imported]
    else:
        try:
            sources = [module.get_source()]
        except Exception:
            sources = []
    tokens = ("sqrt", "tanh", "exp", "native_")
    return [
        line.strip()
        for source in sources
        for line in source.splitlines()
        if any(token in line.lower() for token in tokens)
    ]


def diagnose_layer_norm(dev: Any) -> Dict[str, Any]:
    rows, cols = 128, 768
    rng = np.random.default_rng(0)
    values = rng.uniform(-3.0, 3.0, size=(rows, cols)).astype("float32")
    expected_mean = np.mean(values, axis=1, dtype="float32")
    expected_var = np.mean(
        (values - expected_mean[:, None]) * (values - expected_mean[:, None]),
        axis=1,
        dtype="float32",
    )
    expected_out = (
        (values - expected_mean[:, None]) / np.sqrt(expected_var[:, None] + np.float32(1e-5))
    ).astype("float32")

    mean_mod = build_layer_norm_mean((rows, cols), 256)
    var_mod = build_layer_norm_var((rows, cols), 256)
    norm_mod = build_layer_norm_normalize((rows, cols), 256)
    input_arr = tvm.nd.array(values, dev)
    mean_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
    var_arr = tvm.nd.array(np.zeros((rows,), dtype="float32"), dev)
    out_arr = tvm.nd.array(np.zeros((rows, cols), dtype="float32"), dev)

    mean_mod(input_arr, mean_arr)
    dev.sync()
    actual_mean = mean_arr.numpy()
    var_mod(input_arr, mean_arr, var_arr)
    dev.sync()
    actual_var = var_arr.numpy()
    norm_mod(input_arr, mean_arr, var_arr, out_arr)
    dev.sync()
    actual_out = out_arr.numpy()
    explicit_mod, explicit_norm = build_explicit_layer_norm(rows, cols)
    explicit_out_arr = tvm.nd.array(np.zeros((rows, cols), dtype="float32"), dev)
    explicit_norm(input_arr, mean_arr, var_arr, explicit_out_arr)
    dev.sync()
    explicit_actual_out = explicit_out_arr.numpy()

    sqrt_input = (expected_var[:16] + np.float32(1e-5)).astype("float32")
    sqrt_probe = run_unary(
        dev,
        "sqrt_probe",
        "T.sqrt(x)",
        sqrt_input,
        np.sqrt(sqrt_input).astype("float32"),
    )
    reciprocal_sqrt_probe = run_unary(
        dev,
        "reciprocal_sqrt_probe",
        "T.float32(1) / T.sqrt(x)",
        sqrt_input,
        (np.float32(1) / np.sqrt(sqrt_input)).astype("float32"),
    )
    reciprocal_probe = run_unary(
        dev,
        "reciprocal_probe",
        "T.float32(1) / x",
        sqrt_input,
        (np.float32(1) / sqrt_input).astype("float32"),
    )
    explicit_sqrt_probe = run_unary(
        dev,
        "explicit_sqrt_probe",
        'T.call_pure_extern("sycl::sqrt", x, dtype="float32")',
        sqrt_input,
        np.sqrt(sqrt_input).astype("float32"),
    )

    return {
        "mean": summarize(actual_mean, expected_mean),
        "variance": summarize(actual_var, expected_var),
        "normalize": summarize(actual_out, expected_out),
        "explicit_sycl_normalize": summarize(explicit_actual_out, expected_out),
        "sqrt_probe": sqrt_probe,
        "reciprocal_sqrt_probe": reciprocal_sqrt_probe,
        "reciprocal_probe": reciprocal_probe,
        "explicit_sqrt_probe": explicit_sqrt_probe,
        "normalize_generated_math_lines": generated_math_lines(norm_mod),
        "explicit_normalize_generated_math_lines": generated_math_lines(explicit_mod),
    }


def diagnose_gelu(dev: Any) -> Dict[str, Any]:
    values = np.array([-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0], dtype="float32")
    tanh_expected = np.tanh(values).astype("float32")
    inner_expected = (
        np.float32(0.7978845608028654)
        * (values + np.float32(0.044715) * values * values * values)
    ).astype("float32")
    gelu_expected = (np.float32(0.5) * values * (np.float32(1) + np.tanh(inner_expected))).astype("float32")

    tanh_probe = run_unary(dev, "tanh_probe", "T.tanh(x)", values, tanh_expected)
    explicit_tanh_probe = run_unary(
        dev,
        "explicit_tanh_probe",
        'T.call_pure_extern("sycl::tanh", x, dtype="float32")',
        values,
        tanh_expected,
    )
    inner_probe = run_unary(
        dev,
        "gelu_polynomial_probe",
        "T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x)",
        values,
        inner_expected,
    )
    gelu_probe = run_unary(
        dev,
        "fused_gelu_probe",
        "T.float32(0.5) * x * (T.float32(1) + T.tanh(T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x)))",
        values,
        gelu_expected,
    )
    explicit_gelu_probe = run_unary(
        dev,
        "explicit_gelu_probe",
        'T.float32(0.5) * x * (T.float32(1) + T.call_pure_extern("sycl::tanh", T.float32(0.7978845608028654) * (x + T.float32(0.044715) * x * x * x), dtype="float32"))',
        values,
        gelu_expected,
    )
    lane_values = np.full((256,), np.float32(1.0), dtype="float32")
    tanh_lane_probe = run_unary(
        dev,
        "tanh_lane_probe",
        "T.tanh(x)",
        lane_values,
        np.tanh(lane_values).astype("float32"),
    )
    explicit_tanh_lane_probe = run_unary(
        dev,
        "explicit_tanh_lane_probe",
        'T.call_pure_extern("sycl::tanh", x, dtype="float32")',
        lane_values,
        np.tanh(lane_values).astype("float32"),
    )
    sqrt_lane_values = np.full((256,), np.float32(3.0), dtype="float32")
    sqrt_lane_probe = run_unary(
        dev,
        "sqrt_lane_probe",
        "T.sqrt(x)",
        sqrt_lane_values,
        np.sqrt(sqrt_lane_values).astype("float32"),
    )

    return {
        "tanh_probe": tanh_probe,
        "explicit_tanh_probe": explicit_tanh_probe,
        "tanh_lane_probe": tanh_lane_probe,
        "explicit_tanh_lane_probe": explicit_tanh_lane_probe,
        "sqrt_lane_probe": sqrt_lane_probe,
        "polynomial_inner_probe": inner_probe,
        "fused_gelu_probe": gelu_probe,
        "explicit_sycl_gelu_probe": explicit_gelu_probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    dev = tvm.device("sycl", args.device_id)
    if not dev.exist:
        raise RuntimeError(f"tvm.device('sycl', {args.device_id}) does not exist")

    result = {
        "device_id": args.device_id,
        "device_name": dev.device_name,
        "layer_norm": diagnose_layer_norm(dev),
        "gelu": diagnose_gelu(dev),
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
