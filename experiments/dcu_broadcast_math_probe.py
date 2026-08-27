"""Probe broadcast normalization and erf alternatives on the DCU SYCL backend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tvm

from dcu_math_diagnostic import generated_math_lines, run_unary, summarize


def build_broadcast(name: str, expression: str):
    src = f"""@T.prim_func
def {name}(A: T.Buffer((256,), "float32"),
           Mean: T.Buffer((1,), "float32"),
           Var: T.Buffer((1,), "float32"),
           Out: T.Buffer((256,), "float32")):
    T.func_attr({{"tir.noalias": True}})
    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for tx in T.thread_binding(256, thread="threadIdx.x"):
            i = tx
            x = A[i]
            m = Mean[0]
            v = Var[0]
            Out[i] = {expression}
"""
    prim_func = tvm.script.from_source(src).with_attr("global_symbol", name)
    module = tvm.build(tvm.IRModule({name: prim_func}), target="sycl")
    return module, module[name]


def run_broadcast(dev, name: str, expression: str):
    values = np.full((256,), np.float32(1.0), dtype="float32")
    mean = np.array([0.1], dtype="float32")
    variance = np.array([3.0], dtype="float32")
    expected = ((values - mean[0]) / np.sqrt(variance[0] + np.float32(1e-5))).astype("float32")
    module, function = build_broadcast(name, expression)
    output = tvm.nd.array(np.zeros_like(values), dev)
    function(
        tvm.nd.array(values, dev),
        tvm.nd.array(mean, dev),
        tvm.nd.array(variance, dev),
        output,
    )
    dev.sync()
    result = summarize(output.numpy(), expected)
    result["generated_math_lines"] = generated_math_lines(module)
    return result


def compact(result):
    omitted = {"input", "actual", "expected"}
    return {key: value for key, value in result.items() if key not in omitted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    dev = tvm.device("sycl", args.device_id)
    base = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype="float32")
    erf_values = np.resize(base, 256).astype("float32")
    erf_expected = np.vectorize(math.erf)(erf_values).astype("float32")
    erf_approximation = (
        "T.Select(x >= T.float32(0), T.float32(1), T.float32(-1)) * "
        "(T.float32(1) - (((((T.float32(1.061405429) * "
        "(T.float32(1) / (T.float32(1) + T.float32(0.3275911) * T.abs(x))) "
        "+ T.float32(-1.453152027)) * "
        "(T.float32(1) / (T.float32(1) + T.float32(0.3275911) * T.abs(x))) "
        "+ T.float32(1.421413741)) * "
        "(T.float32(1) / (T.float32(1) + T.float32(0.3275911) * T.abs(x))) "
        "+ T.float32(-0.284496736)) * "
        "(T.float32(1) / (T.float32(1) + T.float32(0.3275911) * T.abs(x))) "
        "+ T.float32(0.254829592)) * "
        "(T.float32(1) / (T.float32(1) + T.float32(0.3275911) * T.abs(x))) "
        "* T.exp(-T.abs(x) * T.abs(x))))"
    )

    result = {
        "device_id": args.device_id,
        "device_name": dev.device_name,
        "broadcast_normalization": {
            "divide_sqrt": run_broadcast(
                dev,
                "broadcast_divide_sqrt",
                "(x - m) / T.sqrt(v + T.float32(0.00001))",
            ),
            "multiply_reciprocal_sqrt": run_broadcast(
                dev,
                "broadcast_multiply_reciprocal_sqrt",
                "(x - m) * (T.float32(1) / T.sqrt(v + T.float32(0.00001)))",
            ),
            "multiply_sycl_rsqrt": run_broadcast(
                dev,
                "broadcast_multiply_sycl_rsqrt",
                '(x - m) * T.call_pure_extern("sycl::rsqrt", v + T.float32(0.00001), dtype="float32")',
            ),
        },
        "erf": {
            "builtin": compact(
                run_unary(dev, "erf_builtin_probe", "T.erf(x)", erf_values, erf_expected)
            ),
            "explicit_sycl": compact(
                run_unary(
                    dev,
                    "erf_explicit_sycl_probe",
                    'T.call_pure_extern("sycl::erf", x, dtype="float32")',
                    erf_values,
                    erf_expected,
                )
            ),
            "double_precision_sycl": compact(
                run_unary(
                    dev,
                    "erf_double_precision_sycl_probe",
                    'T.Cast("float32", T.call_pure_extern("sycl::erf", T.Cast("float64", x), dtype="float64"))',
                    erf_values,
                    erf_expected,
                )
            ),
            "stable_exp_approximation": compact(
                run_unary(
                    dev,
                    "erf_stable_exp_probe",
                    erf_approximation,
                    erf_values,
                    erf_expected,
                )
            ),
        },
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
