"""Test equivalent math expressions for DCU lane-specific SYCL failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tvm

from dcu_math_diagnostic import run_unary


def compact(result):
    keys = (
        "size",
        "finite",
        "nan",
        "pos_inf",
        "neg_inf",
        "max_abs_err_finite",
        "max_rel_err_finite",
        "nonfinite_local_ids",
        "generated_math_lines",
    )
    return {key: result[key] for key in keys if key in result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    dev = tvm.device("sycl", args.device_id)
    values = np.full((256,), np.float32(1.0), dtype="float32")
    tanh_expected = np.tanh(values).astype("float32")
    normalize_expected = np.full_like(values, np.float32(0.9) / np.sqrt(np.float32(3.0)))

    probes = {
        "identity": run_unary(dev, "identity_probe", "x", values, values),
        "tanh_builtin": run_unary(dev, "tanh_builtin_probe", "T.tanh(x)", values, tanh_expected),
        "tanh_stable_exp": run_unary(
            dev,
            "tanh_stable_exp_probe",
            "T.Select(x >= T.float32(0), (T.float32(1) - T.exp(T.float32(-2) * x)) / (T.float32(1) + T.exp(T.float32(-2) * x)), (T.exp(T.float32(2) * x) - T.float32(1)) / (T.exp(T.float32(2) * x) + T.float32(1)))",
            values,
            tanh_expected,
        ),
        "divide_sqrt": run_unary(
            dev,
            "divide_sqrt_probe",
            "(x - T.float32(0.1)) / T.sqrt(T.float32(3.0))",
            values,
            normalize_expected,
        ),
        "multiply_reciprocal_sqrt": run_unary(
            dev,
            "multiply_reciprocal_sqrt_probe",
            "(x - T.float32(0.1)) * (T.float32(1) / T.sqrt(T.float32(3.0)))",
            values,
            normalize_expected,
        ),
        "multiply_sycl_rsqrt": run_unary(
            dev,
            "multiply_sycl_rsqrt_probe",
            '(x - T.float32(0.1)) * T.call_pure_extern("sycl::rsqrt", T.float32(3.0), dtype="float32")',
            values,
            normalize_expected,
        ),
        "divide_constant": run_unary(
            dev,
            "divide_constant_probe",
            "(x - T.float32(0.1)) / T.float32(1.7320508075688772)",
            values,
            normalize_expected,
        ),
    }

    result = {
        "device_id": args.device_id,
        "device_name": dev.device_name,
        "probes": {name: compact(value) for name, value in probes.items()},
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
