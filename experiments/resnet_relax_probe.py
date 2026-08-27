#!/usr/bin/env python3
"""Torchvision ResNet inference correctness probe for TVM Relax.

The script imports torchvision ResNet models through torch.export, compiles the
resulting Relax module for a selected TVM target, and compares the VM output
with the PyTorch CPU output for the same random input and parameters.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import tvm
from torch.export import export
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm.runtime.ndarray import NDArray as TVMNDArray


SEED = 321
DEFAULT_ATOL = 2e-4
DEFAULT_RTOL = 2e-3


MODEL_FACTORY: Dict[str, Callable[..., torch.nn.Module]] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
}


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(xs[f])
    return float(xs[f] * (c - k) + xs[c] * (k - f))


def summarize(values_ms: Sequence[float]) -> Dict[str, float]:
    if not values_ms:
        return {}
    return {
        "latency_mean_ms": float(np.mean(values_ms)),
        "latency_p50_ms": percentile(values_ms, 0.50),
        "latency_p90_ms": percentile(values_ms, 0.90),
        "latency_p99_ms": percentile(values_ms, 0.99),
    }


def sync(dev) -> None:
    if hasattr(dev, "sync") and callable(dev.sync):
        dev.sync()


def to_numpy(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, TVMNDArray):
        return obj.numpy()
    if hasattr(obj, "numpy") and callable(obj.numpy):
        return obj.numpy()
    if hasattr(obj, "asnumpy") and callable(obj.asnumpy):
        return obj.asnumpy()
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return to_numpy(obj[0])
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        try:
            if len(obj) == 1:
                return to_numpy(obj[0])
        except Exception:
            pass
    return np.array(obj)


def timed_runs(fn: Callable[[], object], dev, warmup: int, measure: int) -> List[float]:
    for _ in range(warmup):
        fn()
    sync(dev)
    values = []
    for _ in range(measure):
        t0 = time.perf_counter_ns()
        fn()
        sync(dev)
        values.append((time.perf_counter_ns() - t0) / 1e6)
    return values


def make_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model {model_name}")
    return MODEL_FACTORY[model_name](weights=None, num_classes=num_classes).eval()


def fuse_resnet_batch_norm(model: ResNet) -> ResNet:
    """Fold inference BatchNorm state into adjacent convolutions before export."""

    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = torch.nn.Identity()
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            if not isinstance(block, (BasicBlock, Bottleneck)):
                raise TypeError(f"Unsupported ResNet block: {type(block)!r}")
            block.conv1 = fuse(block.conv1, block.bn1)
            block.bn1 = torch.nn.Identity()
            block.conv2 = fuse(block.conv2, block.bn2)
            block.bn2 = torch.nn.Identity()
            if isinstance(block, Bottleneck):
                block.conv3 = fuse(block.conv3, block.bn3)
                block.bn3 = torch.nn.Identity()
            if block.downsample is not None:
                block.downsample[0] = fuse(block.downsample[0], block.downsample[1])
                block.downsample[1] = torch.nn.Identity()
    return model


def apply_pipeline(mod, target, args):
    if args.pipeline == "default":
        return mod, relax.get_default_pipeline(target)
    if args.pipeline == "static_shape_tuning":
        kwargs = {"target": target, "total_trials": args.total_trials}
        if args.work_dir:
            kwargs["work_dir"] = args.work_dir
        pipeline = relax.get_pipeline("static_shape_tuning", **kwargs)
        mod = pipeline(mod)
        with target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
        return mod, None
    raise ValueError(f"Unknown pipeline {args.pipeline}")


def max_errors(actual: np.ndarray, expected: np.ndarray) -> Tuple[float, float]:
    abs_err = np.abs(actual - expected)
    max_abs = float(np.max(abs_err)) if abs_err.size else 0.0
    max_rel = float(np.max(abs_err / np.maximum(np.abs(expected), 1e-6))) if abs_err.size else 0.0
    return max_abs, max_rel


def make_target(target_kind: str, dev, target_string: str):
    if target_string:
        return tvm.target.Target(target_string)
    try:
        return tvm.target.Target.from_device(dev)
    except (KeyError, ValueError, RuntimeError, tvm.error.TVMError):
        return tvm.target.Target(target_kind)


def runtime_device_kind(target_kind: str) -> str:
    if target_kind != "sycl":
        return target_kind
    if tvm.get_global_func("device_api.oneapi", allow_missing=True) is not None:
        return "oneapi"
    return "sycl"


def run_one(model_name: str, batch: int, image_size: int, num_classes: int, target_kind: str, args):
    torch.manual_seed(SEED + batch * 17 + image_size + len(model_name))
    np.random.seed(SEED + batch * 17 + image_size + len(model_name))
    input_shape = (batch, 3, image_size, image_size)
    started = time.perf_counter()
    stage = "device_setup"
    device_kind = runtime_device_kind(target_kind)
    target = None

    try:
        dev = tvm.device(device_kind, args.device_id)
        target = make_target(target_kind, dev, args.target_string)
        stage = "model_setup"
        model = make_model(model_name, num_classes)
        x_torch = torch.randn(*input_shape, dtype=torch.float32)
        with torch.no_grad():
            unfused_ref = model(x_torch).detach().cpu().numpy()

        fusion_max_abs = 0.0
        if not args.no_fold_batch_norm:
            stage = "batch_norm_fusion"
            model = fuse_resnet_batch_norm(model)
            with torch.no_grad():
                fused_ref = model(x_torch).detach().cpu().numpy()
            fusion_max_abs, _ = max_errors(fused_ref, unfused_ref)
            if not np.allclose(fused_ref, unfused_ref, atol=args.atol, rtol=args.rtol):
                raise RuntimeError(
                    f"Conv-BatchNorm fusion changed output: max abs {fusion_max_abs:.6g}"
                )
            ref = fused_ref
        else:
            ref = unfused_ref

        stage = "frontend_import"
        export_t0 = time.perf_counter()
        params = []
        if args.frontend == "onnx":
            if args.keep_params_as_input:
                raise ValueError("ONNX mode currently requires bound parameters")
            import onnx
            from tvm.relax.frontend.onnx import from_onnx

            buffer = io.BytesIO()
            torch.onnx.export(
                model,
                x_torch,
                buffer,
                input_names=["input"],
                output_names=["output"],
                opset_version=17,
                do_constant_folding=True,
            )
            onnx_model = onnx.load_model_from_string(buffer.getvalue())
            mod = from_onnx(
                onnx_model,
                shape_dict={"input": list(input_shape)},
                dtype_dict={"input": "float32"},
                keep_params_in_input=False,
            )
        else:
            exported = export(model, (x_torch,))
            if not hasattr(exported, "run_decompositions"):
                type(exported).run_decompositions = lambda self: self
            mod = from_exported_program(
                exported,
                keep_params_as_input=args.keep_params_as_input,
                unwrap_unit_return_tuple=True,
            )
            if args.keep_params_as_input:
                mod, detached = relax.frontend.detach_params(mod)
                params = detached["main"]
        export_s = time.perf_counter() - export_t0

        stage = "tvm_compile"
        compile_t0 = time.perf_counter()
        mod, compile_pipeline = apply_pipeline(mod, target, args)
        if compile_pipeline is None:
            ex = tvm.compile(mod, target=target)
        else:
            ex = tvm.compile(mod, target=target, relax_pipeline=compile_pipeline)
        compile_s = time.perf_counter() - compile_t0

        stage = "vm_setup"
        vm = relax.VirtualMachine(ex, dev)
        x_dev = tvm.nd.array(x_torch.detach().cpu().numpy(), dev)
        tvm_params = [tvm.nd.array(p, dev) for p in params]

        stage = "vm_execute"
        out = to_numpy(vm["main"](x_dev, *tvm_params))

        if out.shape != ref.shape:
            return {
                "model": model_name,
                "batch": batch,
                "target": target_kind,
                "runtime_device": device_kind,
                "device_id": args.device_id,
                "status": "fail_shape",
                "input_shape": str(input_shape),
                "output_shape": str(tuple(out.shape)),
                "reference_shape": str(tuple(ref.shape)),
                "pipeline": args.pipeline,
                "frontend": args.frontend,
                "fallback_schedule": args.pipeline == "static_shape_tuning",
                "parameter_mode": "inputs" if args.keep_params_as_input else "bound",
                "batch_norm_folded": not args.no_fold_batch_norm,
                "resolved_target": str(target),
                "total_s": time.perf_counter() - started,
            }

        max_abs, max_rel = max_errors(out, ref)
        allclose = bool(np.allclose(out, ref, atol=args.atol, rtol=args.rtol))

        stage = "timing"
        latency = timed_runs(lambda: vm["main"](x_dev, *tvm_params), dev, args.warmup, args.measure)

        row = {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "runtime_device": device_kind,
            "device_id": args.device_id,
            "status": "pass" if allclose else "fail_tolerance",
            "input_shape": str(input_shape),
            "output_shape": str(tuple(out.shape)),
            "num_params": len(list(model.parameters())),
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "fusion_max_abs_err": fusion_max_abs,
            "atol": args.atol,
            "rtol": args.rtol,
            "pipeline": args.pipeline,
            "frontend": args.frontend,
            "fallback_schedule": args.pipeline == "static_shape_tuning",
            "work_dir": args.work_dir or "",
            "parameter_mode": "inputs" if args.keep_params_as_input else "bound",
            "batch_norm_folded": not args.no_fold_batch_norm,
            "resolved_target": str(target),
            "export_s": export_s,
            "compile_s": compile_s,
            "total_s": time.perf_counter() - started,
        }
        row.update(summarize(latency))
        return row
    except Exception as exc:
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "runtime_device": device_kind,
            "device_id": args.device_id,
            "status": "runtime_error",
            "input_shape": str(input_shape),
            "failure_stage": stage,
            "pipeline": args.pipeline,
            "frontend": args.frontend,
            "fallback_schedule": args.pipeline == "static_shape_tuning",
            "work_dir": args.work_dir or "",
            "parameter_mode": "inputs" if args.keep_params_as_input else "bound",
            "batch_norm_folded": not args.no_fold_batch_norm,
            "resolved_target": str(target) if target is not None else "",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "total_s": time.perf_counter() - started,
        }


def parse_csv_list(text: str, cast: Callable[[str], object] = str) -> List[object]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def run_case_subprocess(script_path: Path, model_name: str, batch: int, target_kind: str, args):
    case_work_dir = ""
    if args.work_dir:
        case_work_dir = str(Path(args.work_dir) / f"{target_kind}_{model_name}_b{batch}")
        Path(case_work_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--models",
        model_name,
        "--batches",
        str(batch),
        "--targets",
        target_kind,
        "--device-id",
        str(args.device_id),
        "--image-size",
        str(args.image_size),
        "--num-classes",
        str(args.num_classes),
        "--warmup",
        str(args.warmup),
        "--measure",
        str(args.measure),
        "--pipeline",
        args.pipeline,
        "--frontend",
        args.frontend,
        "--total-trials",
        str(args.total_trials),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
    ]
    if args.target_string:
        cmd.extend(["--target-string", args.target_string])
    if args.keep_params_as_input:
        cmd.append("--keep-params-as-input")
    if args.no_fold_batch_norm:
        cmd.append("--no-fold-batch-norm")
    if case_work_dir:
        cmd.extend(["--work-dir", case_work_dir])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.case_timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "device_id": args.device_id,
            "status": "timeout",
            "error": f"case exceeded {args.case_timeout_s} s: {exc}",
        }
    if proc.returncode != 0:
        return {
            "model": model_name,
            "batch": batch,
            "target": target_kind,
            "device_id": args.device_id,
            "status": "runtime_error",
            "error": (proc.stderr or proc.stdout)[-4000:],
        }
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line[len("RESULT_JSON=") :])
    return {
        "model": model_name,
        "batch": batch,
        "target": target_kind,
        "device_id": args.device_id,
        "status": "parse_error",
        "error": f"worker produced no result: {proc.stdout[-4000:]}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--models", default="resnet18,resnet34,resnet50")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--targets", default="sycl")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--target-string",
        default="",
        help="Optional complete TVM target string; otherwise detect properties from the device.",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=10)
    parser.add_argument("--pipeline", choices=["default", "static_shape_tuning"], default="default")
    parser.add_argument("--frontend", choices=["torch-export", "onnx"], default="torch-export")
    parser.add_argument("--total-trials", type=int, default=0)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--keep-params-as-input", action="store_true")
    parser.add_argument("--no-fold-batch-norm", action="store_true")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--case-timeout-s", type=int, default=3600)
    parser.add_argument("--out-json", default="resnet_relax_probe.json")
    parser.add_argument("--out-csv", default="resnet_relax_probe.csv")
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    batches = parse_csv_list(args.batches, int)
    targets = parse_csv_list(args.targets)
    if args.worker:
        if len(models) != 1 or len(batches) != 1 or len(targets) != 1:
            raise ValueError("worker mode requires exactly one model, batch, and target")
        row = run_one(models[0], batches[0], args.image_size, args.num_classes, targets[0], args)
        print("RESULT_JSON=" + json.dumps(row, sort_keys=True))
        return

    results = []
    script_path = Path(__file__).resolve()
    for target_kind in targets:
        for model_name in models:
            for batch in batches:
                row = run_case_subprocess(script_path, model_name, batch, target_kind, args)
                results.append(row)
                msg = (
                    f"{target_kind:>4s} {model_name:>8s} B{batch:<3d} "
                    f"{row.get('status')} abs={row.get('max_abs_err', 'NA')} "
                    f"p50={row.get('latency_p50_ms', 'NA')}"
                )
                if "error" in row:
                    msg += f" err={str(row['error'])[:180]}"
                print(msg, flush=True)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": {
                    "seed": SEED,
                    "warmup": args.warmup,
                    "measure": args.measure,
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "image_size": args.image_size,
                    "num_classes": args.num_classes,
                    "device_id": args.device_id,
                    "reference": "PyTorch CPU",
                    "pipeline": args.pipeline,
                    "frontend": args.frontend,
                    "fallback_schedule": args.pipeline == "static_shape_tuning",
                    "total_trials": args.total_trials,
                    "parameter_mode": "inputs" if args.keep_params_as_input else "bound",
                    "batch_norm_folded": not args.no_fold_batch_norm,
                },
                "results": results,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    fieldnames = sorted({k for row in results for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
