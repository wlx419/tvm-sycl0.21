#!/usr/bin/env python3
"""Round-2 compact-model probe (MLP, CNN, Transformer, ViT-style) for CUDA and SYCL.

Same model definitions, parameters, inputs, references and tolerances as
experiments/relax_model_probe.py (inputs are generated with the "sycl" seed alias for both
targets, as in the published paired sweep). Adds --graph and raw latency samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "experiments"))
import relax_model_probe as base  # noqa: E402

import numpy as np  # noqa: E402
import tvm  # noqa: E402
from tvm import relax  # noqa: E402
from tvm.relax.frontend import nn  # noqa: E402


def default_pipeline(target):
    """Relax's default pipeline; plain OpenCL targets (not routed by TVM v0.21) use the generic GPU
    pipeline, as Metal does and as the SYCL pipeline extends."""
    if target.kind.name == "opencl" and "adreno" not in target.keys:
        return relax.backend.gpu_generic.get_default_pipeline(target)
    return relax.get_default_pipeline(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--target", default="sycl")
    ap.add_argument("--target-string", default="")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=50)
    ap.add_argument("--nsys-iters", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    case_seed = base.SEED + a.batch * 17 + sum(ord(c) for c in (a.model + "sycl"))
    rng = np.random.default_rng(case_seed)
    device_kind = base.runtime_device_kind(a.target)
    dev = tvm.device(device_kind, 0)
    target = base.make_target(a.target, dev, a.target_string)
    model, input_shape, ref_fn = base.make_case(a.model, a.batch)
    mod, params = model.export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
    named_np = base.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")
    row = {"model": a.model, "batch": a.batch, "target": a.target, "graph": a.graph, "resolved_target": str(target),
           "env_sync_each_launch": os.environ.get("TVM_SYCL_SYNC_EACH_LAUNCH", ""),
           "env_legacy_warp_sync": os.environ.get("TVM_SYCL_LEGACY_WARP_SYNC", "")}
    try:
        t0 = time.perf_counter()
        with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": bool(a.graph)}):
            ex = tvm.compile(mod, target=target, relax_pipeline=default_pipeline(target))
        row["compile_s"] = time.perf_counter() - t0
        vm = relax.VirtualMachine(ex, dev)
        x_dev = tvm.nd.array(x_np, dev)
        tvm_params = [tvm.nd.array(v, dev) for v in named_np.values()]
        out = vm["forward"](x_dev, *tvm_params).numpy()
        ref = ref_fn(x_np, named_np).astype("float32")
        row["max_abs_err"] = float(np.max(np.abs(out - ref)))
        row["status"] = "pass" if np.allclose(out, ref, atol=base.DEFAULT_ATOL, rtol=base.DEFAULT_RTOL) else "fail_tolerance"
        lat = base.timed_runs(lambda: vm["forward"](x_dev, *tvm_params), dev, a.warmup, a.measure)
        row.update(base.summarize(lat))
        row["latency_samples_ms"] = lat
        # Check again after the timed runs (command graphs: replayed and re-finalized graphs).
        out_after = vm["forward"](x_dev, *tvm_params).numpy()
        row["max_abs_err_after_timing"] = float(np.max(np.abs(out_after - ref)))
        if not np.allclose(out_after, ref, atol=base.DEFAULT_ATOL, rtol=base.DEFAULT_RTOL):
            row["status"] = "fail_tolerance_after_timing"
        # Inferences separated by host sleeps, so a Nsight Systems trace splits per inference.
        for _ in range(a.nsys_iters):
            time.sleep(0.03)
            vm["forward"](x_dev, *tvm_params)
            dev.sync()
    except Exception as exc:  # report, never hide
        import traceback
        row.update({"status": "runtime_error", "error": repr(exc), "traceback": traceback.format_exc()})
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(row, indent=1))
    print(a.target, a.model, f"B{a.batch}", "graph" if a.graph else "eager", row.get("status"),
          "p50=%s" % row.get("latency_p50_ms"), "abs=%s" % row.get("max_abs_err"), row.get("error", "")[:200])


if __name__ == "__main__":
    main()
