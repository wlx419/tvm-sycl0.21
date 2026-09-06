#!/usr/bin/env python3
"""Decide whether the bad layer_norm columns are miscomputed or never written.

Evidence so far rules out the allreduce lowering: thread_warp_size=64 produces a
ds_bpermute warp-shuffle reduction and 32 produces a shared-memory reduction
with barriers, yet both give bit-identical wrong results in the same columns.
mean/var are per-row scalars, so a broken reduction would corrupt whole rows,
not fixed columns.

This test pre-fills layer_norm's output buffer with a sentinel before the call.
If the bad columns still hold the sentinel afterwards, the kernel never stored
to those lanes and the fault is a store/predication bug, not arithmetic. If they
hold Inf instead, the kernel did compute and store a nonfinite value.
"""
import argparse
import sys

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size={ws}")

SENTINEL = -12345.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/workspace/supplement/experiments")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ws", type=int, default=64)
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.exp_dir)
    import relax_model_probe as P  # noqa: E402

    target = tvm.target.Target(ROCM.format(ws=args.ws))
    dev = tvm.device("rocm", args.device_id)

    case_seed = P.SEED + args.batch * 17 + sum(ord(c) for c in (args.model + "sycl"))
    rng = np.random.default_rng(case_seed)
    model, input_shape, _ = P.make_case(args.model, args.batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    named_np = P.materialize_params(params, rng)
    x_np = (rng.standard_normal(size=input_shape) * 0.1).astype("float32")

    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir="/root/rocm_baseline/sent_work")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    state = {"done": False}

    def instrument(func, func_symbol, before_run, ret_value, *call_args):
        if func_symbol != "layer_norm" or state["done"]:
            return None
        tensors = [a for a in call_args if isinstance(a, tvm.nd.NDArray)]
        if len(tensors) < 4:
            return None
        out_buf = tensors[-1]
        if before_run:
            dev.sync()
            fill = np.full([int(d) for d in out_buf.shape], SENTINEL, dtype="float32")
            out_buf.copyfrom(fill)
            dev.sync()
            state["x"] = tensors[0].numpy()
            state["g"] = tensors[1].numpy()
            state["b"] = tensors[2].numpy()
            state["prefill_ok"] = bool((out_buf.numpy() == SENTINEL).all())
        else:
            dev.sync()
            state["y"] = out_buf.numpy()
            state["done"] = True
        return None

    vm.set_instrument(instrument)
    vm["forward"](tvm.nd.array(x_np, dev),
                  *[tvm.nd.array(a, dev) for a in named_np.values()])

    print(f"ws={args.ws}  prefill verified={state.get('prefill_ok')}")
    y = state.get("y")
    if y is None:
        print("layer_norm output not captured")
        return

    x, g, b = state["x"], state["g"], state["b"]
    flat = y.reshape(-1, y.shape[-1])
    ncol = flat.shape[1]

    untouched = flat == SENTINEL
    nonfinite = ~np.isfinite(flat)
    print(f"output {y.shape}: sentinel-left={int(untouched.sum())} "
          f"nonfinite={int(nonfinite.sum())} of {flat.size}")
    print(f"  row0 sentinel columns : {np.flatnonzero(untouched[0]).tolist()}")
    print(f"  row0 nonfinite columns: {np.flatnonzero(nonfinite[0]).tolist()}")

    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    ref = ((x - mu) / np.sqrt(var + 1e-5) * g + b).reshape(-1, ncol)
    ok = ~(untouched | nonfinite)
    if ok.any():
        print(f"  max abs err on stored finite elements: "
              f"{np.abs(flat[ok] - ref[ok]).max():.6g}")

    if untouched.any():
        print("\nVERDICT: those lanes were never stored to -> store/predication "
              "bug, not arithmetic.")
    elif nonfinite.any():
        print("\nVERDICT: the kernel stored a nonfinite value -> the computation "
              "itself produced Inf on those lanes.")
    else:
        print("\nVERDICT: output fully correct under prefill.")


if __name__ == "__main__":
    main()
