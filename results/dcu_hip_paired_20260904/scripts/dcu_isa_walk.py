#!/usr/bin/env python3
"""Dump the AMDGPU ISA and LLVM IR of the failing ROCm layer_norm kernel.

An earlier attempt grabbed the host x86 module by mistake. This walks the whole
module tree, reports each module's type key and kernels, and dumps the device
module that actually contains layer_norm.
"""
import argparse
import os
import re
import sys

import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax
from tvm.relax.frontend import nn

ROCM = ("rocm -keys=rocm,gpu -max_num_threads=256 "
        "-max_shared_memory_per_block=65536 -max_threads_per_block=1024 "
        "-mcpu=gfx906 -mtriple=amdgcn-amd-amdhsa-hcc -thread_warp_size={ws}")


def walk(mod, depth=0, seen=None):
    """Yield (depth, module) for the module and everything it imports."""
    if seen is None:
        seen = set()
    key = id(mod)
    if key in seen:
        return
    seen.add(key)
    yield depth, mod
    try:
        imported = list(mod.imported_modules)
    except Exception:  # noqa: BLE001
        imported = []
    for sub in imported:
        yield from walk(sub, depth + 1, seen)


def build(target, P, model_name, batch, alias):
    case_seed = P.SEED + batch * 17 + sum(ord(c) for c in (model_name + alias))
    rng = np.random.default_rng(case_seed)
    model, input_shape, _ = P.make_case(model_name, batch)
    mod, params = model.export_tvm(
        {"forward": {"x": nn.spec.Tensor(input_shape, "float32")}}
    )
    P.materialize_params(params, rng)
    pipe = relax.get_pipeline("static_shape_tuning", target=target, total_trials=0,
                              work_dir="/root/rocm_baseline/isa_work")
    mod = pipe(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    return tvm.compile(mod, target=target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="/workspace/supplement/experiments")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ws", type=int, default=64)
    ap.add_argument("--kernel", default="layer_norm")
    ap.add_argument("--dump-dir", default="/root/rocm_baseline/isa2")
    args = ap.parse_args()

    sys.path.insert(0, args.exp_dir)
    import relax_model_probe as P  # noqa: E402

    os.makedirs(args.dump_dir, exist_ok=True)
    target = tvm.target.Target(ROCM.format(ws=args.ws))
    ex = build(target, P, args.model, args.batch, "sycl")

    print(f"=== module tree (ws={args.ws}) ===")
    hits = []
    for depth, m in walk(ex.mod):
        try:
            tk = m.type_key
        except Exception:  # noqa: BLE001
            tk = "?"
        srcs = {}
        for fmt in ("asm", "llvm"):
            try:
                s = m.get_source(fmt)
            except Exception:  # noqa: BLE001
                s = ""
            srcs[fmt] = s or ""
        amdgpu = "amdgcn" in srcs["llvm"] or "amdgpu_kernel" in srcs["llvm"]
        # Match the kernel definition itself. A plain substring test also hits
        # other kernels that merely read a T_layer_norm buffer.
        defines = re.findall(r"amdgpu_kernel void @(\w+)", srcs["llvm"])
        owns = f"{args.kernel}_kernel" in defines
        if amdgpu:
            print(f"  {'  ' * depth}[{tk}] asm={len(srcs['asm'])} "
                  f"llvm={len(srcs['llvm'])} kernels={defines}")
        else:
            print(f"  {'  ' * depth}[{tk}] asm={len(srcs['asm'])} "
                  f"llvm={len(srcs['llvm'])} amdgpu=False")
        if amdgpu and owns:
            hits.append((tk, srcs))

    if not hits:
        print(f"\nno AMDGPU module containing {args.kernel}")
        return

    tk, srcs = hits[0]
    base = f"{args.dump_dir}/{args.kernel}_ws{args.ws}"
    for fmt in ("asm", "llvm"):
        with open(f"{base}.{'s' if fmt == 'asm' else 'll'}", "w") as fh:
            fh.write(srcs[fmt])
    print(f"\ndumped {base}.s ({len(srcs['asm'])} B) and {base}.ll "
          f"({len(srcs['llvm'])} B) from [{tk}]")

    asm = srcs["asm"]
    markers = ["ds_bpermute", "ds_permute", "ds_swizzle", "ds_read", "ds_write",
               "s_barrier", "v_rsq", "v_sqrt", "v_readlane", "v_readfirstlane",
               "v_mov_b32_dpp", "s_waitcnt lgkmcnt"]
    print("\n=== ISA markers ===")
    for m in markers:
        n = asm.count(m)
        if n:
            print(f"  {m:<20} {n}")


if __name__ == "__main__":
    main()
