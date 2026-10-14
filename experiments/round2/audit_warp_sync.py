#!/usr/bin/env python3
"""does any evaluated SYCL configuration emit the warp-sync no-op?

Compiles each evaluated model configuration for SYCL with the build under test (no execution),
reads the generated SYCL source and counts, per kernel, the `storage_sync("warp")` sites that the
published code generator emitted as no-ops. Configurations:
  q3     ResNet-18 B1 224x224 with the published SYCL MetaSchedule database (Tables 4-7)
  sweep  ResNet-18/34/50 (224x224) and MLP/CNN/Transformer/ViT, B1/B8/B16, default pipeline
         with the NVIDIA SYCL target (Table 9)
  dcu    the same 21 cases with the static-shape pipeline, no trials (DLight) and the DCU target
         string (Tables 10-11); only the source text is inspected
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tvm  # noqa: E402
from tvm import dlight as dl  # noqa: E402
from tvm import relax  # noqa: E402
from tvm.relax.frontend import nn  # noqa: E402

import relax_model_probe as compact  # noqa: E402
import resnet_relax_probe as resnet  # noqa: E402

NV_SYCL = ("sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=1024 -max_shared_memory_per_block=49152 "
           "-max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32")
DCU_SYCL = ("sycl -keys=sycl,gpu -l2_cache_size_bytes=75497472 -max_num_threads=64 -max_shared_memory_per_block=49152 "
            "-max_threads_per_block=1024 -registers_per_block=65536 -texture_spatial_limit=16384 -thread_warp_size=32")
MARK = "warp storage_sync: no-op"


def resnet_mod(name, batch):
    model = resnet.fuse_resnet_batch_norm(resnet.make_model(name, 1000))
    x = torch.randn(batch, 3, 224, 224)
    ep = torch.export.export(model, (x,))
    return resnet.from_exported_program(ep, keep_params_as_input=False, unwrap_unit_return_tuple=True)


def compact_mod(name, batch):
    model, shape, _ = compact.make_case(name, batch)
    mod, _ = model.export_tvm({"forward": {"x": nn.spec.Tensor(shape, "float32")}})
    return mod


def lower(mod, target, mode, db_dir=""):
    if mode == "default":
        return tvm.compile(mod, target=target, relax_pipeline=relax.get_default_pipeline(target))
    kwargs = {"target": target, "total_trials": 0}
    if db_dir:
        kwargs["work_dir"] = db_dir
    mod = relax.get_pipeline("static_shape_tuning", **kwargs)(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                                      dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
    return tvm.compile(mod, target=target)


def sycl_source(ex):
    srcs, stack, seen = [], [ex.mod], set()
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        stack.extend(m.imported_modules)
        if m.type_key == "sycl":
            srcs.append(m.get_source())
    return "\n".join(srcs)


def count(src):
    per_kernel = {}
    for part in re.split(r"(?=// Function: )", src):
        m = re.match(r"// Function: (\S+)", part)
        if m:
            n = part.count(MARK)
            if n:
                per_kernel[m.group(1)] = n
    kernels = len(re.findall(r"// Function: ", src))
    return {"kernels": kernels, "kernels_with_warp_noop": len(per_kernel), "warp_noop_sites": sum(per_kernel.values()),
            "per_kernel": per_kernel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="q3,sweep,dcu")
    ap.add_argument("--q3-db", default="")
    ap.add_argument("--batches", default="1,8,16")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--src-dir", default="")
    a = ap.parse_args()
    out = Path(a.out_json)
    res = json.loads(out.read_text()) if out.exists() else {}
    cases = []
    sets = a.sets.split(",")
    if "q3" in sets:
        cases.append(("q3", "resnet18", 1, NV_SYCL, "static", a.q3_db))
    for tag, ts, mode in (("sweep", NV_SYCL, "default"), ("dcu", DCU_SYCL, "static")):
        if tag in sets:
            for m in ["resnet18", "resnet34", "resnet50", "mlp", "cnn", "transformer", "vit"]:
                for b in [int(x) for x in a.batches.split(",")]:
                    cases.append((tag, m, b, ts, mode, ""))
    for tag, m, b, ts, mode, db in cases:
        key = f"{tag}/{m}/b{b}"
        if key in res:
            continue
        try:
            mod = resnet_mod(m, b) if m.startswith("resnet") else compact_mod(m, b)
            src = sycl_source(lower(mod, tvm.target.Target(ts), mode, db))
            if a.src_dir:
                Path(a.src_dir).mkdir(parents=True, exist_ok=True)
                Path(a.src_dir, key.replace("/", "_") + ".sycl.cc").write_text(src)
            res[key] = count(src)
        except Exception as exc:  # report, never hide
            res[key] = {"error": repr(exc)}
        print(key, {k: v for k, v in res[key].items() if k != "per_kernel"}, flush=True)
        out.write_text(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
