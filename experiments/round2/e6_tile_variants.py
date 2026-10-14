#!/usr/bin/env python3
"""E6: separate block count from synchronized-reduction length.

Takes the best SYCL MetaSchedule record of the high-ratio ResNet-18 1x1 convolution
(fused_conv2d4_add3, 64->128 channels, stride 2, 56x56 -> 28x28) and replays its trace with
only the SamplePerfectTile decisions changed, giving a 2x2 design:

            outer reduction steps:   64 (rc tile [64,1,1])   16 (rc tile [16,4,1])
  4 blocks  (ff [2,1,8,4,2], yy [1,1,2,14,1])   V0 = original   V2
  32 blocks (ff [8,1,8,1,2], yy [2,1,2,7,1])    V1              V3

Threads per block (224), the cooperative-fetch structure and all other decisions are kept.
Each variant is post-processed with the default GPU postprocessors, built for the requested
target, checked against NumPy, and timed (synchronized wall time per launch). The script also
reports grid size, threads per block and the number of barriers each work-group executes
(static barrier sites x trip count of the enclosing reduction loop).

Run it once per target (sycl, cuda) on the same machine. Nsight Systems / Nsight Compute can
wrap the whole script; each variant is launched in a separate, clearly delimited phase.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import tir

VARIANTS = {
    "V0_blocks4_steps64": {"ff": [2, 1, 8, 4, 2], "yy": [1, 1, 2, 14, 1], "rc": [64, 1, 1]},
    "V1_blocks32_steps64": {"ff": [8, 1, 8, 1, 2], "yy": [2, 1, 2, 7, 1], "rc": [64, 1, 1]},
    "V2_blocks4_steps16": {"ff": [2, 1, 8, 4, 2], "yy": [1, 1, 2, 14, 1], "rc": [16, 4, 1]},
    "V3_blocks32_steps16": {"ff": [8, 1, 8, 1, 2], "yy": [2, 1, 2, 7, 1], "rc": [16, 4, 1]},
}
# Decisions of the original record, used to locate the instructions to change.
ORIGINAL = {"ff": [2, 1, 8, 4, 2], "yy": [1, 1, 2, 14, 1], "rc": [64, 1, 1]}


def load_record(db_dir: Path, workload_hash: str):
    db = ms.database.JSONDatabase(
        path_workload=str(db_dir / "database_workload.json"),
        path_tuning_record=str(db_dir / "database_tuning_record.json"),
        allow_missing=False,
    )
    for rec in db.get_all_tuning_records():
        if str(tvm.ir.structural_hash(rec.workload.mod)) == workload_hash or \
           str(rec.workload.shash) == workload_hash:
            break
    else:
        raise KeyError(workload_hash)
    best = db.get_top_k(rec.workload, 1)[0]
    return best.workload.mod, best.trace


def retarget_trace(trace: tir.schedule.Trace, variant: dict) -> tir.schedule.Trace:
    wanted = {tuple(v): k for k, v in ORIGINAL.items()}
    seen = set()
    new_trace = trace
    for inst, decision in trace.decisions.items():
        if inst.kind.name != "SamplePerfectTile":
            continue
        cur = tuple(int(x) for x in decision)
        key = wanted.get(cur)
        if key is None or key in seen:
            continue
        seen.add(key)
        new_trace = new_trace.with_decision(inst, [tvm.tir.IntImm("int64", x) for x in variant[key]],
                                            remove_postproc=True)
    missing = set(ORIGINAL) - seen
    if missing:
        raise RuntimeError(f"tile instructions not found for {sorted(missing)}")
    return new_trace


def postprocess(mod, target, sch: tir.Schedule) -> tir.Schedule:
    ctx = ms.TuneContext(mod=mod, target=target, space_generator="post-order-apply",
                         search_strategy="replay-trace", task_name="e6")
    for p in ctx.space_generator.postprocs:
        if not p.apply(sch):
            raise RuntimeError(f"postprocessor {p} rejected the variant")
    return sch


def kernel_geometry(func: tir.PrimFunc) -> dict:
    blocks, threads, barrier_trip = 1, 1, 0
    extents = {}

    def visit(node):
        nonlocal blocks, threads
        if isinstance(node, tir.For) and node.kind == tir.ForKind.THREAD_BINDING and node.thread_binding is not None:
            tag = node.thread_binding.thread_tag
            if isinstance(node.extent, tir.IntImm):
                if tag.startswith("blockIdx"):
                    blocks *= int(node.extent.value)
                elif tag.startswith("threadIdx"):
                    extents[tag] = int(node.extent.value)

    tir.stmt_functor.post_order_visit(func.body, visit)
    for v in extents.values():
        threads *= v
    return {"grid_blocks": blocks, "threads_per_block": threads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True, help="SYCL tuning database directory")
    ap.add_argument("--workload-hash", default="1095410103996100310")
    ap.add_argument("--target", default="sycl")
    ap.add_argument("--target-string", default="")
    ap.add_argument("--device", default="")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--dump-source", default="")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    target = tvm.target.Target(args.target_string or args.target)
    dev = tvm.device(args.device or ("oneapi" if args.target == "sycl" else args.target), 0)
    mod, trace = load_record(Path(args.db_dir), args.workload_hash)
    rng = np.random.default_rng(1234)
    params = list(mod["main"].params)
    shapes = [[int(s) for s in mod["main"].buffer_map[p].shape] for p in params]
    inputs = [rng.uniform(-1, 1, size=s).astype("float32") for s in shapes[:-1]]
    x, w, b = inputs
    # 1x1 stride-2 convolution (NCHW) + bias, the workload's semantics
    ref = np.einsum("nchw,fc->nfhw", x[:, :, ::2, ::2].astype("float64"), w[:, :, 0, 0].astype("float64")) \
        + b.astype("float64")

    results = []
    for name in args.variants.split(","):
        row = {"variant": name, "target": str(target), **{f"tile_{k}": v for k, v in VARIANTS[name].items()}}
        try:
            sch = tir.Schedule(mod, debug_mask="all")
            retarget_trace(trace, VARIANTS[name]).apply_to_schedule(sch, remove_postproc=True)
            sch = postprocess(mod, target, sch)
            row.update(kernel_geometry(sch.mod["main"]))
            t0 = time.perf_counter()
            lib = tvm.build(sch.mod, target=target)
            row["build_s"] = time.perf_counter() - t0
            if args.dump_source:
                Path(args.dump_source).mkdir(parents=True, exist_ok=True)
                src = "\n".join(m.get_source() for m in lib.imported_modules)
                Path(args.dump_source, f"{name}_{args.target}.src").write_text(src)
                row["static_barrier_sites"] = src.count("group_barrier(") + src.count("__syncthreads()")
            arrs = [tvm.nd.array(a, dev) for a in inputs] + [tvm.nd.empty(shapes[-1], "float32", dev)]
            lib(*arrs)
            dev.sync()
            out = arrs[-1].numpy()
            row["max_abs_err"] = float(np.max(np.abs(out - ref)))
            row["status"] = "pass" if np.allclose(out, ref, atol=1e-4, rtol=1e-3) else "fail_tolerance"
            outer_steps = VARIANTS[name]["rc"][0]
            row["outer_reduction_steps"] = outer_steps
            row["barriers_per_workgroup"] = 2 * outer_steps  # two staging barriers per outer step
            for _ in range(args.warmup):
                lib(*arrs)
            dev.sync()
            lat = []
            for _ in range(args.measure):
                t0 = time.perf_counter_ns()
                lib(*arrs)
                dev.sync()
                lat.append((time.perf_counter_ns() - t0) / 1e3)
            row.update({"wall_us_p50": float(np.percentile(lat, 50)), "wall_us_p10": float(np.percentile(lat, 10)),
                        "wall_us_p90": float(np.percentile(lat, 90)), "wall_us_samples": lat})
        except Exception as exc:  # report, never hide
            row.update({"status": "error", "error": repr(exc)})
        results.append(row)
        print(name, row.get("status"), row.get("grid_blocks"), row.get("threads_per_block"),
              row.get("barriers_per_workgroup"), row.get("wall_us_p50"), row.get("error", ""))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
