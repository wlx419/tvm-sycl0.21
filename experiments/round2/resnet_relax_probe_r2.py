#!/usr/bin/env python3
"""Round-2 ResNet probe for TVM v0.21 Relax (CUDA and SYCL targets).

Extends experiments/resnet_relax_probe.py with the controls needed for the round-2
interventions. Model import, BatchNorm folding, inputs, reference and tolerances are
identical to the original probe.

New options
  --tuning-driver custom   tune with an explicit seed, runner policy and search space
                           instead of the static_shape_tuning pipeline's defaults
  --tune-seed N            MetaSchedule seed (the published runs were unseeded)
  --trials-per-iter N      candidates measured per task iteration (default 8, as in the
                           published static_shape_tuning runs)
  --runner-max-uses N      PopenPool maximum_process_uses for the LocalRunner
                           (1 = fresh measurement process per candidate)
  --runner-recycle-on-error  replace the measurement process only after a failed candidate
  --min-blocks-json P      reject candidates whose grid has fewer blocks than the
                           per-workload threshold in P ({"workload": blocks, "*": default})
  --apply-db-dir D         apply-only from an existing database directory (cross-replay:
                           e.g. a CUDA database applied while compiling for SYCL)
  --dump-source D          write generated device source to D (warp-sync audit, kernel names)
  --nsys-iters N           after timing, run N iterations separated by 30 ms host sleeps so
                           an Nsight Systems trace can be split per inference
Runtime switches are read from the environment by the round-2 SYCL overlay:
  TVM_SYCL_SYNC_EACH_LAUNCH=1   restore the per-launch wait (published behaviour)
  TVM_SYCL_LEGACY_WARP_SYNC=1   restore the warp-sync no-op lowering (published behaviour)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "experiments"))
import resnet_relax_probe as base  # noqa: E402  (same model/import/reference code)

import torch  # noqa: E402
import tvm  # noqa: E402
from tvm import dlight as dl  # noqa: E402
from tvm import meta_schedule as ms  # noqa: E402
from tvm import relax  # noqa: E402
from tvm.relax import transform as rt  # noqa: E402


# --------------------------------------------------------------------------- search-space controls
def grid_blocks(func: tvm.tir.PrimFunc) -> int:
    blocks = 1
    found = False

    def visit(node):
        nonlocal blocks, found
        if isinstance(node, tvm.tir.For) and node.kind == tvm.tir.ForKind.THREAD_BINDING:
            tag = node.thread_binding.thread_tag if node.thread_binding is not None else ""
            if tag.startswith("blockIdx") and isinstance(node.extent, tvm.tir.IntImm):
                blocks *= int(node.extent.value)
                found = True

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return blocks if found else 0


@ms.utils.derived_object
class BlockCountFilter(ms.postproc.PyPostproc):
    """Reject candidates whose grid exposes fewer blocks than a per-workload threshold."""

    def __init__(self, thresholds: Dict[str, int]) -> None:
        super().__init__()
        self.thresholds = dict(thresholds)
        self.minimum = 0
        self.task_name = ""

    def _initialize_with_tune_context(self, context: "ms.TuneContext") -> None:
        self.task_name = str(context.task_name or "")
        name = self.task_name
        self.minimum = int(self.thresholds.get(name, self.thresholds.get("*", 0)))

    def apply(self, sch: tvm.tir.Schedule) -> bool:
        if self.minimum <= 0:
            return True
        func = sch.mod["main"]
        return grid_blocks(func) >= self.minimum

    def clone(self) -> "BlockCountFilter":
        return BlockCountFilter(self.thresholds)


def make_space(min_blocks: Optional[Dict[str, int]]):
    postprocs = list(ms.Postproc.create("cuda"))
    if min_blocks:
        postprocs.append(BlockCountFilter(min_blocks))
    # Rule kind "cuda" for both targets. Async-copy pipelining inside MultiLevelTiling is still
    # gated by the target's `arch` attribute, so it is controlled through the target string.
    return ms.space_generator.PostOrderApply(
        sch_rules="cuda", postprocs=postprocs, mutator_probs="cuda"
    )


@ms.utils.derived_object
class RecycleOnErrorRunner(ms.runner.PyRunner):
    """LocalRunner that replaces its measurement process after a failed candidate.

    A device fault (e.g. a misaligned or illegal address) leaves the CUDA context of the
    measurement process unusable; a reused process then fails every later candidate at its
    first allocation, which is how most SYCL trials of the published tuning were lost.
    LocalRunner.run measures synchronously, so each result is known before the next one starts.
    """

    def __init__(self, watchdog_sec: float = 120.0) -> None:
        super().__init__()
        self.inner = ms.runner.LocalRunner()
        self.recycled = 0
        self.watchdog_sec = watchdog_sec
        self.measured = 0

    def run(self, runner_inputs):
        import threading  # pylint: disable=import-outside-toplevel

        futures = []
        for runner_input in runner_inputs:
            # LocalRunner's own timeout is enforced by a Python timer inside the worker, which
            # cannot fire while the worker is blocked in a native device wait (e.g. a kernel that
            # never finishes); this watchdog kills the worker from the parent instead.
            box = {}
            start = time.perf_counter()
            thread = threading.Thread(
                target=lambda: box.setdefault("f", self.inner.run([runner_input])[0]), daemon=True
            )
            thread.start()
            thread.join(self.watchdog_sec)
            timed_out = thread.is_alive()
            if timed_out:
                kill_pool_workers(self.inner.pool)
                thread.join(60)
            future = box.get("f")
            if future is None:
                future = ms.runner.local_runner.LocalRunnerFuture(
                    res=None, error_message=f"RecycleOnErrorRunner: no result after {self.watchdog_sec} s"
                )
            elapsed = time.perf_counter() - start
            self.measured += 1
            err = future.result().error_msg
            if err is not None:
                # The next submission starts a fresh worker process.
                kill_pool_workers(self.inner.pool)
                self.recycled += 1
                print(f"[runner] candidate {self.measured} failed after {elapsed:.1f} s"
                      f"{' (watchdog)' if timed_out else ''}; measurement process replaced"
                      f" ({self.recycled}): {err.strip().splitlines()[-1][:160]}", flush=True)
            elif elapsed > float(os.environ.get("RUNNER_SLOW_SEC", "10")):
                print(f"[runner] candidate {self.measured} took {elapsed:.1f} s", flush=True)
            futures.append(future)
        return futures


def kill_pool_workers(pool) -> None:
    """Kill the worker processes of a PopenPoolExecutor (it has no shutdown method)."""
    with pool._lock:  # pylint: disable=protected-access
        for worker in pool._worker_map.values():  # pylint: disable=protected-access
            worker.kill()
        pool._worker_map.clear()  # pylint: disable=protected-access


def make_runner(max_uses: int, recycle_on_error: bool = False):
    if recycle_on_error:
        return RecycleOnErrorRunner()
    runner = ms.runner.LocalRunner()
    if max_uses > 0:
        from tvm.contrib.popen_pool import PopenPoolExecutor

        kill_pool_workers(runner.pool)
        runner.pool = PopenPoolExecutor(
            max_workers=1,
            timeout=runner.timeout_sec,
            initializer=None,
            maximum_process_uses=max_uses,
        )
    return runner


def custom_tuning_pipeline(mod, target, args):
    with target:
        mod = tvm.transform.Sequential(
            [rt.DecomposeOpsForInference(), rt.CanonicalizeBindings(), relax.get_pipeline("zero")]
        )(mod)
        if args.total_trials > 0:
            min_blocks = None
            if args.min_blocks_json:
                min_blocks = json.loads(Path(args.min_blocks_json).read_text())
            ms.relax_integration.tune_relax(
                mod=mod,
                params={},
                target=target,
                work_dir=args.work_dir,
                max_trials_global=args.total_trials,
                # The pass used by the published runs (MetaScheduleTuneIRMod ->
                # tvm.meta_schedule.tune_relax) measures 8 candidates per task iteration;
                # Python's tune_relax defaults to 64, which with 19 tasks and 1,024 trials
                # ends before every task has been visited once.
                num_trials_per_iter=args.trials_per_iter,
                runner=make_runner(args.runner_max_uses, args.runner_recycle_on_error),
                space=make_space(min_blocks),
                seed=args.tune_seed,
            )
        mod = rt.MetaScheduleApplyDatabase(args.apply_db_dir or args.work_dir)(mod)
    return mod


def scheduled_function_report(mod) -> Dict[str, List[str]]:
    sched, unsched = [], []
    for gv, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc):
            (sched if func.attrs is not None and "tir.is_scheduled" in func.attrs else unsched).append(gv.name_hint)
    return {"db_scheduled": sorted(sched), "dlight_fallback": sorted(unsched)}


def default_pipeline(target):
    """Relax's default pipeline; plain OpenCL targets (not routed by TVM v0.21) use the generic GPU
    pipeline, as Metal does and as the SYCL pipeline extends."""
    if target.kind.name == "opencl" and "adreno" not in target.keys:
        return relax.backend.gpu_generic.get_default_pipeline(target)
    return relax.get_default_pipeline(target)


def apply_pipeline_r2(mod, target, args):
    report = {}
    if args.pipeline == "default":
        return mod, default_pipeline(target), report
    if args.tuning_driver == "custom" or args.apply_db_dir:
        mod = custom_tuning_pipeline(mod, target, args)
    else:
        kwargs = {"target": target, "total_trials": args.total_trials}
        if args.work_dir:
            kwargs["work_dir"] = args.work_dir
        mod = relax.get_pipeline("static_shape_tuning", **kwargs)(mod)
    report = scheduled_function_report(mod)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(), dl.gpu.GeneralReduction(), dl.gpu.Fallback()
        )(mod)
    return mod, None, report


def dump_sources(ex, out_dir: Path, tag: str) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"warp_sync_noops": 0, "group_barriers": 0, "files": 0}
    lib = ex.mod if hasattr(ex, "mod") else ex
    stack = [lib]
    seen = set()
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        stack.extend(list(m.imported_modules))
        if m.type_key in ("sycl", "cuda"):
            src = m.get_source()
            p = out_dir / f"{tag}_{m.type_key}_{counts['files']}.src"
            p.write_text(src, encoding="utf-8")
            counts["files"] += 1
            counts["warp_sync_noops"] += src.count("warp storage_sync: no-op")
            counts["group_barriers"] += src.count("group_barrier(")
    return counts


# --------------------------------------------------------------------------- one case
def run_one(model_name: str, batch: int, image_size: int, num_classes: int, target_kind: str, args):
    torch.manual_seed(base.SEED + batch * 17 + image_size + len(model_name))
    np.random.seed(base.SEED + batch * 17 + image_size + len(model_name))
    input_shape = (batch, 3, image_size, image_size)
    started = time.perf_counter()
    stage = "device_setup"
    device_kind = base.runtime_device_kind(target_kind)
    row = {"model": model_name, "batch": batch, "target": target_kind, "runtime_device": device_kind,
           "device_id": args.device_id, "input_shape": str(input_shape), "pipeline": args.pipeline,
           "tuning_driver": args.tuning_driver, "tune_seed": args.tune_seed,
           "trials_per_iter": args.trials_per_iter,
           "runner_max_uses": args.runner_max_uses,
           "runner_recycle_on_error": args.runner_recycle_on_error,
           "min_blocks_json": args.min_blocks_json,
           "apply_db_dir": args.apply_db_dir, "work_dir": args.work_dir,
           "graph": bool(args.graph),
           "env_sync_each_launch": os.environ.get("TVM_SYCL_SYNC_EACH_LAUNCH", ""),
           "env_jit_cache": os.environ.get("TVM_SYCL_JIT_CACHE", ""), "env_jit_jobs": os.environ.get("TVM_SYCL_JIT_JOBS", ""),
           "env_legacy_warp_sync": os.environ.get("TVM_SYCL_LEGACY_WARP_SYNC", "")}
    try:
        dev = tvm.device(device_kind, args.device_id)
        target = base.make_target(target_kind, dev, args.target_string)
        row["resolved_target"] = str(target)
        stage = "model_setup"
        model = base.make_model(model_name, num_classes)
        x_torch = torch.randn(*input_shape, dtype=torch.float32)
        model = base.fuse_resnet_batch_norm(model)
        with torch.no_grad():
            ref = model(x_torch).detach().cpu().numpy()
        stage = "frontend_import"
        exported = torch.export.export(model, (x_torch,))
        mod = base.from_exported_program(exported, keep_params_as_input=False, unwrap_unit_return_tuple=True)
        stage = "tvm_compile"
        t0 = time.perf_counter()
        # --graph: RewriteCUDAGraph captures static regions; CUDA replays them as CUDA graphs and
        # the SYCL pipeline as SYCL command graphs (vm.builtin.sycl_graph.*).
        with tvm.transform.PassContext(config={"relax.backend.use_cuda_graph": bool(args.graph)}):
            mod, pipeline, report = apply_pipeline_r2(mod, target, args)
            ex = tvm.compile(mod, target=target) if pipeline is None else tvm.compile(mod, target=target, relax_pipeline=pipeline)
        row["compile_s"] = time.perf_counter() - t0
        row.update({k: ",".join(v) for k, v in report.items()})
        row["n_db_scheduled"] = len(report.get("db_scheduled", []))
        row["n_dlight_fallback"] = len(report.get("dlight_fallback", []))
        if args.dump_source:
            row.update({f"src_{k}": v for k, v in dump_sources(ex, Path(args.dump_source),
                                                                f"{target_kind}_{model_name}_b{batch}").items()})
        stage = "vm_execute"
        vm = relax.VirtualMachine(ex, dev)
        x_dev = tvm.nd.array(x_torch.numpy(), dev)
        out = base.to_numpy(vm["main"](x_dev))
        max_abs, max_rel = base.max_errors(out, ref)
        row.update({"max_abs_err": max_abs, "max_rel_err": max_rel,
                    "status": "pass" if np.allclose(out, ref, atol=args.atol, rtol=args.rtol) else "fail_tolerance"})
        stage = "timing"
        lat = base.timed_runs(lambda: vm["main"](x_dev), dev, args.warmup, args.measure)
        row.update(base.summarize(lat))
        row["latency_samples_ms"] = lat
        if args.nsys_iters > 0:
            stage = "nsys_iters"
            base.sync(dev)
            for _ in range(args.nsys_iters):
                time.sleep(0.03)
                vm["main"](x_dev)
                base.sync(dev)
        row["total_s"] = time.perf_counter() - started
        return row
    except Exception as exc:  # report, never hide
        row.update({"status": "runtime_error", "failure_stage": stage, "error": repr(exc),
                    "traceback": traceback.format_exc(), "total_s": time.perf_counter() - started})
        return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet18")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--target", default="sycl")
    p.add_argument("--target-string", default="")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=50)
    p.add_argument("--pipeline", choices=["default", "static_shape_tuning"], default="static_shape_tuning")
    p.add_argument("--tuning-driver", choices=["builtin", "custom"], default="custom")
    p.add_argument("--total-trials", type=int, default=0)
    p.add_argument("--trials-per-iter", type=int, default=8)
    p.add_argument("--tune-seed", type=int, default=20261012)
    p.add_argument("--runner-max-uses", type=int, default=0)
    p.add_argument("--runner-recycle-on-error", action="store_true",
                   help="replace the measurement process only after a failed candidate")
    p.add_argument("--min-blocks-json", default="")
    p.add_argument("--work-dir", default="")
    p.add_argument("--apply-db-dir", default="")
    p.add_argument("--dump-source", default="")
    p.add_argument("--nsys-iters", type=int, default=0)
    p.add_argument("--graph", action="store_true", help="capture static regions as CUDA/SYCL graphs")
    p.add_argument("--atol", type=float, default=base.DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=base.DEFAULT_RTOL)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()
    row = run_one(args.model, args.batch, args.image_size, args.num_classes, args.target, args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(row, indent=1, sort_keys=True), encoding="utf-8")
    print(f"{args.target} {args.model} B{args.batch} {row.get('status')} p50={row.get('latency_p50_ms')} "
          f"abs={row.get('max_abs_err')} db={row.get('n_db_scheduled')} fallback={row.get('n_dlight_fallback')}")
    if row.get("status") == "runtime_error":
        print(row.get("traceback", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
