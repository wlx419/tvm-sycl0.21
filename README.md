# TVM-SYCL: a SYCL 2020 backend for Apache TVM v0.21

This repository contains the source overlay, experiment entry points, raw
records, derived summaries, and generated figures for a SYCL 2020 backend for
Apache TVM v0.21. The overlay adds a SYCL code-generation target, a SYCL
runtime module and device API, and the associated target metadata and build
integration.

The backend is evaluated on an NVIDIA RTX 4090 through the oneAPI CUDA adapter
and on a Sugon DCU Z100L through the HIP adapter.

## Version 2.0

v2.0 changes the backend's code generation, runtime, schedule integration and tuning measurement.
Each change is listed with the switch that restores the v1 behaviour in
[`source_overlay/README.md`](source_overlay/README.md). The v2.0 scripts are in
[`experiments/round2/`](experiments/round2/README.md) and their results in
[`results/round2/`](results/round2/README.md); the v1 material below is unchanged. On the RTX 4090,
ResNet-18 B1 tuned with equal budgets and identical settings runs at 1.03-1.06x of TVM-CUDA
(v1: 2.61x), the seven-model sweep at 0.99-1.09x for the ResNets, and the launch-bound compact
models at 1.06-2.14x without and 1.10-1.26x with command graphs; built against DPC++ v7.1.1, the
compact models run at 1.06-1.11x without graphs.

The backend descends from the `sycl` branch of RELOAD22/tvm (commit `5b36163df4`), ported to TVM
v0.21 and changed as described in the overlay README; 937 of its 2,177 code lines (43%) are
identical to that branch (`experiments/round2/analysis/lineage_compare.py`).

## Quick Start

The backend is distributed as an overlay over an unmodified Apache TVM v0.21
tree rather than as a fork:

```bash
git clone --recursive --branch v0.21.0 https://github.com/apache/tvm.git
cd tvm

# Copy the changed files over the checkout, preserving relative paths.
rsync -a --exclude README.md --exclude CMakeLists.txt.patch \
      /path/to/tvm-sycl0.21/source_overlay/ .

# Register cmake/modules/SYCL.cmake in the root CMakeLists.txt.
git apply /path/to/tvm-sycl0.21/source_overlay/CMakeLists.txt.patch

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=/path/to/llvm-config \
  -DUSE_CUDA=ON -DUSE_RPC=ON \
  -DUSE_SYCL=/path/to/icpx
cmake --build build -j
```

See [`source_overlay/README.md`](source_overlay/README.md) for the full list of
integration changes and the runtime environment variables required for
SYCL-on-NVIDIA (`ONEAPI_DEVICE_SELECTOR=cuda:*`) and SYCL-on-DCU
(`USE_SYCL_HIP`, `--offload-arch=gfx906`).

## Repository Layout

| Path | Contents |
| --- | --- |
| `source_overlay/` | TVM v0.21 files changed for the SYCL backend (v2.0) |
| `experiments/` | Run, analysis, profiling, and figure-generation entry points (v1) |
| `experiments/round2/` | Run, build and analysis scripts of v2.0 |
| `results/` | Raw records and derived summaries for every reported experiment (v1) |
| `results/round2/` | Raw records and summaries of v2.0 |
| `figures/` | Generated figures summarizing the v1 results |

## License

The overlay is derived from Apache TVM and is released under the Apache
License, Version 2.0; see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`NOTICE.tvm`](NOTICE.tvm). Modified upstream files retain their original
Apache Software Foundation license headers. This repository is an independent
research artifact and is not endorsed by or affiliated with the Apache Software
Foundation.

## Compiler and Scheduling Scope (v1)

All model experiments use TVM v0.21 Relax, TensorIR, and the Relax VM. The
focused ResNet-18 B1 comparison tunes CUDA and SYCL with MetaSchedule under the
same nominal 1,024-trial budget. The broader seven-model batch sweep uses the
default Relax scheduling pipeline to cover all models and batch sizes at
manageable cost; it is reported as correctness and robustness evidence rather
than tuned performance. The independent DCU seven-model sweep uses the
static-shape Relax pipeline with DLight scheduling and no search trials and is
reported as portability, correctness, and batch-robustness evidence.

## Evidence Map (v1)

- `results/paired_resnet18_metaschedule_formal1024/` contains the formal paired
  CUDA/SYCL MetaSchedule databases, three independent complete-model runs per
  target, unsuccessful-record census, matched-workload analysis, generated
  kernels, scheduled TIR, Nsight Systems operator profiles, and Nsight Compute
  occupancy and bandwidth counters.
- `results/seven_model_relax_4090/` contains scheduling coverage for seven B1
  models and the paired CUDA/SYCL B={1,8,16} sweep. All 42 backend runs pass
  reference-output validation.
- `results/operator_correctness/` contains NVIDIA and DCU tensor-operator
  correctness records.
- `results/dcu_validation_20260819/` contains the final dual-card DCU operator
  regression (32/32 device-configuration runs), the seven-model B={1,8,16}
  sweep (21/21 runs), math-lowering diagnostics, and the tested source snapshot.
- `results/dcu_hip_paired_20260904/` contains the paired TVM-HIP/SYCL DCU sweep
  in which both backends run from one `libtvm.so`, together with the LayerNorm
  math-lowering diagnostics and generated AMDGPU ISA that explain why the
  Transformer and ViT-style points have no valid HIP baseline.
- `results/alternative_paths_4090/` contains the IREE timing audit and the
  verified oneDNN NVIDIA vendor-backend comparison.
- `source_overlay/` contains the TVM v0.21 files changed for SYCL code
  generation, runtime execution, target registration, Relax routing, and
  MetaSchedule integration.
- `experiments/` contains the corresponding run, analysis, profiling,
  summarization, and figure-generation entry points.

## Primary Paired Result (v1)

The formal ResNet-18 experiment uses torchvision ResNet-18 with BatchNorm
folded before `torch.export`, input `(1,3,224,224)`, identical model parameters
and random seed, and matched target metadata (`max_num_threads=1024`,
`thread_warp_size=32`). MetaSchedule extracts 19 workloads and serializes 1,029
records for each target. Both targets obtain finite records for 15 workloads
and use DLight fallback for four.

Across three independent database-reuse processes, the median synchronized
complete-model p50 is 0.782 ms for CUDA and 2.038 ms for SYCL, a 2.606x ratio.
All six runs pass `allclose(atol=2e-4, rtol=2e-3)`. The detailed result and
protocol are in
`results/paired_resnet18_metaschedule_formal1024/README.md`.

## Reproduction Entry Points (v1)

Run the paired tuning and complete-model experiment on the RTX 4090 host:

```bash
TRIALS=1024 \
OUT_ROOT=/path/to/output \
bash experiments/run_paired_resnet18_metaschedule_4090.sh
```

Rebuild and profile selected schedules from the paired databases:

```bash
ROOT=/path/to/output \
WORKLOAD_KEYS="7990226299771950393 5769126934908556840 1095410103996100310" \
bash experiments/run_paired_metaschedule_profiles_4090.sh
```

Collect Nsight Compute counters for the same rebuilt schedules:

```bash
ROOT=/path/to/output \
WORKLOAD_KEYS="7990226299771950393 5769126934908556840 1095410103996100310" \
PAUSE_PID=<optional pid to suspend during profiling> \
bash experiments/run_paired_metaschedule_ncu_4090.sh
```

The scripts provide defaults matching the experiment host; every path can be
overridden through the documented environment variables. Nsight Systems
intervals and Nsight Compute counters are both included. Nsight Compute
serializes and replays each profiled launch, so its instrumented durations are
not comparable with the Nsight Systems intervals; only the counters from that
pass are used. On this host, counter collection requires administrative
privileges because the driver is loaded with `RmProfilingAdminOnly=1`.

The `results/` tree records the outputs of these scripts as they were collected
on the evaluation hosts, so the reported numbers can be checked without access
to the hardware. The three main experiment directories
(`paired_resnet18_metaschedule_formal1024/`, `seven_model_relax_4090/`, and
`dcu_validation_20260819/`) each carry a `README.md` describing the protocol
used for that experiment.
