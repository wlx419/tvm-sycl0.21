# TVM-SYCL: a SYCL 2020 backend for Apache TVM v0.21

This repository contains the source overlay, experiment entry points, raw
records, derived summaries, and generated figures for the SYCL backend
described in the paper below. The overlay adds a SYCL code-generation target,
a SYCL runtime module and device API, and the associated target metadata and
build integration to Apache TVM v0.21.

The backend is evaluated on an NVIDIA RTX 4090 through the oneAPI CUDA adapter
and on a Sugon DCU Z100L through the HIP adapter.

## Paper

> Longxiang Wang, Xingjun Zhang, Xiaoshe Dong, Qiang Wang, Jia Wei, Zhao Mao,
> Weijun Xia, Zhixin Zhu, Yuchen Wu, Bolin Li, and Bowen Li.
> *Toward SYCL Support in TVM: Backend Implementation and Initial Evaluation
> for Deep Learning Inference.* Under review.

Corresponding author: Longxiang Wang, `wlx419@xjtu.edu.cn`, School of Computer
Science and Technology, Xi'an Jiaotong University.

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
| `source_overlay/` | TVM v0.21 files changed for the SYCL backend |
| `experiments/` | Run, analysis, profiling, and figure-generation entry points |
| `results/` | Raw records and derived summaries for every reported experiment |
| `figures/` | Generated figures as they appear in the paper |

## License

The overlay is derived from Apache TVM and is released under the Apache
License, Version 2.0; see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`NOTICE.tvm`](NOTICE.tvm). Modified upstream files retain their original
Apache Software Foundation license headers. This repository is an independent
research artifact and is not endorsed by or affiliated with the Apache Software
Foundation.

## Compiler and Scheduling Scope

All model experiments use TVM v0.21 Relax, TensorIR, and the Relax VM. The
focused ResNet-18 B1 comparison tunes CUDA and SYCL with MetaSchedule under the
same nominal 1,024-trial budget. The broader seven-model batch sweep uses the
default Relax scheduling pipeline to cover all models and batch sizes at
manageable cost; it is reported as correctness and robustness evidence rather
than tuned performance. The independent DCU seven-model sweep uses the
static-shape Relax pipeline with DLight scheduling and no search trials and is
reported as portability, correctness, and batch-robustness evidence.

## Evidence Map

- `results/paired_resnet18_metaschedule_formal1024/` contains the formal paired
  CUDA/SYCL MetaSchedule databases, three independent complete-model runs per
  target, unsuccessful-record census, matched-workload analysis, generated
  kernels, scheduled TIR, and direct operator profiles.
- `results/seven_model_relax_4090/` contains scheduling coverage for seven B1
  models and the paired CUDA/SYCL B={1,8,16} sweep. All 42 backend runs pass
  reference-output validation.
- `results/operator_correctness/` contains NVIDIA and DCU tensor-operator
  correctness records.
- `results/dcu_validation_20260819/` contains the final dual-card DCU operator
  regression (32/32 device-configuration runs), the seven-model B={1,8,16}
  sweep (21/21 runs), math-lowering diagnostics, and the tested source snapshot.
- `results/alternative_paths_4090/` contains the IREE timing audit and the
  verified oneDNN NVIDIA vendor-backend comparison.
- `source_overlay/` contains the TVM v0.21 files changed for SYCL code
  generation, runtime execution, target registration, Relax routing, and
  MetaSchedule integration.
- `experiments/` contains the corresponding run, analysis, profiling,
  summarization, and figure-generation entry points.

## Primary Paired Result

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

## Reproduction Entry Points

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

The scripts provide defaults matching the experiment host; every path can be
overridden through the documented environment variables. Nsight Systems
intervals are included. Hardware-counter collection was unavailable under the
host profiling permissions, so no occupancy or bandwidth-counter claim is
made.

The `results/` tree records the outputs of these scripts as they were collected
on the evaluation hosts, so the reported numbers can be checked without access
to the hardware. The three main experiment directories
(`paired_resnet18_metaschedule_formal1024/`, `seven_model_relax_4090/`, and
`dcu_validation_20260819/`) each carry a `README.md` describing the protocol
used for that experiment.
