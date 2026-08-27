# Paired ResNet-18 Relax/MetaSchedule Experiment

This directory contains the formal RTX 4090 evidence used for the paired
ResNet-18 B1 result in the manuscript.

## Experimental Control

- TVM: v0.21 Relax/TensorIR/MetaSchedule
- Model: torchvision ResNet-18, BatchNorm folded before `torch.export`
- Input: `(1,3,224,224)`
- Targets: CUDA and SYCL on the same RTX 4090
- Target metadata: `max_num_threads=1024`, `thread_warp_size=32`
- Nominal global budget: 1,024 trials per target
- Extracted workloads: 19 per target
- Serialized records: 1,029 per target
- Complete-model timing: 10 warm-ups and 50 synchronized measurements in
  each of three independent database-reuse processes

## Primary Result

| Target | Finite workloads | DLight fallback | Unsuccessful records | Three p50 values (ms) | Median (ms) | Max abs. error |
|---|---:|---:|---:|---|---:|---:|
| CUDA | 15/19 | 4 | 215/1,029 (20.9%) | 0.777 / 0.782 / 0.782 | 0.782 | 2.15e-6 |
| SYCL | 15/19 | 4 | 439/1,029 (42.7%) | 2.034 / 2.039 / 2.038 | 2.038 | 3.81e-6 |

The synchronized complete-model SYCL/CUDA ratio is 2.606x. All six
database-reuse executions pass `allclose(atol=2e-4, rtol=2e-3)`.

## Matched-Workload Localization

Twelve workloads have finite records on both targets. Scheduler-weighted
best-record sums are 0.3282 ms for CUDA and 0.6819 ms for SYCL. Conv2d 3x3,
1x1, and 7x7 account for 73.9%, 19.8%, and 2.1% of the matched gap,
respectively; pooling accounts for 4.3%.

The aggregate files are:

- `paired_metaschedule_summary.json` and `.md`: complete-model runs
- `paired_metaschedule_matched_workloads.csv`: matched operator groups
- `record_analysis/paired_metaschedule_workloads.csv`: per-workload records
- `paired_metaschedule_failure_classes.json`: unsuccessful-record census

## Direct Profiles

Three paired convolution schedules are retained under `operator_profiles/`.
Each contains generated CUDA/SYCL source, scheduled TIR, synchronized wall
measurements, and Nsight Systems artifacts.

| Case | Best-record ratio | CUDA/SYCL wall us | CUDA/SYCL kernel us | CUDA/SYCL launch API us |
|---|---:|---:|---:|---:|
| 7x7 low | 1.29x | 34.76 / 49.21 | 27.74 / 26.72 | 4.32 / 5.86 |
| 3x3 medium | 2.33x | 25.23 / 56.01 | 17.86 / 32.80 | 4.46 / 5.70 |
| 1x1 high | 5.51x | 15.24 / 58.40 | 8.70 / 41.34 | 4.55 / 6.13 |

The high-ratio SYCL 1x1 schedule exposes four 224-thread work-groups and a
64-way serial outer reduction with two barriers inside the loop. The CUDA
schedule exposes 28 blocks and an eight-way tiled reduction.

Nsight Compute hardware counters were unavailable under the host profiling
permissions. The manuscript therefore relies only on synchronized wall time,
device-kernel intervals, driver launch intervals, launch geometry, and
generated-code structure.
