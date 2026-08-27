# Paired ResNet-18 Relax/MetaSchedule Results

Equal global budget: 1024 trials per target.
Workload sets identical: True.
All apply-only runs pass: True.

| Target | MS measured | Fallback | Trials | Failed | Median p50 ms |
|---|---:|---:|---:|---:|---:|
| CUDA | 15/19 | 4 | 1029 | 215 | 0.781963 |
| SYCL | 15/19 | 4 | 1029 | 439 | 2.037659 |

Complete-model SYCL/CUDA ratio: **2.606x**.

The complete-model ratio is primary. The matched-workload database table is unweighted diagnostic evidence because graph call multiplicities differ.
