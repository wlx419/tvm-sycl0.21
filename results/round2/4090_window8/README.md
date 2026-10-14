# Window 8 (RTX 4090 device 1, 2026-10-13): final backend

Script: `experiments/round2/run_window8.sh`, two sessions: Part A 19:43:09-20:00:05 (`FINAL_ENV` then
provisionally `TVM_SYCL_SUBGROUP_REDUCE=1`, so its `a_*_final` rows repeat the sub-group column and are
not used), Part B 20:03:34-20:51:39. Device 1 was used exclusively during both parts.
Code: overlay_r2 after the cleanup and F2 (`build-4090-r2`, DPC++ 646ab08). Final configuration:
the defaults (M1, J1, F1, F2, R2 on; N1, sub-group and group reductions off), `FINAL_ENV='X=1'`.
Every entry is the median of three processes' p50 (10 warm-ups, 50 measurements), in ms.
Nsight reports (`*.nsys-rep`) stay on the server; the CSV exports are in `e1/`.

## Part A: single switches on the defaults (SYCL)

| Workload | defaults | F2 off | sub-group reduce | N1 |
|---|---:|---:|---:|---:|
| Transformer B1, graphs | 0.0801 | 0.0794 | 0.0780 | 0.0792 |
| ViT-style B1, graphs | 0.0998 | 0.1003 | 0.1007 | 0.0997 |
| CNN B1, graphs | 0.0455 | 0.0455 | 0.0456 | 0.0464 |
| MLP B1, graphs | 0.0498 | 0.0505 | 0.0493 | 0.0503 |
| ResNet-18 B1, DLight | 2.671 | 2.673 | | 2.599 |
| ResNet-50 B1, DLight | 4.047 | 4.045 | | 4.043 |

E11: conv2d 16x64x56^2 3x3 SYCL 0.348 (N1 0.299), CUDA 0.287; matmul 4096^3 SYCL 6.354 (N1 5.932),
CUDA 5.931. F2 and the reduction variants are within the process-to-process spread (about 2% for
the compact models); N1 helps the DLight GEMM/convolution kernels (E11, ResNet-18 3%) but is off by
default because it slows the tuned schedules (window 7).

## Part B: final configuration

- Seven-model sweep, eager and graphs on both backends: `batch_sweep_rows.tex`
  (`experiments/round2/analysis/batch_sweep_tex.py results/round2/4090_window8 results/round2/4090_window8`), the seven-model batch-sweep table.
  198/198 runs pass. ResNets 1.03-1.09x of CUDA (ResNet-34 B1 0.99x: CUDA processes settle at
  4.92 or 5.35 ms, two of three at 5.35 here); compact eager 1.06-2.14x; compact graphs 1.10-1.26x
  (MLP B1 1.74x).
- Retuned ResNet-18 B1 (window-3 databases): SYCL 0.538 eager / 0.512 graphs; CUDA 0.508 / 0.493
  (1.06x / 1.04x).
- Dispersion (`experiments/round2/analysis/dispersion.py results/round2/4090_window8`): process deviation from the median, median
  0.22% (ResNet eager), 0.08% (ResNet graphs), 1.94% (compact eager), 1.47% (compact graphs); largest
  55% (MLP B1 CUDA eager, one process at 0.0434 against 0.0281/0.0279). Within-process IQR/p50 median
  0.23% (ResNets), 1.05-1.37% (compact).
- Traces (`experiments/round2/analysis/e1_compact_kernels.py results/round2/4090_window8`): Transformer B1 kernels 48.3 (CUDA) /
  51.9 (SYCL) us eager, 46.9 / 50.4 us with graphs; ViT-style 66.8 / 70.6 and 65.1 / 68.5 us. Device
  idle against the untraced p50: Transformer eager 55% / 78%, graphs 28% / 37%.
