# Window 7 (RTX 4090 device 1, 2026-10-13 17:36:43-18:56:44): retune with N1

Script: `experiments/round2/run_window7.sh`. Device 1 was used exclusively during the window. Code: overlay_r2 before F2 (`build-4090-r2`,
DPC++ 646ab08).

Retune of ResNet-18 B1 for SYCL with `TVM_SYCL_NATIVE_IDS=1` (N1), with the settings of the
window-3 retunes: 1,024 trials, seed 20261012, 8 candidates per iteration, rule kind cuda,
`-arch=sm_89`, T1 runner (3 measurement processes replaced), T2 timer; 67 min, 1,029 records
(`tune_sycl_n1/`).

p50 in ms, median of three processes (10 warm-ups, 50 measurements):

| Database / code generation | eager | graphs |
|---|---:|---:|
| N1 database, N1 on | 0.571 | 0.542 |
| N1 database, N1 off | 0.720 | |
| window-3 database (nd_item indices), N1 off | 0.530 | 0.513 |
| CUDA, window-3 CUDA database | 0.507 | 0.493 |

The retune with N1 does not recover the 5-24% that N1 costs the schedules tuned without it, and
its own schedules are slower than the window-3 schedules with the default code generation; N1
stays opt-in. The default configuration runs at 1.045x / 1.041x of CUDA in this session.
`e1/n1db_sycl*`: Nsight trace of the N1 database (per-kernel comparison with window 6d).
