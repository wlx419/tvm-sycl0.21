# Results of v2.0

Raw records and summaries of the v2.0 experiments, as collected on the evaluation hosts. The
scripts are in `experiments/round2/` (see its README for the measurement protocol); the code is
`source_overlay/`. Every RTX 4090 timing configuration has one JSON record per process
(`<config>_run<N>.json`, with the per-process p50 and all samples); summaries use the median of the
three per-process p50 values. Window logs list every run with its pass/fail status.

Nsight Systems reports are not included (their
CSV exports are); symbolic links of the measurement directories are listed in `LINKS.txt`.

| Directory | Content |
|---|---|
| `4090_window1`, `4090_window1_e1` | Window 1: asynchronous dispatch, command graphs, ResNet-18 ablation steps; whole-model traces |
| `4090_windows_1_3/` | Windows 1, 1b (C8, C9, C10 with CUDA's schedules), 2 (launch microbenchmark, `discard_events`, cross-replay of the v1 databases), 3 (equal-budget retunes: `window3/tune_sycl_pipe_iso`, `window3/tune_cuda_iso`) |
| `4090_windows_4_5/` | Windows 4 (seven-model sweep, compute-bound shapes, TVM-OpenCL) and 5 (reduction lowerings), E10 with the window-4 build |
| `4090_window6`, `4090_windows_6/` | Window 6 and its follow-ups 6d, 6e (index width), 6g (graph-replay cost), 6h (R2 long runs, graph sweep); E10 with the final build (`e10_final2*`) |
| `4090_window7` | Retune with N1 (`tune_sycl_n1/`) and its comparison with the window-3 databases |
| `4090_window8` | Final backend: single-switch rows (F2, reductions, N1), seven-model sweep at B1/B8/B16 eager and with graphs, retuned ResNet-18, dispersion, Transformer/ViT traces |
| `4090_window9` | The same sources against DPC++ 646ab08 and v7.1.1: launch and graph-replay microbenchmarks, compact models, ResNets, compute-bound shapes, L1 on/off |
| `4090_window10` | Retune with DPC++ v7.1.1 (`tune_sycl_v711/`), traces of the window-3 database on both builds |
| `4090_summaries` | Per-kernel and per-window summaries of windows 1b-6 |
| `4090_e10*` | Operator correctness on the RTX 4090 (float64 references, masked relative error, ULP) |
| `4090_aux/` | JIT compile time (`jit`), grid limit and rsqrt accuracy (`e8e9`), warp-sync audit (`audit`), timer diagnostics (`tune_diag*`), register/SASS comparisons (`kcmp`, `ptx*_cmp`, `tcmp_*`), compute-bound oneDNN runs (`e11`), functional checks (`func*`) |
| `dcu_20261012`, `dcu_e10_r2` | DCU operator correctness |
| `dcu_r2_20261013`, `dcu_final_20261013` | DCU seven-model sweeps: v1 library, v2.0, TVM-HIP, switch ablations |

Key numbers (RTX 4090; ratio = SYCL / TVM-CUDA in the same session):

- ResNet-18 B1 tuned with equal budgets and identical settings: 0.529 vs 0.513 ms eager,
  0.514 vs 0.493 ms with command graphs (`4090_window9`); 1.03-1.06x across four sessions (v1:
  2.04 vs 0.78 ms).
- Same schedules on both backends (CUDA's database): 3.74x with the v1 code generation, 1.16x
  with v2.0 (`4090_windows_1_3/window1b`, `4090_window6`).
- Seven models at B1/B8/B16, DLight schedules: ResNets 0.99-1.09x; compact models 1.06-2.14x
  eager, 1.10-1.26x with command graphs (`4090_window8`).
- DPC++ v7.1.1 with L1: compact models 1.06-1.11x eager (646ab08: 1.70-2.03x); a launch costs
  2.5 us of host time (646ab08: 5.0 us; CUDA: 2.2 us); tuned ResNet-18 runs 6-8% slower than with
  646ab08, also after retuning (`4090_window9`, `4090_window10`).
