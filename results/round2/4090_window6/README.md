# RTX 4090 windows 6, 6d, 6e, 6g, 6h (2026-10-13)

Raw files: `../4090_windows_6/` (JSON, logs, Nsight CSV exports; .nsys-rep/.sqlite left on the
server). Device 1, used exclusively during each window. p50 = median over three processes of the per-process p50 (ms).

Code: overlay_r2 with M1 (math workarounds only off `sm_*`), J1 (no implicit global offset), F1
(`-ffp-contract=fast`), R2 (graph re-finalize every 32 replays, from window 6h on); N1 opt-in.

## Window 6: final code (N1 off)

`summary_window6.md` (sweep eager columns, E11), `batch_sweep_rows.tex` (paper table: eager from
window 6, graphs from window 6h).

| ResNet-18 B1 | SYCL | CUDA | ratio |
|---|---:|---:|---:|
| retuned (window-3 databases), eager | 0.540 | 0.512 | 1.05 |
| retuned, graphs (window 6h) | 0.513 | 0.493 | 1.04 |
| identical schedules (CUDA published db) | 0.908 | 0.784 | 1.16 |
| published SYCL db, final | 1.232 | | |
| published SYCL db, final, graphs | 1.219 | | |
| published SYCL db, F1 off | 1.248 | | |
| published SYCL db, J1 off | 1.232 | | |
| published SYCL db, N1 on | 1.276 | | |

Switch-off ablation, compact models with graphs (B1): Transformer final 0.0796, M1 off 0.1025, F1 off
0.0836, J1 off 0.0825, all three off 0.1061, N1 on 0.0800; ViT final 0.1033, M1 off 0.1269.

Graph-mode traces (B1): Transformer kernels 50.5 vs 46.9 us (1.08x), gaps 50.6 vs 27.8 us (two region
boundaries: +11.8 and +11.0 us); ViT kernels 68.4 vs 64.9 us (1.05x). Remaining per-kernel ratios:
layer_norm/softmax ~1.2x, residual-add matmuls 1.1-1.2x, others 1.00-1.09x.

## Window 6d/6e: index form (N1)

p = nd_item indices (default), m31 = PTX registers masked to 31 bits, hw = hardware-range masks.

| config | p | m31 | hw |
|---|---:|---:|---:|
| e11 matmul 4096^3 | 6.352 | 5.931 | 5.879 |
| rn18 DLight | 2.673 | 2.599 | 2.597 |
| rn18 retuned | 0.532 | 0.659 | 0.675 |
| rn18 published SYCL db | 1.232 | 1.289 | 1.406 |
| rn50 DLight | 4.049 | 3.761 | 4.045 |
| rn50 DLight B16 | 10.696 | 10.574 | 10.342 |
| Transformer graph B1 / B16 | 0.0796 / 0.0870 | 0.0796 / 0.0874 | 0.0798 / 0.0867 |
| ViT graph B1 / B16 | 0.1021 / 0.1243 | 0.1024 / 0.1260 | 0.1027 / 0.1264 |

Retuned ResNet-18 per kernel (6d/6e traces): residual-add convolutions 1.25-1.9x slower with native
indices; the rest unchanged. Registers fall for all retuned kernels.

## Window 6g: graph replay host cost (microbenchmark)

`sycl_graph_replay_overhead.cpp`, `cuda_graph_replay_overhead.cu`; 26 small kernels per graph.
SYCL replays 1-100: 17 us; 101-1,000: 137 us; 1,001-3,000: 500 us (DPC++ 646ab08 keeps and checks the
event of every earlier replay). Re-finalize every 8/16/32/64/128 replays: 21.5/14.6/12.4/14.3/21.1 us
(finalize ~140 us). cudaGraphLaunch: 2.2 us.

## Window 6h: 3,000 consecutive inferences with graphs (B1), median per replay range

| | 1-50 | 51-100 | 101-1000 | 1001-3000 |
|---|---:|---:|---:|---:|
| Transformer SYCL, R2 off | 0.0795 | 0.0816 | 0.1216 | 0.3120 |
| Transformer SYCL, R2 on | 0.0812 | 0.0806 | 0.0805 | 0.0801 |
| Transformer CUDA | 0.0645 | 0.0645 | 0.0644 | 0.0639 |
| ViT SYCL, R2 off | 0.1037 | 0.1123 | 0.1705 | 0.3381 |
| ViT SYCL, R2 on | 0.1014 | 0.1006 | 0.0995 | 0.0868 |
| ViT CUDA | 0.0810 | 0.0809 | 0.0808 | 0.0724 |

## E10 (Table 3) with the final build

`e10_final2_sm89`: target naming sm_89 (native float32 erf): 16/16 pass; GELU max abs 4.33e-7, masked
rel 9.06e-6 (the float64-erf form: 5.15e-7 / 2.35e-5). Other operators identical to the earlier run.
