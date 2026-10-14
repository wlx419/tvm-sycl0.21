# Window 9 (RTX 4090 device 1, 2026-10-13 21:28:43-22:05:05): DPC++ v7.1.1 side by side

Script: `experiments/round2/run_window9.sh`; summary: `summary_window9.md`
(`experiments/round2/analysis/summarize_window9.py results/round2/4090_window9`). Device 1 was used exclusively during the window.

The same r2 sources (overlay_r2 with L1, default configuration) built twice:
`build-4090-r2` against DPC++ 646ab08 (clang 18, libsycl 7.2; the toolchain of the main results) and
`build-4090-v711` against DPC++ v7.1.1 (intel/llvm release tarball, clang 22.1, libsycl 9.0,
UR 0.12; `experiments/round2/build_dpcpp_v711.sh`, `build_tvm_v711.sh`). Within each configuration the
processes of the two builds alternate; TVM-CUDA ran in the same session. R2 re-finalized every 32
replays on both builds in this window (the libsycl-9 default of 0 came after it; the
`c_transformer_graph_r2off` row shows the difference is nil at 50 replays). 156/156 runs pass.

Host cost (us, microbenchmarks built with each toolchain, two processes each):

| | 646ab08 | v7.1.1 |
|---|---:|---:|
| launch, queue::submit + handler | 5.01 | 4.02 |
| launch, queue::parallel_for shortcut | 5.25 | 3.49 |
| launch, khr::launch_grouped (L1) | - | 2.51 |
| graph replay, replays 1-100 / 101-1,000 / 1,001-3,000 | 16.7 / 136.6 / 495.0 | 3.73 / 3.47 / 3.41 |
| graph replay, re-finalized every 32 | 10.2 | 5.4 |

Models (p50 ms; ratio to TVM-CUDA of the same session):

| | 646ab08 | v7.1.1 | v7.1.1, L1 off | CUDA |
|---|---:|---:|---:|---:|
| Transformer B1 eager | 0.2226 (2.03x) | 0.1205 (1.10x) | 0.1710 | 0.1096 |
| Transformer B1 graphs | 0.0793 (1.22x) | 0.0703 (1.08x) | 0.0731 | 0.0648 |
| ViT-style B1 eager | 0.3341 (1.97x) | 0.1878 (1.11x) | 0.2528 | 0.1699 |
| ViT-style B1 graphs | 0.1006 (1.24x) | 0.0873 (1.07x) | 0.0906 | 0.0813 |
| CNN B1 eager | 0.0639 (1.71x) | 0.0407 (1.09x) | 0.0524 | 0.0374 |
| CNN B1 graphs | 0.0458 (1.24x) | 0.0403 (1.09x) | 0.0416 | 0.0370 |
| MLP B1 eager | 0.0462 (1.70x) | 0.0289 (1.06x) | 0.0387 | 0.0272 |
| MLP B1 graphs | 0.0491 (1.69x) | 0.0374 (1.28x) | 0.0403 | 0.0291 |
| ResNet-18 B1 DLight | 2.672 (1.06x) | 2.669 (1.06x) | | 2.509 |
| ResNet-50 B1 DLight | 4.043 (1.03x) | 4.093 (1.04x) | | 3.924 |
| ResNet-18 B1, window-3 databases, eager | 0.5288 (1.03x) | 0.5592 (1.09x) | 0.5648 | 0.5126 |
| ResNet-18 B1, window-3 databases, graphs | 0.5139 (1.04x) | 0.5549 (1.12x) | | 0.4934 |
| E11 conv2d 16x64x56^2 3x3 | 0.3477 (1.21x) | 0.2960 (1.03x) | | 0.2874 |
| E11 matmul 4096^3 | 6.293 (1.06x) | 6.223 (1.05x) | | 5.924 |

Machine code (`experiments/round2/tcmp_toolchains.sh`, same generated source, runtime JIT flags):
- E11 conv2d: 128 -> 64 registers (SASS 928 -> 824), the effect N1 forces with 646ab08.
- Tuned ResNet-18 (window-3 SYCL database): per-kernel changes in both directions; spill stores
  52 -> 240 B in h1053957056 (40 us/inference), 0 -> 12 B in h1977475787 (32 us), 89 -> 102
  registers and +27% SASS in h2971782926 (40 us); 197 -> 168 registers in h894117287 (91 us).
  The schedules were tuned with 646ab08's code; window 10 retunes with v7.1.1.

`e1/`: Nsight traces of the tuned ResNet-18 on the 646ab08 build and CUDA (the retune comparison: kernel
time 502.8 vs 482.4 us per inference eager, 494.0 vs 481.3 us with graphs).
