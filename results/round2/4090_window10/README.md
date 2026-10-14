# Window 10 (RTX 4090 device 1, 2026-10-13 22:13:32-23:40:24): retune with DPC++ v7.1.1

Script: `experiments/round2/run_window10.sh`. Device 1 was used exclusively during the window. Builds as in window 9, after R2's
libsycl-9 default (no re-finalization on the v7.1.1 build).

Retune of ResNet-18 B1 with the v7.1.1 build, settings of the window-3 retunes (1,024 trials, seed
20261012, 8 candidates per iteration, rule kind cuda, `-arch=sm_89`, T1 runner, T2 timer): 69 min
(646ab08: 75 min), 1,029 records, 3 failed (window 3: 17), no measurement process replaced
(`tune_sycl_v711/`; the per-task logs stay on the server).

p50 in ms, median of three processes (10 warm-ups, 50 measurements):

| Database | build | eager | graphs |
|---|---|---:|---:|
| v7.1.1 retune | v7.1.1 | 0.569 | 0.555 |
| v7.1.1 retune | 646ab08 | 0.657 | |
| window 3 (646ab08) | v7.1.1 | 0.563 | 0.549 |
| window 3 (646ab08) | 646ab08 | 0.529 | 0.514 |
| CUDA, window 3 | | 0.507 | 0.493 |

Compact models with command graphs on v7.1.1 with no re-finalization: Transformer 0.0701, ViT-style
0.0872, CNN 0.0400, MLP 0.0372 ms (window 9 with re-finalization: 0.0703 / 0.0873 / 0.0403 /
0.0374).

Traces (`e1/`, `experiments/round2/analysis/compare_by_order.py`): the window-3 database on the two builds, 500.3 vs
534.3 us of kernel time per inference; the two launches of h894117287 take 45 vs 56 us, h1053957056
40 vs 46 us; the v7.1.1 database on v7.1.1, 538.2 us. Static comparison of the same source
(`experiments/round2/tcmp_toolchains.sh`): v7.1.1 removes the 64-bit index arithmetic (h894117287:
SHF.R.S64 292 -> 5, IMAD.HI 150 -> 0, LDS.128 245 -> 288, 197 -> 168 registers) but the kernel
body changes shape (FFMA 948 -> 1152, LOP3 10 -> 266) and runs slower. A tuned schedule is specific
to the compiler that measured it, as with N1 (window 7).
