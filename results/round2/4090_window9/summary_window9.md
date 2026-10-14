## Host cost per launch / graph replay (us, median of processes)

| Benchmark | 646ab08 | v7.1.1 |
|---|---:|---:|
| launch in_order | 5.008 | 4.0200000000000005 |
| launch in_order+discard_events | 3.9915000000000003 | 4.0280000000000005 |
| launch in_order+launch_grouped |  | 2.5075000000000003 |
| launch in_order+parallel_for | 5.248 | 3.4935 |
| launch in_order+discard_events+launch_grouped |  | 2.498 |
| launch in_order+discard_events+parallel_for | 4.2115 | 3.5295 |
| graph growth K=10 replays 1-100 | 16.705 | 3.7300000000000004 |
| graph growth K=10 replays 1001-3000 | 495.015 | 3.4050000000000002 |
| graph growth K=10 replays 101-1000 | 136.56 | 3.4699999999999998 |
| graph re-finalize every 32 (K=10) | 10.23 | 5.42 |

## Models (p50 ms, median of three processes)

| Configuration | 646ab08 | v7.1.1 | v7.1.1 L1 off | CUDA | v711/r2 | r2/CUDA | v711/CUDA | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| c_cnn_eager | 0.0639 | 0.0407 | 0.0524 | 0.0374 | 0.64x | 1.71x | 1.09x | 3/3/3/3 |
| c_cnn_graph | 0.0458 | 0.0403 | 0.0416 | 0.0370 | 0.88x | 1.24x | 1.09x | 3/3/3/3 |
| c_mlp_eager | 0.0462 | 0.0289 | 0.0387 | 0.0272 | 0.63x | 1.70x | 1.06x | 3/3/3/3 |
| c_mlp_graph | 0.0491 | 0.0374 | 0.0403 | 0.0291 | 0.76x | 1.69x | 1.28x | 3/3/3/3 |
| c_transformer_eager | 0.2226 | 0.1205 | 0.1710 | 0.1096 | 0.54x | 2.03x | 1.10x | 3/3/3/3 |
| c_transformer_graph | 0.0793 | 0.0703 | 0.0731 | 0.0648 | 0.89x | 1.22x | 1.08x | 3/3/3/3 |
| c_vit_eager | 0.3341 | 0.1878 | 0.2528 | 0.1699 | 0.56x | 1.97x | 1.11x | 3/3/3/3 |
| c_vit_graph | 0.1006 | 0.0873 | 0.0906 | 0.0813 | 0.87x | 1.24x | 1.07x | 3/3/3/3 |
| dl_resnet18 | 2.672 | 2.669 |  | 2.509 | 1.00x | 1.06x | 1.06x | 3/3/0/3 |
| dl_resnet50 | 4.043 | 4.093 |  | 3.924 | 1.01x | 1.03x | 1.04x | 3/3/0/3 |
| e11/conv2d | 0.3477 | 0.2960 |  | 0.2874 | 0.85x | 1.21x | 1.03x | 3/3/0/3 |
| e11/matmul | 6.293 | 6.223 |  | 5.924 | 0.99x | 1.06x | 1.05x | 3/3/0/3 |
| rn18_tuned_eager | 0.5288 | 0.5592 | 0.5648 | 0.5126 | 1.06x | 1.03x | 1.09x | 3/3/3/3 |
| rn18_tuned_graph | 0.5139 | 0.5549 |  | 0.4934 | 1.08x | 1.04x | 1.12x | 3/3/0/3 |
| c_transformer_graph_r2off (v7.1.1) | | 0.0700 | | | | | | 3 |

passing runs: 156/156
