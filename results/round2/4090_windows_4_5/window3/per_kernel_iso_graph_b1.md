# Per-workload kernel time, ms per inference (median)

| Workload | iso_cuda_graph_b1 | iso_sycl_graph_b1 | iso_sycl_graph_b1/iso_cuda_graph_b1 |
|---|---:|---:|---:|
| fused_conv2d9_add7_add8_relu4 | 0.0749 | 0.0910 | 1.21 |
| fused_conv2d6_add5_add6_relu3 | 0.0713 | 0.0565 | 0.79 |
| fused_conv2d8_add7_relu4 | 0.0540 | 0.0414 | 0.77 |
| fused_conv2d3_add3_add4_relu2 | 0.0395 | 0.0445 | 1.13 |
| fused_conv2d1_add1_add2_relu1 | 0.0377 | 0.0426 | 1.13 |
| fused_conv2d9_add7_relu4 | 0.0335 | 0.0398 | 1.19 |
| fused_conv2d1_add1_relu1 | 0.0306 | 0.0320 | 1.05 |
| fused_conv2d6_add5_relu3 | 0.0285 | 0.0281 | 0.99 |
| fused_conv2d3_add3_relu2 | 0.0269 | 0.0209 | 0.78 |
| fused_conv2d2_add3_relu2 | 0.0198 | 0.0182 | 0.92 |
| fused_conv2d5_add5_relu3 | 0.0180 | 0.0215 | 1.19 |
| fused_conv2d_add_relu | 0.0174 | 0.0202 | 1.16 |
| fused_conv2d4_add3 | 0.0083 | 0.0058 | 0.70 |
| fused_conv2d10_add7 | 0.0063 | 0.0146 | 2.30 |
| fused_conv2d7_add5 | 0.0049 | 0.0081 | 1.66 |
| fused_matmul_add9 | 0.0043 | 0.0079 | 1.86 |
| max_pool2d | 0.0024 | 0.0032 | 1.33 |
| adaptive_avg_pool2d | 0.0015 | 0.0015 | 1.02 |
| **sum of kernels** | 0.4798 | 0.4977 | 1.04 |
| **device span** | 0.5013 | 0.5291 | 1.06 |

launches per inference: iso_cuda_graph_b1=23, iso_sycl_graph_b1=23; inferences: iso_cuda_graph_b1=20, iso_sycl_graph_b1=20
