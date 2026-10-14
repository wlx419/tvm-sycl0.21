# Per-workload kernel time, ms per inference (median)

| Workload | iso_cuda_b1 | iso_sycl_b1 | iso_sycl_b1/iso_cuda_b1 |
|---|---:|---:|---:|
| fused_conv2d9_add7_add8_relu4 | 0.0750 | 0.0911 | 1.21 |
| fused_conv2d6_add5_add6_relu3 | 0.0716 | 0.0574 | 0.80 |
| fused_conv2d8_add7_relu4 | 0.0542 | 0.0400 | 0.74 |
| fused_conv2d3_add3_add4_relu2 | 0.0397 | 0.0454 | 1.14 |
| fused_conv2d1_add1_add2_relu1 | 0.0385 | 0.0429 | 1.11 |
| fused_conv2d9_add7_relu4 | 0.0338 | 0.0399 | 1.18 |
| fused_conv2d1_add1_relu1 | 0.0306 | 0.0328 | 1.07 |
| fused_conv2d6_add5_relu3 | 0.0288 | 0.0284 | 0.99 |
| fused_conv2d3_add3_relu2 | 0.0270 | 0.0211 | 0.78 |
| fused_conv2d2_add3_relu2 | 0.0196 | 0.0183 | 0.93 |
| fused_conv2d5_add5_relu3 | 0.0181 | 0.0220 | 1.21 |
| fused_conv2d_add_relu | 0.0173 | 0.0202 | 1.16 |
| fused_conv2d4_add3 | 0.0085 | 0.0059 | 0.69 |
| fused_conv2d10_add7 | 0.0064 | 0.0141 | 2.22 |
| fused_conv2d7_add5 | 0.0050 | 0.0081 | 1.62 |
| fused_matmul_add9 | 0.0043 | 0.0076 | 1.79 |
| max_pool2d | 0.0022 | 0.0029 | 1.35 |
| adaptive_avg_pool2d | 0.0013 | 0.0014 | 1.02 |
| **sum of kernels** | 0.4819 | 0.4995 | 1.04 |
| **device span** | 0.4972 | 0.5510 | 1.11 |

launches per inference: iso_cuda_b1=23, iso_sycl_b1=23; inferences: iso_cuda_b1=20, iso_sycl_b1=20
