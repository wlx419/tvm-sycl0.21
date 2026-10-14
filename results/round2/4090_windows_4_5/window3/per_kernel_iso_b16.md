# Per-workload kernel time, ms per inference (median)

| Workload | iso_cuda_b16 | iso_sycl_b16 | iso_sycl_b16/iso_cuda_b16 |
|---|---:|---:|---:|
| fused_conv2d9_add7_add8_relu4 | 0.9224 | 1.0228 | 1.11 |
| fused_conv2d6_add5_add6_relu3 | 0.6917 | 0.7616 | 1.10 |
| fused_conv2d3_add3_add4_relu2 | 0.5482 | 0.6630 | 1.21 |
| fused_conv2d1_add1_relu1 | 0.5185 | 0.6632 | 1.28 |
| fused_conv2d1_add1_add2_relu1 | 0.5151 | 0.6590 | 1.28 |
| fused_conv2d9_add7_relu4 | 0.4625 | 0.5116 | 1.11 |
| fused_conv2d6_add5_relu3 | 0.3469 | 0.3841 | 1.11 |
| fused_conv2d3_add3_relu2 | 0.2753 | 0.3333 | 1.21 |
| fused_conv2d_add_relu | 0.2408 | 0.3155 | 1.31 |
| fused_conv2d8_add7_relu4 | 0.2230 | 0.2505 | 1.12 |
| fused_conv2d5_add5_relu3 | 0.1746 | 0.1958 | 1.12 |
| fused_conv2d2_add3_relu2 | 0.1418 | 0.1745 | 1.23 |
| fused_conv2d4_add3 | 0.0237 | 0.0229 | 0.96 |
| max_pool2d | 0.0233 | 0.0295 | 1.27 |
| fused_matmul_add9 | 0.0232 | 0.0251 | 1.08 |
| fused_conv2d7_add5 | 0.0212 | 0.0231 | 1.09 |
| fused_conv2d10_add7 | 0.0204 | 0.0215 | 1.06 |
| adaptive_avg_pool2d | 0.0048 | 0.0050 | 1.03 |
| **sum of kernels** | 5.1775 | 6.0589 | 1.17 |
| **device span** | 5.1908 | 6.0727 | 1.17 |

launches per inference: iso_cuda_b16=23, iso_sycl_b16=23; inferences: iso_cuda_b16=20, iso_sycl_b16=20
