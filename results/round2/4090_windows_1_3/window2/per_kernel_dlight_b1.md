# Per-workload kernel time, ms per inference (median)

| Workload | iso_cuda_b1 | iso_sycl_b1 | iso_sycl_b1/iso_cuda_b1 |
|---|---:|---:|---:|
| fused_conv2d9_add7_add8_relu4 | 0.7147 | 1.1387 | 1.59 |
| fused_conv2d6_add5_add6_relu3 | 0.3618 | 0.5873 | 1.62 |
| fused_conv2d9_add7_relu4 | 0.3571 | 0.5839 | 1.63 |
| fused_conv2d3_add3_add4_relu2 | 0.1909 | 0.3039 | 1.59 |
| fused_conv2d8_add7_relu4 | 0.1901 | 0.3069 | 1.61 |
| fused_conv2d6_add5_relu3 | 0.1807 | 0.2933 | 1.62 |
| fused_conv2d1_add1_relu1 | 0.0968 | 0.1589 | 1.64 |
| fused_conv2d1_add1_add2_relu1 | 0.0960 | 0.1605 | 1.67 |
| fused_conv2d3_add3_relu2 | 0.0953 | 0.1515 | 1.59 |
| fused_conv2d5_add5_relu3 | 0.0944 | 0.1387 | 1.47 |
| fused_conv2d2_add3_relu2 | 0.0496 | 0.0742 | 1.50 |
| fused_conv2d_add_relu | 0.0228 | 0.0445 | 1.95 |
| fused_conv2d10_add7 | 0.0130 | 0.0197 | 1.52 |
| fused_conv2d7_add5 | 0.0072 | 0.0096 | 1.33 |
| fused_conv2d4_add3 | 0.0045 | 0.0059 | 1.32 |
| fused_matmul_add9 | 0.0029 | 0.0029 | 1.00 |
| max_pool2d | 0.0026 | 0.0033 | 1.27 |
| adaptive_avg_pool2d | 0.0013 | 0.0014 | 1.07 |
| **sum of kernels** | 2.4819 | 3.9847 | 1.61 |
| **device span** | 2.4971 | 4.0001 | 1.60 |

launches per inference: iso_cuda_b1=23, iso_sycl_b1=23; inferences: iso_cuda_b1=20, iso_sycl_b1=20
