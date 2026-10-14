# Per-workload kernel time, ms per inference (median)

| Workload | cudadb_cuda_b1 | cudadb_sycl_c8_b1 | cudadb_sycl_c10_b1 | sycldb_sycl_c10_b1 | cudadb_sycl_c8_b1/cudadb_cuda_b1 | cudadb_sycl_c10_b1/cudadb_cuda_b1 | sycldb_sycl_c10_b1/cudadb_cuda_b1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused_conv2d3_add3_add4_relu2 | 0.1909 | 0.3133 | 0.3038 | 0.0535 | 1.64 | 1.59 | 0.28 |
| fused_conv2d9_add7_add8_relu4 | 0.0958 | 0.2383 | 0.1038 | 0.1505 | 2.49 | 1.08 | 1.57 |
| fused_conv2d5_add5_relu3 | 0.0944 | 0.1602 | 0.1407 | 0.0279 | 1.70 | 1.49 | 0.30 |
| fused_conv2d9_add7_relu4 | 0.0588 | 0.1778 | 0.0616 | 0.0648 | 3.02 | 1.05 | 1.10 |
| fused_conv2d6_add5_add6_relu3 | 0.0536 | 0.3142 | 0.1138 | 0.5873 | 5.87 | 2.12 | 10.97 |
| fused_conv2d8_add7_relu4 | 0.0452 | 0.2078 | 0.0585 | 0.0417 | 4.60 | 1.30 | 0.92 |
| fused_conv2d1_add1_add2_relu1 | 0.0421 | 0.2807 | 0.0441 | 0.0439 | 6.66 | 1.05 | 1.04 |
| fused_conv2d1_add1_relu1 | 0.0362 | 0.3603 | 0.0379 | 0.0452 | 9.95 | 1.05 | 1.25 |
| fused_conv2d6_add5_relu3 | 0.0326 | 0.1080 | 0.0354 | 0.2934 | 3.31 | 1.09 | 9.01 |
| fused_conv2d_add_relu | 0.0287 | 0.1290 | 0.0315 | 0.0175 | 4.49 | 1.10 | 0.61 |
| fused_conv2d2_add3_relu2 | 0.0193 | 0.1215 | 0.0196 | 0.0742 | 6.29 | 1.01 | 3.84 |
| fused_conv2d3_add3_relu2 | 0.0182 | 0.1454 | 0.0239 | 0.0459 | 7.99 | 1.31 | 2.52 |
| fused_conv2d10_add7 | 0.0135 | 0.1958 | 0.0145 | 0.0223 | 14.47 | 1.07 | 1.65 |
| fused_conv2d4_add3 | 0.0101 | 0.0897 | 0.0103 | 0.0304 | 8.87 | 1.02 | 3.01 |
| fused_conv2d7_add5 | 0.0085 | 0.0310 | 0.0159 | 0.0057 | 3.64 | 1.86 | 0.67 |
| fused_matmul_add9 | 0.0029 | 0.0040 | 0.0030 | 0.0299 | 1.39 | 1.06 | 10.38 |
| max_pool2d | 0.0028 | 0.0033 | 0.0033 | 0.0028 | 1.20 | 1.19 | 1.03 |
| adaptive_avg_pool2d | 0.0013 | 0.0018 | 0.0014 | 0.0014 | 1.33 | 1.05 | 1.05 |
| **sum of kernels** | 0.7547 | 2.8819 | 1.0230 | 1.5383 | 3.82 | 1.36 | 2.04 |
| **device span** | 0.7695 | 2.8984 | 1.0383 | 1.5729 | 3.77 | 1.35 | 2.04 |

launches per inference: cudadb_cuda_b1=23, cudadb_sycl_c8_b1=23, cudadb_sycl_c10_b1=23, sycldb_sycl_c10_b1=23; inferences: cudadb_cuda_b1=20, cudadb_sycl_c8_b1=20, cudadb_sycl_c10_b1=20, sycldb_sycl_c10_b1=20
