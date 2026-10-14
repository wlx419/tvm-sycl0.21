# Per-workload kernel time, ms per inference (median)

| Workload | transformer_cuda | transformer_sycl | transformer_sycl/transformer_cuda |
|---|---:|---:|---:|
| fused_matmul_add | 0.0097 | 0.0000 | 0.00 |
| transpose | 0.0058 | 0.0000 | 0.00 |
| fused_matmul3_add_add1 | 0.0055 | 0.0000 | 0.00 |
| layer_norm | 0.0041 | 0.0000 | 0.00 |
| fused_matmul2_add2_gelu | 0.0036 | 0.0000 | 0.00 |
| fused_matmul_add_add1 | 0.0034 | 0.0000 | 0.00 |
| fused_reshape_transpose1 | 0.0021 | 0.0000 | 0.00 |
| transpose5 | 0.0015 | 0.0000 | 0.00 |
| matmul1 | 0.0014 | 0.0000 | 0.00 |
| fused_matmul1_multiply | 0.0014 | 0.0000 | 0.00 |
| transpose4 | 0.0014 | 0.0000 | 0.00 |
| softmax | 0.0014 | 0.0000 | 0.00 |
| fused_matmul4_add3 | 0.0013 | 0.0000 | 0.00 |
| fused_reshape_transpose1_transpose2 | 0.0012 | 0.0000 | 0.00 |
| sum | 0.0012 | 0.0000 | 0.00 |
| transpose6 | 0.0011 | 0.0000 | 0.00 |
| fused_transpose3_reshape1 | 0.0011 | 0.0000 | 0.00 |
| divide | 0.0010 | 0.0000 | 0.00 |
| Typeinfo name for h1350265394_mcf6358d3b0deda1ce | 0.0000 | 0.0110 | - |
| Typeinfo name for h1534756871_mcf6358d3b0deda1ce | 0.0000 | 0.0016 | - |
| Typeinfo name for h1737888063_mcf6358d3b0deda1ce | 0.0000 | 0.0013 | - |
| Typeinfo name for h1753718718_mcf6358d3b0deda1ce | 0.0000 | 0.0014 | - |
| Typeinfo name for h1951265625_mcf6358d3b0deda1ce | 0.0000 | 0.0044 | - |
| Typeinfo name for h2033247447_mcf6358d3b0deda1ce | 0.0000 | 0.0263 | - |
| Typeinfo name for h2337775380_mcf6358d3b0deda1ce | 0.0000 | 0.0019 | - |
| Typeinfo name for h250168041_mcf6358d3b0deda1ce | 0.0000 | 0.0025 | - |
| Typeinfo name for h2762380778_mcf6358d3b0deda1ce | 0.0000 | 0.0012 | - |
| Typeinfo name for h3012509384_mcf6358d3b0deda1ce | 0.0000 | 0.0018 | - |
| Typeinfo name for h3050828056_mcf6358d3b0deda1ce | 0.0000 | 0.0012 | - |
| Typeinfo name for h3276024936_mcf6358d3b0deda1ce | 0.0000 | 0.0019 | - |
| Typeinfo name for h3611227213_mcf6358d3b0deda1ce | 0.0000 | 0.0016 | - |
| Typeinfo name for h3617141213_mcf6358d3b0deda1ce | 0.0000 | 0.0067 | - |
| Typeinfo name for h4119474894_mcf6358d3b0deda1ce | 0.0000 | 0.0012 | - |
| Typeinfo name for h599492245_mcf6358d3b0deda1ce | 0.0000 | 0.0048 | - |
| Typeinfo name for h748868295_mcf6358d3b0deda1ce | 0.0000 | 0.0062 | - |
| Typeinfo name for h81117850_mcf6358d3b0deda1ce | 0.0000 | 0.0012 | - |
| **sum of kernels** | 0.0484 | 0.0783 | 1.62 |
| **device span** | 0.1744 | 0.3352 | 1.92 |

launches per inference: transformer_cuda=26, transformer_sycl=26; inferences: transformer_cuda=20, transformer_sycl=20
