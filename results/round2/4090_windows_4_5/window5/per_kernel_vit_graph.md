# Per-workload kernel time, ms per inference (median)

| Workload | vit_cuda_graph | vit_sycl_graph | vit_sycl_graph/vit_cuda_graph |
|---|---:|---:|---:|
| transpose | 0.0142 | 0.0000 | 0.00 |
| fused_matmul1_add2 | 0.0094 | 0.0000 | 0.00 |
| transpose1 | 0.0055 | 0.0000 | 0.00 |
| fused_matmul4_add2_add1 | 0.0054 | 0.0000 | 0.00 |
| layer_norm | 0.0038 | 0.0000 | 0.00 |
| fused_matmul3_add3_gelu | 0.0035 | 0.0000 | 0.00 |
| fused_matmul1_add2_add1 | 0.0033 | 0.0000 | 0.00 |
| fused_matmul_add | 0.0031 | 0.0000 | 0.00 |
| fused_reshape1_transpose2 | 0.0020 | 0.0000 | 0.00 |
| transpose6 | 0.0014 | 0.0000 | 0.00 |
| softmax | 0.0014 | 0.0000 | 0.00 |
| matmul2 | 0.0014 | 0.0000 | 0.00 |
| transpose5 | 0.0014 | 0.0000 | 0.00 |
| fused_matmul2_multiply | 0.0014 | 0.0000 | 0.00 |
| fused_matmul5_add4 | 0.0013 | 0.0000 | 0.00 |
| fused_reshape_add1 | 0.0012 | 0.0000 | 0.00 |
| fused_reshape1_transpose2_transpose3 | 0.0012 | 0.0000 | 0.00 |
| sum | 0.0012 | 0.0000 | 0.00 |
| transpose7 | 0.0010 | 0.0000 | 0.00 |
| fused_transpose4_reshape2 | 0.0010 | 0.0000 | 0.00 |
| divide | 0.0009 | 0.0000 | 0.00 |
| Typeinfo name for h1303809897_mcdf71afc7c8155e7b | 0.0000 | 0.0014 | - |
| Typeinfo name for h1350265394_mcdf71afc7c8155e7b | 0.0000 | 0.0031 | - |
| Typeinfo name for h1534756871_mcdf71afc7c8155e7b | 0.0000 | 0.0015 | - |
| Typeinfo name for h2516655986_mcdf71afc7c8155e7b | 0.0000 | 0.0012 | - |
| Typeinfo name for h2652090963_mcdf71afc7c8155e7b | 0.0000 | 0.0058 | - |
| Typeinfo name for h2739039314_mcdf71afc7c8155e7b | 0.0000 | 0.0023 | - |
| Typeinfo name for h2762380778_mcdf71afc7c8155e7b | 0.0000 | 0.0015 | - |
| Typeinfo name for h3012509384_mcdf71afc7c8155e7b | 0.0000 | 0.0018 | - |
| Typeinfo name for h3050828056_mcdf71afc7c8155e7b | 0.0000 | 0.0010 | - |
| Typeinfo name for h3052360949_mcdf71afc7c8155e7b | 0.0000 | 0.0066 | - |
| Typeinfo name for h3688175330_mcdf71afc7c8155e7b | 0.0000 | 0.0013 | - |
| Typeinfo name for h370828354_mcdf71afc7c8155e7b | 0.0000 | 0.0018 | - |
| Typeinfo name for h3718562212_mcdf71afc7c8155e7b | 0.0000 | 0.0011 | - |
| Typeinfo name for h3958365298_mcdf71afc7c8155e7b | 0.0000 | 0.0043 | - |
| Typeinfo name for h3984580297_mcdf71afc7c8155e7b | 0.0000 | 0.0261 | - |
| Typeinfo name for h4119474894_mcdf71afc7c8155e7b | 0.0000 | 0.0011 | - |
| Typeinfo name for h4264150612_mcdf71afc7c8155e7b | 0.0000 | 0.0108 | - |
| Typeinfo name for h599492245_mcdf71afc7c8155e7b | 0.0000 | 0.0046 | - |
| Typeinfo name for h647666694_mcdf71afc7c8155e7b | 0.0000 | 0.0019 | - |
| Typeinfo name for h748868295_mcdf71afc7c8155e7b | 0.0000 | 0.0172 | - |
| Typeinfo name for h878658426_mcdf71afc7c8155e7b | 0.0000 | 0.0013 | - |
| **sum of kernels** | 0.0648 | 0.0976 | 1.51 |
| **device span** | 0.1204 | 0.1764 | 1.47 |

launches per inference: vit_cuda_graph=29, vit_sycl_graph=29; inferences: vit_cuda_graph=20, vit_sycl_graph=20
