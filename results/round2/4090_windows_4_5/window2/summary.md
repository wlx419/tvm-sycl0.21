| Configuration | procs | status | p50 median (ms) | p50 range over procs | IQR of samples | compile s | max abs err |
|---|---:|---|---:|---|---|---:|---:|
| cnn_cuda | 3 | pass | 0.0374 | 0.0373-0.0375 | 0.0371-0.0376 | 2.6 | 1.09e-11 |
| cnn_cuda_graph | 3 | pass | 0.0363 | 0.0361-0.0368 | 0.0360-0.0368 | 2.6 | 1.09e-11 |
| cnn_sycl_c10 | 3 | pass | 0.0650 | 0.0643-0.0655 | 0.0644-0.0656 | 1.4 | 2.18e-11 |
| cnn_sycl_c10_de | 3 | pass | 0.0575 | 0.0561-0.0576 | 0.0565-0.0578 | 1.4 | 2.18e-11 |
| cnn_sycl_c10_de_graph | 3 | pass | 0.0629 | 0.0628-0.0638 | 0.0600-0.0664 | 1.4 | 2.18e-11 |
| cnn_sycl_c10_graph | 3 | pass | 0.0540 | 0.0535-0.0547 | 0.0535-0.0547 | 1.4 | 2.18e-11 |
| e3_cuda_with_cuda_noasync_db | 3 | pass | 1.1448 | 1.1410-1.1450 | 1.1412-1.1451 | 21.0 | 2.38e-06 |
| e3_cuda_with_sycl_db | 0 | runtime_error | nan | nan-nan | nan-nan | 14.4 | 0.00e+00 |
| e3_sycl_with_cuda_noasync_db | 3 | pass | 1.7249 | 1.7191-1.7267 | 1.7200-1.7264 | 11.6 | 2.38e-06 |
| mlp_cuda | 3 | pass | 0.0282 | 0.0275-0.0287 | 0.0278-0.0289 | 1.2 | 8.38e-09 |
| mlp_cuda_graph | 3 | pass | 0.0294 | 0.0293-0.0301 | 0.0293-0.0301 | 1.2 | 8.38e-09 |
| mlp_sycl_c10 | 3 | pass | 0.0468 | 0.0462-0.0478 | 0.0464-0.0477 | 0.7 | 8.38e-09 |
| mlp_sycl_c10_de | 3 | pass | 0.0407 | 0.0397-0.0411 | 0.0399-0.0410 | 0.8 | 8.38e-09 |
| mlp_sycl_c10_de_graph | 3 | pass | 0.0659 | 0.0653-0.0665 | 0.0590-0.0739 | 0.7 | 8.38e-09 |
| mlp_sycl_c10_graph | 3 | pass | 0.0518 | 0.0495-0.0523 | 0.0504-0.0524 | 0.8 | 8.38e-09 |
| rn18_cuda_iso_eager | 3 | pass | 2.5088 | 2.5079-2.5089 | 2.5080-2.5091 | 14.3 | 2.03e-06 |
| rn18_cuda_iso_graph | 3 | pass | 2.4961 | 2.4950-2.4961 | 2.4953-2.4966 | 13.8 | 2.03e-06 |
| rn18_cuda_with_sycl_iso_db | 3 | pass | 2.5096 | 2.5089-2.5122 | 2.5083-2.5116 | 14.0 | 2.03e-06 |
| rn18_cudadb_sycl_c10_de | 3 | pass | 1.0603 | 1.0546-1.0609 | 1.0557-1.0609 | 34.7 | 2.15e-06 |
| rn18_cudadb_sycl_c10_de_graph | 3 | pass | 1.0602 | 1.0596-1.0680 | 1.0584-1.0661 | 34.1 | 2.15e-06 |
| rn18_sycl_pipe_iso_eager | 3 | pass | 4.0179 | 3.6989-4.0185 | 3.6993-4.0187 | 7.2 | 2.03e-06 |
| rn18_sycl_pipe_iso_graph | 3 | pass | 3.6876 | 3.6834-4.0029 | 3.6857-4.0060 | 7.2 | 2.03e-06 |
| rn18_sycl_with_cuda_iso_db | 3 | pass | 4.0165 | 3.7071-4.0193 | 3.7065-4.0190 | 7.2 | 2.03e-06 |
| transformer_cuda | 3 | pass | 0.1093 | 0.1092-0.1105 | 0.1087-0.1111 | 4.1 | 1.12e-08 |
| transformer_cuda_graph | 3 | pass | 0.0649 | 0.0649-0.0649 | 0.0647-0.0652 | 4.1 | 1.12e-08 |
| transformer_sycl_c10 | 3 | pass | 0.2196 | 0.2192-0.2357 | 0.2188-0.2335 | 1.9 | 1.49e-08 |
| transformer_sycl_c10_de | 3 | pass | 0.1876 | 0.1866-0.2037 | 0.1859-0.2006 | 1.8 | 1.49e-08 |
| transformer_sycl_c10_de_graph | 3 | pass | 0.1302 | 0.1294-0.1302 | 0.1239-0.1365 | 1.9 | 1.49e-08 |
| transformer_sycl_c10_graph | 3 | pass | 0.1075 | 0.1066-0.1079 | 0.1067-0.1080 | 1.9 | 1.49e-08 |
| vit_cuda | 3 | pass | 0.1712 | 0.1685-0.1716 | 0.1698-0.1731 | 4.2 | 2.42e-08 |
| vit_cuda_graph | 3 | pass | 0.0809 | 0.0809-0.0813 | 0.0808-0.0813 | 4.2 | 2.42e-08 |
| vit_sycl_c10 | 3 | pass | 0.3322 | 0.3278-0.3392 | 0.3287-0.3378 | 2.0 | 9.31e-09 |
| vit_sycl_c10_de | 3 | pass | 0.2825 | 0.2799-0.2841 | 0.2798-0.2848 | 2.0 | 9.31e-09 |
| vit_sycl_c10_de_graph | 3 | pass | 0.1620 | 0.1618-0.1624 | 0.1503-0.1744 | 2.0 | 9.31e-09 |
| vit_sycl_c10_graph | 3 | pass | 0.1312 | 0.1305-0.1314 | 0.1291-0.1326 | 2.0 | 9.31e-09 |

| Ratio | point | bootstrap 95% CI |
|---|---:|---|
| mlp_sycl_c10 / mlp_cuda | 1.659 | 1.652-1.671 |
| cnn_sycl_c10 / cnn_cuda | 1.741 | 1.734-1.748 |
| transformer_sycl_c10 / transformer_cuda | 2.008 | 1.986-2.020 |
| vit_sycl_c10 / vit_cuda | 1.940 | 1.928-1.949 |
| mlp_sycl_c10_graph / mlp_cuda_graph | 1.760 | 1.749-1.772 |
| cnn_sycl_c10_graph / cnn_cuda_graph | 1.486 | 1.477-1.499 |
| transformer_sycl_c10_graph / transformer_cuda_graph | 1.657 | 1.654-1.660 |
| vit_sycl_c10_graph / vit_cuda_graph | 1.622 | 1.611-1.629 |
| mlp_sycl_c10_de / mlp_sycl_c10 | 0.870 | 0.866-0.872 |
| cnn_sycl_c10_de / cnn_sycl_c10 | 0.884 | 0.880-0.887 |
| transformer_sycl_c10_de / transformer_sycl_c10 | 0.854 | 0.849-0.859 |
| vit_sycl_c10_de / vit_sycl_c10 | 0.850 | 0.844-0.855 |
