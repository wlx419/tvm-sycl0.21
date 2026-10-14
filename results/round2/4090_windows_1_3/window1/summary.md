| Configuration | procs | status | p50 median (ms) | p50 range over procs | IQR of samples | compile s | max abs err |
|---|---:|---|---:|---|---|---:|---:|
| cnn_cuda_eager | 3 | pass | 0.0384 | 0.0383-0.0385 | 0.0381-0.0387 | 2.6 | 1.09e-11 |
| cnn_cuda_graph | 3 | pass | 0.0370 | 0.0361-0.0373 | 0.0364-0.0374 | 2.6 | 1.09e-11 |
| cnn_sycl_a0_sync_legacy | 3 | pass | 0.1199 | 0.1173-0.1202 | 0.1180-0.1202 | 1.3 | 2.18e-11 |
| cnn_sycl_a2_async_fixed | 3 | pass | 0.0669 | 0.0663-0.0677 | 0.0664-0.0678 | 1.3 | 2.18e-11 |
| cnn_sycl_a3_graph | 3 | pass | 0.0639 | 0.0637-0.0640 | 0.0636-0.0642 | 1.3 | 2.18e-11 |
| mlp_cuda_eager | 3 | pass | 0.0291 | 0.0288-0.0347 | 0.0288-0.0347 | 1.1 | 8.38e-09 |
| mlp_cuda_graph | 3 | pass | 0.0306 | 0.0304-0.0311 | 0.0304-0.0311 | 1.2 | 8.38e-09 |
| mlp_sycl_a0_sync_legacy | 3 | pass | 0.0642 | 0.0638-0.0646 | 0.0638-0.0647 | 0.7 | 8.38e-09 |
| mlp_sycl_a2_async_fixed | 3 | pass | 0.0471 | 0.0457-0.0471 | 0.0460-0.0472 | 0.7 | 8.38e-09 |
| mlp_sycl_a3_graph | 3 | pass | 0.0515 | 0.0513-0.0527 | 0.0511-0.0528 | 0.7 | 8.38e-09 |
| rn18_cuda_eager | 3 | pass | 0.7805 | 0.7804-0.7844 | 0.7803-0.7841 | 56.9 | 2.15e-06 |
| rn18_cuda_graph | 3 | pass | 0.7698 | 0.7655-0.7703 | 0.7661-0.7703 | 56.8 | 2.15e-06 |
| rn18_published_cuda | 3 | pass | 0.7798 | 0.7796-0.7799 | 0.7792-0.7805 | 64.8 | 2.15e-06 |
| rn18_published_sycl | 3 | pass | 2.0563 | 2.0469-2.0594 | 2.0489-2.0591 | 201.6 | 3.81e-06 |
| rn18_sycl_a0_sync_legacy | 3 | pass | 2.0078 | 2.0054-2.0964 | 2.0054-2.0945 | 9.0 | 3.81e-06 |
| rn18_sycl_a1_async_legacy | 3 | pass | 1.7853 | 1.7808-1.7859 | 1.7815-1.7860 | 9.0 | 3.81e-06 |
| rn18_sycl_a2_async_fixed | 3 | pass | 1.7814 | 1.7812-1.7815 | 1.7806-1.7821 | 8.5 | 3.81e-06 |
| rn18_sycl_a3_graph | 3 | pass | 1.7701 | 1.7701-1.7705 | 1.7698-1.7709 | 9.1 | 3.81e-06 |
| transformer_cuda_eager | 3 | pass | 0.1206 | 0.1099-0.1222 | 0.1104-0.1225 | 4.0 | 1.12e-08 |
| transformer_cuda_graph | 3 | pass | 0.0658 | 0.0648-0.0660 | 0.0651-0.0661 | 4.0 | 1.12e-08 |
| transformer_sycl_a0_sync_legacy | 3 | pass | 0.4149 | 0.4086-0.4202 | 0.4098-0.4173 | 1.8 | 1.49e-08 |
| transformer_sycl_a2_async_fixed | 3 | pass | 0.2282 | 0.2261-0.3491 | 0.2260-0.3470 | 1.8 | 1.49e-08 |
| transformer_sycl_a3_graph | 3 | pass | 0.1724 | 0.1717-0.1732 | 0.1719-0.1732 | 1.8 | 1.49e-08 |
| vit_cuda_eager | 3 | pass | 0.1810 | 0.1809-0.1816 | 0.1802-0.1842 | 4.2 | 2.42e-08 |
| vit_cuda_graph | 3 | pass | 0.0814 | 0.0813-0.0816 | 0.0813-0.0817 | 4.2 | 2.42e-08 |
| vit_sycl_a0_sync_legacy | 3 | pass | 0.5454 | 0.5403-0.5465 | 0.5420-0.5473 | 1.9 | 9.31e-09 |
| vit_sycl_a2_async_fixed | 3 | pass | 0.3361 | 0.3348-0.3528 | 0.3339-0.3489 | 1.9 | 9.31e-09 |
| vit_sycl_a3_graph | 3 | pass | 0.2000 | 0.1995-0.2003 | 0.1979-0.2016 | 1.9 | 9.31e-09 |

| Ratio | point | bootstrap 95% CI |
|---|---:|---|
| rn18_published_sycl / rn18_published_cuda | 2.637 | 2.635-2.638 |
| rn18_sycl_a0_sync_legacy / rn18_cuda_eager | 2.573 | 2.571-2.573 |
| rn18_sycl_a1_async_legacy / rn18_cuda_eager | 2.287 | 2.286-2.288 |
| rn18_sycl_a2_async_fixed / rn18_cuda_eager | 2.282 | 2.281-2.283 |
| rn18_sycl_a3_graph / rn18_cuda_graph | 2.300 | 2.299-2.300 |
| rn18_sycl_a2_async_fixed / rn18_sycl_a0_sync_legacy | 0.887 | 0.887-0.888 |
| mlp_sycl_a0_sync_legacy / mlp_cuda_eager | 2.206 | 2.194-2.216 |
| mlp_sycl_a2_async_fixed / mlp_cuda_eager | 1.617 | 1.608-1.624 |
| mlp_sycl_a3_graph / mlp_cuda_graph | 1.681 | 1.670-1.700 |
| cnn_sycl_a0_sync_legacy / cnn_cuda_eager | 3.125 | 3.114-3.137 |
| cnn_sycl_a2_async_fixed / cnn_cuda_eager | 1.743 | 1.736-1.750 |
| cnn_sycl_a3_graph / cnn_cuda_graph | 1.725 | 1.712-1.733 |
| transformer_sycl_a0_sync_legacy / transformer_cuda_eager | 3.442 | 3.385-3.461 |
| transformer_sycl_a2_async_fixed / transformer_cuda_eager | 1.893 | 1.864-1.906 |
| transformer_sycl_a3_graph / transformer_cuda_graph | 2.621 | 2.614-2.626 |
| vit_sycl_a0_sync_legacy / vit_cuda_eager | 3.013 | 2.992-3.022 |
| vit_sycl_a2_async_fixed / vit_cuda_eager | 1.857 | 1.842-1.864 |
| vit_sycl_a3_graph / vit_cuda_graph | 2.458 | 2.445-2.463 |
