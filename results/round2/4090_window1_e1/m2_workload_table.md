| Workload | Kind | Calls | CUDA schedule | SYCL schedule | CUDA (ms) | SYCL (ms) | SYCL - CUDA |
|---|---|---:|---|---|---:|---:|---:|
| `fused_conv2d_add_relu` | 7x7 conv | 1 | MS | MS | 0.029 | 0.027 | -0.001 |
| `max_pool2d` | pool | 1 | MS | MS | 0.003 | 0.003 | +0.000 |
| `fused_conv2d1_add1_relu1` | 3x3 conv | 2 | MS | MS | 0.036 | 0.069 | +0.033 |
| `fused_conv2d1_add1_add2_relu1` | 3x3 conv | 2 | MS | MS | 0.042 | 0.055 | +0.013 |
| `fused_conv2d3_add3_relu2` | 3x3 conv | 1 | MS | MS | 0.018 | 0.073 | +0.055 |
| `fused_conv2d4_add3` | 1x1 conv | 1 | MS | MS | 0.010 | 0.044 | +0.034 |
| `fused_conv2d7_add5` | 1x1 conv | 1 | MS | MS | 0.009 | 0.010 | +0.001 |
| `fused_conv2d8_add7_relu4` | 3x3 conv | 1 | MS | MS | 0.045 | 0.061 | +0.015 |
| `fused_conv2d9_add7_relu4` | 3x3 conv | 1 | MS | MS | 0.059 | 0.080 | +0.021 |
| `fused_conv2d9_add7_add8_relu4` | 3x3 conv | 2 | MS | MS | 0.096 | 0.182 | +0.087 |
| `fused_conv2d10_add7` | 1x1 conv | 1 | MS | MS | 0.014 | 0.033 | +0.020 |
| `adaptive_avg_pool2d` | pool | 1 | MS | MS | 0.001 | 0.001 | +0.000 |
| `fused_conv2d2_add3_relu2` | 3x3 conv | 1 | MS | DLight | 0.019 | 0.082 | +0.063 |
| `fused_conv2d6_add5_relu3` | 3x3 conv | 1 | MS | DLight | 0.033 | 0.295 | +0.262 |
| `fused_conv2d6_add5_add6_relu3` | 3x3 conv | 2 | MS | DLight | 0.054 | 0.592 | +0.539 |
| `fused_conv2d3_add3_add4_relu2` | 3x3 conv | 2 | DLight | MS | 0.191 | 0.078 | -0.113 |
| `fused_conv2d5_add5_relu3` | 3x3 conv | 1 | DLight | MS | 0.094 | 0.045 | -0.049 |
| `fused_matmul_add9` | dense | 1 | DLight | MS | 0.003 | 0.038 | +0.035 |
| `reshape4` | reshape | 1 | DLight | DLight | - | - | - |

| Set | Calls | CUDA (ms) | SYCL (ms) | SYCL - CUDA |
|---|---:|---:|---:|---:|
| both MetaSchedule | 15 | 0.361 | 0.640 | +0.279 |
| SYCL fell back to DLight | 4 | 0.105 | 0.970 | +0.864 |
| CUDA fell back to DLight | 4 | 0.288 | 0.162 | -0.126 |
| no kernel | 1 | 0.000 | 0.000 | +0.000 |
| kernel time, all | 24 | 0.755 | 1.771 | +1.017 |
| idle between kernels | | 0.015 | 0.370 | +0.355 |
| host time outside the device span | | 0.021 | 0.050 | +0.029 |
| wall time per traced inference | | 0.790 | 2.191 | +1.401 |
