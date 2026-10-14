# E1: whole-model gap decomposition (median over traced inferences)

| Target | B | wall p50 | launches | sum kernel | device span | inter-kernel idle | host-only | matched | sycl_only_ms | cuda_only_ms | neither |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda_b1 | 1 | 0.790 | 23 | 0.755 | 0.770 | 0.015 | 0.021 | 0.361 | 0.288 | 0.106 | 0.000 |
| sycl_b1 | 1 | 2.191 | 23 | 1.771 | 2.141 | 0.370 | 0.050 | 0.640 | 0.162 | 0.970 | 0.000 |
| sycl_r2async_b1 | 1 | 1.795 | 23 | 1.749 | 1.765 | 0.016 | 0.030 | 0.619 | 0.159 | 0.971 | 0.000 |
| cuda_b16 | 16 | 5.219 | 23 | 5.175 | 5.188 | 0.013 | 0.031 | 3.246 | 0.745 | 1.181 | 0.000 |
| sycl_b16 | 16 | 10.976 | 23 | 10.553 | 10.922 | 0.369 | 0.053 | 6.588 | 1.717 | 2.248 | 0.000 |
| sycl_r2async_b16 | 16 | 10.780 | 23 | 10.715 | 10.729 | 0.014 | 0.050 | 6.734 | 1.731 | 2.250 | 0.000 |

## sycl_b1 - cuda_b1 = 1.401 ms
| Component | ms | share |
|---|---:|---:|
| kernel time, matched workloads | 0.279 | 19.9% |
| kernel time, other workloads | 0.738 | 52.7% |
| inter-kernel idle | 0.355 | 25.3% |
| host-only (outside device span) | 0.029 | 2.1% |
| residual | -0.000 | |

## sycl_r2async_b1 - cuda_b1 = 1.005 ms
| Component | ms | share |
|---|---:|---:|
| kernel time, matched workloads | 0.258 | 25.7% |
| kernel time, other workloads | 0.736 | 73.3% |
| inter-kernel idle | 0.001 | 0.1% |
| host-only (outside device span) | 0.009 | 0.9% |
| residual | -0.000 | |

## sycl_b16 - cuda_b16 = 5.756 ms
| Component | ms | share |
|---|---:|---:|
| kernel time, matched workloads | 3.342 | 58.1% |
| kernel time, other workloads | 2.037 | 35.4% |
| inter-kernel idle | 0.356 | 6.2% |
| host-only (outside device span) | 0.022 | 0.4% |
| residual | -0.000 | |

## sycl_r2async_b16 - cuda_b16 = 5.560 ms
| Component | ms | share |
|---|---:|---:|
| kernel time, matched workloads | 3.488 | 62.7% |
| kernel time, other workloads | 2.053 | 36.9% |
| inter-kernel idle | 0.000 | 0.0% |
| host-only (outside device span) | 0.019 | 0.3% |
| residual | -0.000 | |

```json
{
 "cuda_b1": {
  "name": "cuda_b1",
  "target": "cuda",
  "batch": 1,
  "iters": 20,
  "wall_p50_ms": 0.7902165,
  "launches": 23,
  "kernel_ms": 0.7545955,
  "span_ms": 0.7695395,
  "idle_ms": 0.014912499999999995,
  "host_only_ms": 0.020676999999999945,
  "by_set_ms": {
   "cuda_only_ms": 0.105568,
   "matched": 0.3608185,
   "sycl_only_ms": 0.28829000000000005
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.001344,
   "fused_conv2d10_add7": 0.013568,
   "fused_conv2d1_add1_add2_relu1": 0.042064000000000004,
   "fused_conv2d1_add1_relu1": 0.036128,
   "fused_conv2d2_add3_relu2": 0.019376499999999998,
   "fused_conv2d3_add3_add4_relu2": 0.190913,
   "fused_conv2d3_add3_relu2": 0.018128,
   "fused_conv2d4_add3": 0.010112,
   "fused_conv2d5_add5_relu3": 0.0944325,
   "fused_conv2d6_add5_add6_relu3": 0.053567500000000004,
   "fused_conv2d6_add5_relu3": 0.032544,
   "fused_conv2d7_add5": 0.008528,
   "fused_conv2d8_add7_relu4": 0.045152,
   "fused_conv2d9_add7_add8_relu4": 0.095745,
   "fused_conv2d9_add7_relu4": 0.058768,
   "fused_conv2d_add_relu": 0.02864,
   "fused_matmul_add9": 0.0028955,
   "max_pool2d": 0.002752
  }
 },
 "sycl_b1": {
  "name": "sycl_b1",
  "target": "sycl",
  "batch": 1,
  "iters": 20,
  "wall_p50_ms": 2.1910350000000003,
  "launches": 23,
  "kernel_ms": 1.7711735,
  "span_ms": 2.141064,
  "idle_ms": 0.3698600000000001,
  "host_only_ms": 0.04997100000000021,
  "by_set_ms": {
   "cuda_only_ms": 0.9697145,
   "matched": 0.639778,
   "sycl_only_ms": 0.1617605
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.00144,
   "fused_conv2d10_add7": 0.03312,
   "fused_conv2d1_add1_add2_relu1": 0.055168499999999995,
   "fused_conv2d1_add1_relu1": 0.069088,
   "fused_conv2d2_add3_relu2": 0.082112,
   "fused_conv2d3_add3_add4_relu2": 0.07833599999999999,
   "fused_conv2d3_add3_relu2": 0.0734725,
   "fused_conv2d4_add3": 0.0443685,
   "fused_conv2d5_add5_relu3": 0.0453765,
   "fused_conv2d6_add5_add6_relu3": 0.592482,
   "fused_conv2d6_add5_relu3": 0.29496100000000003,
   "fused_conv2d7_add5": 0.009936,
   "fused_conv2d8_add7_relu4": 0.060608499999999996,
   "fused_conv2d9_add7_add8_relu4": 0.1824965,
   "fused_conv2d9_add7_relu4": 0.079952,
   "fused_conv2d_add_relu": 0.027264,
   "fused_matmul_add9": 0.038048,
   "max_pool2d": 0.0029115
  }
 },
 "sycl_r2async_b1": {
  "name": "sycl_r2async_b1",
  "target": "sycl",
  "batch": 1,
  "iters": 20,
  "wall_p50_ms": 1.7947865,
  "launches": 23,
  "kernel_ms": 1.748521,
  "span_ms": 1.7647935000000001,
  "idle_ms": 0.016384999999999983,
  "host_only_ms": 0.029992999999999936,
  "by_set_ms": {
   "cuda_only_ms": 0.9712059999999999,
   "matched": 0.6185485,
   "sycl_only_ms": 0.158624
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.00144,
   "fused_conv2d10_add7": 0.02976,
   "fused_conv2d1_add1_add2_relu1": 0.054447999999999996,
   "fused_conv2d1_add1_relu1": 0.06824000000000001,
   "fused_conv2d2_add3_relu2": 0.082145,
   "fused_conv2d3_add3_add4_relu2": 0.077904,
   "fused_conv2d3_add3_relu2": 0.0729445,
   "fused_conv2d4_add3": 0.043456,
   "fused_conv2d5_add5_relu3": 0.045152,
   "fused_conv2d6_add5_add6_relu3": 0.5915714999999999,
   "fused_conv2d6_add5_relu3": 0.2975695,
   "fused_conv2d7_add5": 0.0099525,
   "fused_conv2d8_add7_relu4": 0.060321,
   "fused_conv2d9_add7_add8_relu4": 0.169344,
   "fused_conv2d9_add7_relu4": 0.078721,
   "fused_conv2d_add_relu": 0.0270245,
   "fused_matmul_add9": 0.035536,
   "max_pool2d": 0.00288
  }
 },
 "cuda_b16": {
  "name": "cuda_b16",
  "target": "cuda",
  "batch": 16,
  "iters": 20,
  "wall_p50_ms": 5.2193305,
  "launches": 23,
  "kernel_ms": 5.1746415,
  "span_ms": 5.188079999999999,
  "idle_ms": 0.013488499999999792,
  "host_only_ms": 0.031250500000000514,
  "by_set_ms": {
   "cuda_only_ms": 1.1806945,
   "matched": 3.2459875,
   "sycl_only_ms": 0.7451715000000001
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.004768,
   "fused_conv2d10_add7": 0.020304000000000003,
   "fused_conv2d1_add1_add2_relu1": 0.5143715,
   "fused_conv2d1_add1_relu1": 0.517379,
   "fused_conv2d2_add3_relu2": 0.141889,
   "fused_conv2d3_add3_add4_relu2": 0.5478745,
   "fused_conv2d3_add3_relu2": 0.275138,
   "fused_conv2d4_add3": 0.0235685,
   "fused_conv2d5_add5_relu3": 0.174033,
   "fused_conv2d6_add5_add6_relu3": 0.691076,
   "fused_conv2d6_add5_relu3": 0.346626,
   "fused_conv2d7_add5": 0.021248,
   "fused_conv2d8_add7_relu4": 0.2207855,
   "fused_conv2d9_add7_add8_relu4": 0.922998,
   "fused_conv2d9_add7_relu4": 0.460867,
   "fused_conv2d_add_relu": 0.2401615,
   "fused_matmul_add9": 0.023296499999999998,
   "max_pool2d": 0.023344
  }
 },
 "sycl_b16": {
  "name": "sycl_b16",
  "target": "sycl",
  "batch": 16,
  "iters": 20,
  "wall_p50_ms": 10.975626,
  "launches": 23,
  "kernel_ms": 10.553244,
  "span_ms": 10.922287,
  "idle_ms": 0.3691880000000012,
  "host_only_ms": 0.05333899999999936,
  "by_set_ms": {
   "cuda_only_ms": 2.2479014999999998,
   "matched": 6.5877825,
   "sycl_only_ms": 1.7168584999999998
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.00896,
   "fused_conv2d10_add7": 0.047904,
   "fused_conv2d1_add1_add2_relu1": 1.1891755,
   "fused_conv2d1_add1_relu1": 1.1926459999999999,
   "fused_conv2d2_add3_relu2": 0.337314,
   "fused_conv2d3_add3_add4_relu2": 1.3423124999999998,
   "fused_conv2d3_add3_relu2": 0.6632195000000001,
   "fused_conv2d4_add3": 0.053872,
   "fused_conv2d5_add5_relu3": 0.325074,
   "fused_conv2d6_add5_add6_relu3": 1.2719585,
   "fused_conv2d6_add5_relu3": 0.6363719999999999,
   "fused_conv2d7_add5": 0.0477605,
   "fused_conv2d8_add7_relu4": 0.398498,
   "fused_conv2d9_add7_add8_relu4": 1.5542660000000001,
   "fused_conv2d9_add7_relu4": 0.7735080000000001,
   "fused_conv2d_add_relu": 0.627459,
   "fused_matmul_add9": 0.0504,
   "max_pool2d": 0.02896
  }
 },
 "sycl_r2async_b16": {
  "name": "sycl_r2async_b16",
  "target": "sycl",
  "batch": 16,
  "iters": 20,
  "wall_p50_ms": 10.779539,
  "launches": 23,
  "kernel_ms": 10.7154615,
  "span_ms": 10.729218,
  "idle_ms": 0.013886000000000287,
  "host_only_ms": 0.05032100000000028,
  "by_set_ms": {
   "cuda_only_ms": 2.2496205,
   "matched": 6.7340824999999995,
   "sycl_only_ms": 1.7307799999999998
  },
  "by_workload_ms": {
   "adaptive_avg_pool2d": 0.008991,
   "fused_conv2d10_add7": 0.048427,
   "fused_conv2d1_add1_add2_relu1": 1.1915395000000002,
   "fused_conv2d1_add1_relu1": 1.2010904999999998,
   "fused_conv2d2_add3_relu2": 0.33508499999999997,
   "fused_conv2d3_add3_add4_relu2": 1.360371,
   "fused_conv2d3_add3_relu2": 0.680377,
   "fused_conv2d4_add3": 0.055162,
   "fused_conv2d5_add5_relu3": 0.31943900000000003,
   "fused_conv2d6_add5_add6_relu3": 1.275915,
   "fused_conv2d6_add5_relu3": 0.6386529999999999,
   "fused_conv2d7_add5": 0.047307,
   "fused_conv2d8_add7_relu4": 0.39671049999999997,
   "fused_conv2d9_add7_add8_relu4": 1.6420689999999998,
   "fused_conv2d9_add7_relu4": 0.8032445,
   "fused_conv2d_add_relu": 0.6306705,
   "fused_matmul_add9": 0.050523,
   "max_pool2d": 0.029405
  }
 }
}
```
