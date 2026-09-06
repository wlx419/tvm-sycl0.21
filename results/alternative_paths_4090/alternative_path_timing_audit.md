# Alternative-path timing audit

Date: 2026-08-11
Hardware: NVIDIA GeForce RTX 4090

This audit checks whether the available IREE and oneDNN executions support a
fair numerical speedup comparison with TVM-SYCL kernel p50 latency.

## IREE Vulkan

- Version: IREE 3.11.0rc20260316
- Backend: Vulkan/SPIR-V
- Target-specific rebuild: `--iree-vulkan-target=ampere`
- Inputs: saved NumPy arrays matching the four operator shapes
- Metric: `process_time` reported by `iree-benchmark-module`

| Operator | Ampere, batch size 1 (ms) | Ampere, batch size 100 (ms/invocation) |
|---|---:|---:|
| Matmul | 1.522704 | 0.0150071 |
| Row-sum | 1.497854 | 0.0148697 |
| Softmax | 1.555365 | 0.0155939 |
| NHWC conv2d | 2.729656 | 0.0273120 |

The approximately 100-fold difference tracks invocation batching, while an
Ampere-targeted rebuild changes batch-size-1 latency only marginally relative
to the generic Vulkan modules. For these small operators, host and VM call
amortization dominates a ratio against a kernel-only p50.

## oneDNN generic control

- Version: oneDNN 3.9.0
- Runtime: DPC++ GPU runtime
- Device: NVIDIA GeForce RTX 4090
- Timed scope: primitive execution plus stream wait; construction and transfers
  are outside the timed region
- Verbose implementation for every measured primitive: `gpu:0,dpcpp:ref:any`

| Operator | Repeated p50 (ms) |
|---|---:|
| Matmul | 12.230256 |
| Row-sum | 3.082611 |
| Softmax | 1.432656 |
| NHWC conv2d | 10.141068 |

All four outputs pass their reference checks. These timings characterize the
generic oneDNN reference implementation selected in this control build, not a
vendor-tuned oneDNN primitive path. The subsequent dedicated NVIDIA build and
matched five-repeat experiment are stored under
`onednn_nvidia_comparison_20260811/`; that experiment selects
`backend:Nvidia` and `cuda:cudnn:any` and supersedes the generic-control values
for the reported comparison.

## Reporting decision

The IREE measurements are retained quantitatively at both invocation batch
sizes, because they document batching sensitivity and observed deployment
behavior. Neither IREE point is converted into a TVM/IREE speedup: such a
ratio would mix IREE host/VM invocation semantics with synchronized TVM-SYCL
device submission and completion. The oneDNN NVIDIA vendor result is reported
separately under its matched timing protocol. The resulting comparison is
multi-dimensional and operator-specific rather than a universal ranking.
