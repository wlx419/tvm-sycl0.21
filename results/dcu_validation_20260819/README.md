# DCU Operator and Seven-Model Validation

Date: 2026-08-19

This directory contains the final DCU Z100L validation records used in the
manuscript and reviewer response. TVM reports version 0.21.0 and is built for
the HIP-backed DPC++ target `gfx906`. Cases run in isolated subprocesses with
one visible accelerator per process.

## Operator Validation

The extended probe covers 16 operator-shape configurations drawn from nine
categories: matmul, batched matmul, row-wise reduction, layer normalization,
NHWC conv2d, NHWC max-pooling, GELU, 3D transpose, and a three-kernel softmax.
The complete set was repeated on both installed Z100L cards.

| Device | Passed | Total | Largest max abs | LayerNorm max abs | GELU max abs |
|---|---:|---:|---:|---:|---:|
| 0 | 16 | 16 | 5.9128e-5 | 1.1921e-6 | 4.7684e-7 |
| 1 | 16 | 16 | 5.9128e-5 | 1.1921e-6 | 4.7684e-7 |

The final records are under `operators_device0_codegenfix/` and
`operators_device1_codegenfix/`. The diagnostic JSON files retain the
before/after evidence for reciprocal square root, tanh, and erf lowering.

## Model Validation

All model cases use five warm-ups, 30 synchronized measurements, and
`allclose(atol=2e-4, rtol=2e-3)`. The sweep uses the TVM v0.21
Relax/TensorIR/VM stack with the static-shape pipeline, DLight schedules, and
no search trials. It is therefore correctness and portability evidence, not a
tuned cross-device comparison.

| Model | B1 p50 (ms) | B8 p50 (ms) | B16 p50 (ms) | Worst max abs |
|---|---:|---:|---:|---:|
| ResNet-18 | 10.634 | 19.887 | 35.600 | 7.63e-6 |
| ResNet-34 | 21.691 | 39.263 | 70.566 | 3.81e-4 |
| ResNet-50 | 17.938 | 44.726 | 81.291 | 1.41e-4 |
| MLP | 0.286 | 0.456 | 0.467 | 2.42e-8 |
| CNN | 0.355 | 0.345 | 0.349 | 5.09e-11 |
| Transformer | 1.358 | 1.486 | 1.390 | 2.98e-8 |
| ViT-style | 1.293 | 1.875 | 1.532 | 7.45e-8 |

All 21 model-batch cases pass. The ResNet files use standard 224x224 inputs,
folded BatchNorm, an ONNX-to-Relax import, and PyTorch CPU references. The four
compact Relax models use NumPy references.

## Provenance

`source_snapshot_20260819/` records the three backend files and build
configuration used for the remote run. The publishable source is the cleaned
`source_overlay/` at the package root; it contains the same tested in-order
queue and math-lowering changes without unrelated host-side debug code.
