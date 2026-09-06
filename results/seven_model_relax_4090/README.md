# RTX 4090 Seven-Model Experiments (2026-08-10)

This directory contains the formal RTX 4090 seven-model evidence.

## Protocol and outcome

- Stack: TVM v0.21 Relax/TensorIR/VM with CUDA and SYCL backends.
- Models: ResNet-18, ResNet-34, ResNet-50, MLP, CNN, Transformer, and ViT-style.
- Paired sweep: batch sizes 1, 8, and 16; identical parameters, inputs, shapes, seeds, and tolerances for each CUDA/SYCL pair.
- Timing: 5 warm-ups and 30 synchronized measurements; reported latency is p50.
- Correctness: `allclose(atol=2e-4, rtol=2e-3)`.
- Result: all 21 pair descriptors match and all 42 backend runs pass.
- Observed SYCL/CUDA p50 ratio range: 1.65x--5.07x.
- ResNet sweep shape: standard `(B, 3, 224, 224)` with complete 1,000-class torchvision topologies.

The B1 scheduling-coverage campaign uses a nominal 1,024-trial MetaSchedule budget per model. Workloads without a finite measured record use DLight fallback; therefore, successful complete-model execution is not labeled as complete MetaSchedule coverage. ResNet scheduling coverage uses `32 x 32` inputs to bound tuning cost, independently of the standard-`224 x 224` paired sweep.

## Canonical summaries

- `all_seven/all_seven_4090_summary.md`: human-readable seven-model coverage and paired results.
- `all_seven/all_seven_4090_summary.json`: machine-audited checks, protocol, coverage, errors, and latencies.
- `all_seven/all_seven_4090_coverage.csv`: per-model scheduling coverage.
- `all_seven/all_seven_4090_sweep.csv`: all 21 paired CUDA/SYCL points.
- `unified_resnet/`: ResNet-only CSV/JSON/Markdown summaries.
- `unified_compact/`: compact-model CSV/JSON summaries.

## Formal raw archives

- `unified_resnet_formal_20260810.tar.gz`
  - SHA-256: `C37498C9A99FF75EDFB165C762C6F3B6479CDAF3380578A4B9135624F17486A7`
- `unified_compact_formal_20260810.tar.gz`
  - SHA-256: `B1B72E63FF5A6917618BBF27D1D03779F61BCEF980D50AB65D7C5BD681D0873C`
