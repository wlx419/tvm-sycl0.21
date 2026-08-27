# TVM v0.21 SYCL Source Overlay

This directory contains the TVM source files changed for the updated SYCL
backend. Apply the files over an Apache TVM v0.21 source tree while preserving
their relative paths.

## Relevant Integration Changes

- register SYCL/oneAPI as the standard DLPack `kDLOneAPI` runtime device;
- route `sycl` targets through Relax's generic GPU pipelines;
- recognize `sycl` as a GPU target in MetaSchedule;
- derive device limits from the selected SYCL device rather than compile-time
  platform macros;
- keep JIT-compiled device images loaded for the process lifetime and give each
  generated wrapper a module-unique symbol;
- create in-order SYCL queues while retaining explicit error-propagating waits;
- lower float32 reciprocal-square-root, tanh, and erf expressions through forms
  validated on both CUDA-backed NVIDIA and HIP-backed DCU plugins;
- propagate synchronous and asynchronous SYCL failures as errors instead of
  allowing a failing call to terminate with a successful process status; and
- support MetaSchedule's local runner by building the RPC runtime components.

## Build Requirements

The validated build enables LLVM, CUDA, RPC, and SYCL. `USE_SYCL` is the path
to the DPC++ compiler. A representative configuration is:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_LLVM=/path/to/llvm-config \
  -DUSE_CUDA=ON \
  -DUSE_RPC=ON \
  -DUSE_SYCL=/path/to/icpx
cmake --build build -j
```

For SYCL-on-NVIDIA, the oneAPI Unified Runtime CUDA adapter must be visible in
`LD_LIBRARY_PATH`, and `ONEAPI_DEVICE_SELECTOR=cuda:*` selects the CUDA device.
For the evaluated DCU build, `USE_SYCL_HIP` targets
`amdgcn-amd-amdhsa` with `--offload-arch=gfx906`.

## Unified ResNet Validation

Use `experiments/resnet_relax_probe.py` for individual runs and
`experiments/run_unified_resnet_4090.sh` for the formal ResNet-18/34/50 matrix.
The latter performs B1 MetaSchedule coverage runs, executes B1/B8/B16 on both
CUDA and SYCL through Relax VM, and validates every output against the same
PyTorch CPU model.
