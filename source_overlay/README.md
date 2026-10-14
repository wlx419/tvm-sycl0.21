# TVM v0.21 SYCL Source Overlay

This directory contains the TVM source files changed for the SYCL backend. Apply the files over an
Apache TVM v0.21 source tree while preserving their relative paths (see the top-level README). Since
v2.0 the overlay also adds new files: the SYCL command-graph builtins
(`src/runtime/sycl/sycl_graph_builtin.cc`), the sub-group intrinsics
(`src/target/source/intrin_rule_sycl.cc`), the Relax SYCL pipeline (`python/tvm/relax/backend/sycl/`),
and changed TVM files for reductions, DLight, device-module grouping and target detection.

## Changes in v2.0

Every behavioural change of v2.0 has a switch that restores the behaviour of v1 (commit `472a43e`),
so that each change can be measured on its own. Switches are environment variables read when a
module is compiled or loaded, unless stated otherwise. The measurements cited below are in
`results/round2/`.

| ID | Files | Change | Switch restoring v1 |
|---|---|---|---|
| A1 | runtime/sycl/sycl_module.cc | No host wait after each launch; the in-order queue orders work; synchronization at copies, frees and `StreamSync` | `TVM_SYCL_SYNC_EACH_LAUNCH=1` |
| A2 | runtime/sycl/sycl_module.cc | Wrapper symbol resolved once per function (was `dlsym` per launch) | none |
| A3 | runtime/sycl/sycl_graph_builtin.cc (new), python/tvm/relax/backend/sycl (new), python/tvm/relax/pipeline.py, python/tvm/relax/backend/\_\_init\_\_.py | SYCL command-graph replay of `RewriteCUDAGraph` regions (`vm.builtin.sycl_graph.*`, sycl_ext_oneapi_graph record/replay); the redirect pass is also in the default build pipeline | omit `relax.backend.use_cuda_graph` (default off) |
| B1 | runtime/sycl/sycl_common.h, sycl_module.cc | Content-addressed JIT libraries (hash of source, flags, compiler), atomic publication, reuse across modules and processes | `TVM_SYCL_JIT_CACHE=0` |
| B2 | runtime/sycl/sycl_module.cc, target/source/codegen_sycl.cc | Optional split compilation into N translation units (`-fno-sycl-rdc`); default 1 because the link costs more than it saves | `TVM_SYCL_JIT_JOBS=1` (default) |
| B2' | python/tvm/tir/build.py | One device module per target (functions were grouped by Target object identity, i.e. one device-compiler run per kernel) | `TVM_SPLIT_DEVICE_MODULE_PER_FUNC=1` |
| C1 | tir/transforms/lower_thread_allreduce.cc, target/source/codegen_sycl.cc | SYCL allreduce without in-warp phase; a residual `storage_sync("warp")` is a compile error | `TVM_SYCL_LEGACY_WARP_SYNC=1` |
| C2 | target/source/codegen_sycl.cc | Over-budget group-local allocation is a compile error (was trimmed with a warning) | none |
| C3 | runtime/sycl/sycl_module.cc, target/source/codegen_sycl.cc | Grid range reversed like the block range; `blockIdx.d` read from SYCL dimension 2-d (removes the 65,535-block limit on `blockIdx.x` on NVIDIA) | none |
| C4 | runtime/sycl/sycl_module.cc | DPC++ compile status checked | none |
| C5 | tir/transforms/lower_thread_allreduce.cc, target/source/intrin_rule_sycl.cc (new), codegen_sycl.cc | Optional warp-level reductions with SYCL 2020 sub-group shuffles | opt-in `TVM_SYCL_SUBGROUP_REDUCE=1` |
| C6 | python/tvm/target/detect_target.py, runtime/sycl/sycl_device_api.cc | `Target.from_device` for SYCL: work-group limit, sub-group size, local memory (capped at the 48 KB static limit), L2, `arch` behind the CUDA plugin | explicit target string |
| C7 | target/source/codegen_sycl.{cc,h} | Group-local budget from the target (`max_shared_memory_per_block`, at most 48 KB) | none |
| C8 | target/source/codegen_sycl.cc | Vector loads/stores whose width has no matching `sycl::vec` layout (3 lanes: `vec<T,3>` is 16-byte sized and aligned, CUDA's `float3` 12/4) are emitted element-wise; the cast form faulted with a misaligned address and touched a fourth element | none |
| C9 | target/source/codegen_sycl.cc | Group-local buffers allocated with `group_local_memory_for_overwrite`: `group_local_memory<T>(g)` value-initializes T, i.e. work-item 0 zero-fills every shared buffer byte by byte and the work-group waits at a barrier, at the start of every work-group | `TVM_SYCL_LOCAL_ZERO_INIT=1` |
| C10 | target/source/codegen_sycl.{cc,h} | `[[intel::max_work_group_size(z,y,x)]]` from the static thread extents; DPC++ lowers it to PTX `.maxntid`, the directive CUDA's `__launch_bounds__` produces, so ptxas budgets registers for the real work-group size (DPC++ 646ab08 does not lower `reqd_work_group_size` to `.reqntid`) | `TVM_SYCL_LAUNCH_BOUNDS=0` |
| C11 | target/source/codegen_sycl.{cc,h} | Kernels that use sub-group shuffles carry `[[sycl::reqd_sub_group_size(W)]]`, W = target `thread_warp_size` | none |
| C12 | target/source/codegen_sycl.cc, runtime/sycl/*.cc | DPC++ 2022 compatibility (DCU): `TVM_SYCL_AS_CAST` with `-DTVM_SYCL_NO_ADDRESS_SPACE_CAST`; libsycl < 6 guards for `max_work_item_sizes` and `backend_version`; command graphs only with `SYCL_EXT_ONEAPI_GRAPH`; `kMaxThreadDimensions` reported in TVM x,y,z order | build flag |
| C13 | tir/transforms/lower_thread_allreduce.cc, target/source/codegen_sycl.cc | Single-value add/max/min reductions over a whole work-group lowered to SYCL 2020 `sycl::reduce_over_group` | opt-in `TVM_SYCL_GROUP_REDUCE=1` |
| D1 | python/tvm/dlight/base/utils.py, dlight/gpu/*.py | `schedule_kind(target)`: SYCL with `-arch=sm_XY` uses CUDA's DLight configurations, `-arch=gfx...` ROCm's | omit `-arch` |
| F1 | runtime/sycl/sycl_common.h | For `nvptx64` JIT targets, `-ffp-contract=fast`, the contraction nvcc applies by default (`--fmad=true`): DPC++ contracts only within one expression, and `sycl::vec` arithmetic is a chain of operator calls, so DLight's `acc = acc + a * b` compiled to separate multiplies and adds (0 vs 256 FMAs in a Transformer matmul kernel) | `TVM_SYCL_FP_CONTRACT=0` |
| F2 | target/source/codegen_sycl.cc | Floating-point `min`/`max` emitted as `sycl::fmin`/`sycl::fmax`, the counterparts of CUDA's `fminf`/`fmaxf`; `sycl::min`/`max` lower to a compare and a select | `TVM_SYCL_FMINMAX=0` |
| J1 | runtime/sycl/sycl_common.h | For `nvptx64` JIT targets, `-enable-global-offset=false`: DPC++ otherwise gives every kernel an implicit global-offset parameter and a 16-byte local stack frame (SYCL 1.2.1 offsets, which TVM never uses); `TVM_SYCL_EXTRA_FLAGS` appends device-compiler flags | `TVM_SYCL_GLOBAL_OFFSET=1` |
| L1 | target/source/codegen_sycl.cc | Kernels submitted through `TVM_SYCL_LAUNCH`: `sycl::khr::launch_grouped` (sycl_khr_free_function_commands, enabled with `__DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS`) where the DPC++ headers provide it, which DPC++ v7 submits without a command-group handler and without an event (2.5 us per launch on the RTX 4090, against 4.0 us through `queue::submit`); `queue::submit` + `handler::parallel_for` otherwise (DPC++ 646ab08 and the 2022 HIP build have no such path, so their code is unchanged) | `TVM_SYCL_DIRECT_SUBMIT=0` |
| M1 | target/source/codegen_sycl.{cc,h} | The float32 `erf`-through-float64 and sign-stable `tanh` forms (workarounds for HIP-plugin defects) are emitted only when the target `arch` is not `sm_*`; behind the CUDA plugin the kernels call `sycl::erf`/`sycl::tanh`, as CUDA calls `erff`/`tanhf` (the float64 `erf` made a GELU kernel 7.5 times slower than CUDA's) | `TVM_SYCL_MATH_WORKAROUNDS=1` |
| N1 | target/source/codegen_sycl.{cc,h} | Opt-in: thread and block indices read from the 32-bit PTX registers (masked to 31 bits) instead of `nd_item`'s sign-extended `size_t`; halves the registers of DLight matmul and convolution kernels (123 -> 64) and brings a 4096^3 matmul and a ResNet stage-1 convolution to 1.00x / 1.04x of TVM-CUDA (1.07x / 1.21x without), but ResNet-18 schedules tuned with `nd_item` indices run 5-24% slower, and a retune with N1 does not recover it (0.571 vs 0.530 ms) | default off; `TVM_SYCL_NATIVE_IDS=1` (`hw`: hardware-range masks) |
| P1 | target/target_kind.cc, target/source/codegen_sycl.{cc,h} | `arch` attribute for SYCL targets (enables MetaSchedule's multi-stage software pipelining, as for CUDA); async-queue scopes emitted as plain stage bodies | omit `-arch` from the SYCL target string |
| R1 | runtime/sycl/sycl_common.h, sycl_device_api.cc, sycl_graph_builtin.cc | Optional `discard_events` queue; copies and graph replay wait on the queue | opt-in `TVM_SYCL_DISCARD_EVENTS=1` |
| R2 | runtime/sycl/sycl_graph_builtin.cc | Executable command graphs are finalized again every 32 replays with SYCL runtimes before libsycl 9: DPC++ 646ab08 keeps the event of every replay and checks all of them before the next one, so a replay's host cost grows by about 0.25 us per earlier replay (17 us over the first 100 replays, 500 us after 1,000 on the RTX 4090); re-finalizing bounds it at 10-13 us. DPC++ v7.1.1 (libsycl 9) replays in a constant 3.4-3.7 us, so the default period is 0 there | `TVM_SYCL_GRAPH_REFINALIZE=0` (period: any N) |
| T1 | experiments/round2/resnet_relax_probe_r2.py (`--runner-recycle-on-error`) | The tuning runner replaces its measurement process after a failed candidate: a device fault leaves the process's CUDA context unusable, and the v1 SYCL tuning lost 335 of 1,024 candidates to the resulting "device allocation failed" errors | omit the option |
| T2 | runtime/sycl/sycl_common.h, sycl_device_api.cc | The SYCL `time_evaluator` timer synchronizes the queue at Start/Stop (the inherited timer read the host clock only: per-launch-wait time with a wait after each launch, enqueue time after asynchronous dispatch) | none |

The commented-out code inherited from the RELOAD22 branch (about 300 lines in `codegen_sycl.cc`,
and single statements in the runtime) was removed.

**Regression tests.** `tests/python/codegen/test_target_codegen_sycl_r2.py` (needs a SYCL device)
pins C1, C2, C3, C8, C9, C10, F2, M1, N1 and L1 on single kernels. Graph replay and
re-finalization (A3, R2) are checked end to end by `experiments/round2/compact_probe_r2.py`, which
compares the output again after the timed runs. Without pytest (the DCU container),
`experiments/round2/run_r2_tests_nopytest.py` runs the same tests. All 9 pass on the RTX 4090 with
DPC++ 646ab08 and v7.1.1 and on the DCU with the 2022 HIP build.

## Integration Changes of v1

- register SYCL/oneAPI as the standard DLPack `kDLOneAPI` runtime device;
- route `sycl` targets through Relax's generic GPU pipelines;
- recognize `sycl` as a GPU target in MetaSchedule;
- derive device limits from the selected SYCL device rather than compile-time
  platform macros;
- keep JIT-compiled device images loaded for the process lifetime and give each
  generated wrapper a module-unique symbol;
- create in-order SYCL queues (v1 kept an error-propagating wait after each launch; see A1);
- lower float32 reciprocal-square-root, tanh, and erf expressions through forms
  validated on both CUDA-backed NVIDIA and HIP-backed DCU plugins (see M1);
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
`amdgcn-amd-amdhsa` with `--offload-arch=gfx906`, and DPC++ 2022 needs
`-DTVM_SYCL_NO_ADDRESS_SPACE_CAST` in `SYCL_FLAGS` (C12).

The evaluated DPC++ builds are intel/llvm commit `646ab08` (clang 18, SYCL runtime 7.2) with the CUDA
plugin, the intel/llvm v7.1.1 release (clang 22, SYCL runtime 9; built from source with
`experiments/round2/build_dpcpp_v711.sh`) with the CUDA plugin, and a 2022 build with the HIP plugin.
`cmake/modules/SYCL.cmake` is unchanged from v1; the measurement hosts used their own
platform-specific copies of this module.

## Unified ResNet Validation (v1)

Use `experiments/resnet_relax_probe.py` for individual runs and
`experiments/run_unified_resnet_4090.sh` for the formal ResNet-18/34/50 matrix.
The latter performs B1 MetaSchedule coverage runs, executes B1/B8/B16 on both
CUDA and SYCL through Relax VM, and validates every output against the same
PyTorch CPU model. The v2.0 counterparts are `experiments/round2/resnet_relax_probe_r2.py` and
the timing-window scripts described in `experiments/round2/README.md`.
