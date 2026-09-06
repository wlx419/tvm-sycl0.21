# Paired DCU sweep: SYCL backend vs TVM ROCm/HIP backend (2026-09-04)

Adds the TVM-HIP baseline that the DCU results previously lacked, so the DCU
evaluation has the same paired structure as the NVIDIA CUDA/SYCL sweep.

## Environment

- Host: DCU node, container `tvm_sycl_test`
  (`pytorch:2.1.0-ubuntu20.04-dtk24.04.1-py3.10`).
- Device: 2x Sugon Z100L, `gfx906:sramecc+:xnack-`, DTK 24.04.1. All runs on
  device 0.
- TVM v0.21 source tree: `/home/wlx/tvm-sycl-build/apache-tvm-src-sycl-v0.21.0`.

### Why a new build was needed

The existing `build/` has `USE_ROCM OFF`, so `tvm.device("rocm", 0).exist` is
false and only the ROCm *codegen* is present, not the runtime. A ROCm-enabled
library was built in a separate directory, `/root/tvm_rocm_build`, and the
existing `build/libtvm.so` was left untouched, because every previously
published DCU SYCL result depends on that library.

Two build notes, both handled by `scripts/dcu_build_rocm.sh`:

- `USE_SYCL` must stay on. This fork guards `add_subdirectory(ffi)` behind
  `if(USE_SYCL)`, so disabling SYCL leaves `add_dsymutil()` undefined. The
  resulting library therefore carries both backends.
- `cmake` on `PATH` is 3.19 and `FindLLVM.cmake` needs `cmake_path` (>= 3.20);
  the pip-installed 3.31 is used instead.

`onnx` was missing and installing it pulled NumPy 2.x, which breaks the
container's torch 2.1. NumPy was pinned back to 1.26.4.

## Protocol

Mirrors the published DCU SYCL sweep: `static_shape_tuning` pipeline with 0
trials (DLight fallback, no search), 5 warm-ups, 30 synchronized measurements,
device 0. Compact models use seed 123 and NumPy references; ResNets use seed
321, the ONNX frontend, 224x224 inputs, 1000 classes, folded BatchNorm, and
PyTorch CPU references.

Both backends were run from the same `libtvm.so`, so the comparison is not
confounded by a build difference.

Input pairing: the compact-model seed depends on the target name, so the ROCm
runs pass `--seed-target-alias sycl` to get bit-identical inputs. The ResNet
seed does not depend on the target. Pairing was confirmed by the maximum
absolute errors agreeing exactly across backends, for example resnet34 B1 at
5.91278076171875e-05 on both.

## Results (p50, ms, 30 measurements)

| model | B | ROCm/HIP | SYCL | SYCL/HIP | ROCm status |
|---|---|---|---|---|---|
| resnet18 | 1 | 7.619 | 10.640 | 1.40x | pass |
| resnet18 | 8 | 14.428 | 19.889 | 1.38x | pass |
| resnet18 | 16 | 21.425 | 35.114 | 1.64x | pass |
| resnet34 | 1 | 15.406 | 21.691 | 1.41x | pass |
| resnet34 | 8 | 27.927 | 39.236 | 1.40x | pass |
| resnet34 | 16 | 42.050 | 70.499 | 1.68x | pass |
| resnet50 | 1 | 11.312 | 17.910 | 1.58x | pass |
| resnet50 | 8 | 25.456 | 44.803 | 1.76x | pass |
| resnet50 | 16 | 39.828 | 81.521 | 2.05x | pass |
| mlp | 1 | 0.075 | 0.292 | 3.91x | pass |
| mlp | 8 | 0.204 | 0.445 | 2.18x | pass |
| mlp | 16 | 0.202 | 0.462 | 2.28x | pass |
| cnn | 1 | 0.114 | 0.303 | 2.66x | pass |
| cnn | 8 | 0.139 | 0.348 | 2.50x | pass |
| cnn | 16 | 0.227 | 0.299 | 1.32x | pass |
| transformer | 1 | 0.183 | 1.358 | n/a | fail_tolerance (nonfinite) |
| transformer | 8 | 0.190 | 1.140 | n/a | fail_tolerance (nonfinite) |
| transformer | 16 | 0.199 | 1.349 | n/a | fail_tolerance (nonfinite) |
| vit | 1 | 0.265 | 1.261 | n/a | fail_tolerance (nonfinite) |
| vit | 8 | 0.348 | 1.496 | n/a | fail_tolerance (nonfinite) |
| vit | 16 | 0.373 | 1.825 | n/a | fail_tolerance (nonfinite) |

15 of 21 cases pass on both backends. On those, SYCL is 1.32x-3.91x slower than
the TVM ROCm backend; the ResNet range, 1.38x-2.05x, is comparable to the
1.65x-2.36x SYCL/CUDA ResNet range measured on the RTX 4090.

Ratios are only computed where both sides pass validation. The ROCm
Transformer and ViT timings are listed for completeness but are not comparable,
because those runs do not produce correct output.

## SYCL reproducibility check

The SYCL half was re-run from the new library and reproduces the published
2026-08-19 numbers, which validates both the earlier results and this build:

| case | published | re-run | delta |
|---|---|---|---|
| resnet18 B1 | 10.634 | 10.640 | +0.06% |
| resnet34 B1 | 21.691 | 21.691 | 0.00% |
| resnet50 B1 | 17.938 | 17.910 | -0.15% |
| resnet50 B16 | 81.291 | 81.521 | +0.28% |
| transformer B1 | 1.358 | 1.358 | 0.00% |

## ROCm nonfinite outputs on Transformer and ViT: localized to `layer_norm`

Both attention models produce nonfinite outputs on the TVM ROCm backend at all
three batch sizes, while passing on the SYCL backend on the same device, same
library, and same inputs. MLP, CNN, and all three ResNets pass on both.

The failure was localized by instrumenting the Relax VM of the actual failing
module (`scripts/dcu_bisect_nonfinite.py`), rather than by testing synthesized
primitives. The hook inspects every kernel call and flags the first one handed
finite inputs that returns a nonfinite output. TIR kernels are
destination-passing, so the output buffer is read from the arguments after the
call, and the device is synchronized first.

**`layer_norm` is that kernel**, in both models:

| model | position in kernel sequence | layer_norm output |
|---|---|---|
| transformer B1 | kernel 0 of 26 | 256 Inf of 1024 |
| vit B1 | kernel 3 of 29 | 256 Inf of 1024 |

Its inputs are finite (x absmax 0.3625, weight absmax 1, bias all zero) and
every later kernel is NaN purely by propagation.

### What the failure is

- Deterministic and localized to fixed columns: 8-11, 24-27, 40-43, 56-59, that
  is 16 of every 64, identical in every row. Equivalently, the failing lanes are
  those with `lane % 16` in `{8, 9, 10, 11}`.
- The other 768 elements are correct to 4.77e-07.
- Not a measurement artifact: an explicit `dev.sync()` before reading changes
  nothing.
- Not a missing store. Pre-filling `layer_norm`'s output buffer with a sentinel
  before the call (`scripts/dcu_store_sentinel.py`) leaves no sentinel behind,
  so all 64 lanes are written and the kernel genuinely computes Inf.

## Root cause: `v_sqrt_f32` on the DCU, not the allreduce

The reduction is innocent. The defect is a read-after-write hazard on the DCU's
transcendental unit that neither TVM backend nor the DTK compiler guards
against.

**`v_sqrt_f32` returns exactly 0, instead of the correct result, on the lanes
with `lane % 16` in `{8, 9, 10, 11}`, whenever its operand register was written
by the immediately preceding VALU instruction.** When the operand instead comes
from memory, so that an `s_waitcnt` separates producer from consumer, the result
is correct. No `s_nop` appears anywhere in the generated code.

`layer_norm` hits this because TVM's ROCm backend expands `rsqrt(var + eps)`
into `1.0 / sqrt(...)`, and the `v_add_f32` that forms `var + eps` lands
directly on the `v_sqrt_f32` that consumes it. `sqrt` returns 0, the reciprocal
turns it into Inf, and Inf propagates to the output.

### How this was established

Each step measured the real kernel or a minimal replica rather than reasoning
from the source; `scripts/dcu_layernorm_stages.py` and
`scripts/dcu_sqrt_mechanism.py` are the two decisive ones.

1. **The allreduce is correct.** A replica of `layer_norm` built with the same
   two-value `comm_reducer` and DLight schedule exports the reduced sum and sum
   of squares as per-lane buffers. Both are bit-identical across all 64 lanes
   and match NumPy. Only `1/sqrt(var+eps)` diverges: 48 lanes give 10.958627,
   16 give Inf.
2. **The operand is correct on the failing lanes.** On the same operand,
   `1.0/v`, `v/3.0`, `v*2.0` and `exp(v)` are all correct on all 64 lanes, so
   the value in those lanes' registers is right; only `sqrt` and `log` return 0.
3. **`sqrt` itself is fine.** A plain elementwise kernel that loads the operand
   from global memory computes `sqrt` and `rsqrt` correctly at every magnitude
   from 1e-7 to 1e3, with `v_sqrt_f32` emitted.
4. **The producer of the operand is what matters.** Five routes to the same
   value 1e-5 in one kernel: operands arriving from a load pass, operands formed
   by a VALU add or fmac fail. The dumped ISA shows the split directly:
   `global_load_dword ; s_waitcnt vmcnt(0) ==> v_sqrt_f32` passes, while
   `v_add_f32_e32 v1, v1, v2 ==> v_sqrt_f32_e32 v1, v1` and
   `v_fmac_f32_e32 v2, 0x3727c5ac, v1 ==> v_sqrt_f32_e32 v1, v2` fail.
5. **It is not TVM-specific.** The same minimal cases fail on the SYCL backend
   too, on the same lanes, whenever its scheduler places a VALU op immediately
   before the sqrt.

### Hypotheses that were tested and rejected

- *Cross-thread allreduce lowering.* At `thread_warp_size=64` the reduction
  lowers to a 14-instruction `ds_bpermute_b32` warp-shuffle tree; at 32 it
  lowers to shared memory with 26 `ds_read_b32`, 14 `ds_write_b32` and 5
  `s_barrier`. Two structurally unrelated implementations fail identically, and
  the earlier note claiming warp size "changes nothing" was measuring the host
  x86 module by mistake. A standalone allreduce also passes.
- *An error in the reduced value.* mean and variance are per-row scalars
  broadcast to all 64 columns, so a wrong reduction would corrupt whole rows,
  not fixed columns. Measurement confirms both are correct.
- *A shared-memory visibility bug.* The ISA has `s_waitcnt vmcnt(0) lgkmcnt(0)`
  before `s_barrier` and `s_waitcnt lgkmcnt(0)` after `ds_read2_b32`, and the
  read address is the uniform `v0 = 0`, so all lanes read what thread 0 wrote.
- *A VCC clobber in the fdiv expansion.* `v_div_fmas_f32` reads VCC implicitly,
  but `v_div_scale_f32` writes it 7 instructions earlier and nothing overwrites
  it in between. Plain division is also correct.
- *Store predication.* The sentinel pre-fill shows every lane is written.
- *DLight scheduling.* The failure survives every rule set tried.
- *Forcing x to all zeros.* Still 16 NaN lanes, which no dataflow through a
  reduction of zeros can produce. This is what redirected the search away from
  the reduction.

### Why the models pass on the SYCL backend

Not luck: this is a deliberate mitigation in the SYCL code generator.
`CodeGenSYCL::VisitExpr_(const DivNode*)` pattern-matches float32
`a / sqrt(b)` and emits `a * sycl::rsqrt(b)` instead:

```cpp
if (func->value == "sqrt" || func->value == "sqrtf") {
  os << '(';
  PrintExpr(op->a, os);
  os << " * sycl::rsqrt(";
```

That reaches `v_rsq_f32` and never issues the affected `v_sqrt_f32`, so
LayerNorm is correct on the SYCL path. TVM's LLVM/AMDGPU path has no equivalent
rewrite -- `rsqrt` appears nowhere under `src/target/llvm/`, only as the generic
TIR op registration -- so it expands `rsqrt` into `1.0 / sqrt(x)`, emits
`v_sqrt_f32`, and fails.

This is consistent with the minimal tests above, where a *bare* `te.sqrt` fails
on SYCL too: the rewrite only fires on the `a / sqrt(b)` pattern, which is what
LayerNorm produces but not what those tests wrote.

It also reproduces, from the ROCm side and at kernel granularity, the earlier
256-lane SYCL diagnostic in
`results/dcu_validation_20260819/dcu_broadcast_math_codegenfix_device0_20260819.json`
(`broadcast_normalization/divide_sqrt`): that diagnostic found broadcast
`a/sqrt(b)` nonfinite in 64 of 256 lanes, and this one finds 16 of 64, the same
one-quarter-of-each-16-lane-group signature.

### Consequence for the evaluation

The defect is in the device and its toolchain, not in either compiler being
compared, so the Transformer and ViT rows are not a TVM-HIP correctness result.
The honest statement is that those two models have no usable TVM-HIP baseline on
this DCU and that the 15 passing cases carry the comparison.

The one claim this does support is about the code generator: handling `a/sqrt(b)`
as `a*rsqrt(b)` is not a micro-optimization on this hardware but the difference
between correct and nonfinite output, and TVM's own ROCm backend demonstrates
what happens without it.

## Files

- `results/compact_{rocm,sycl}.{json,csv}` - MLP, CNN, Transformer, ViT.
- `results/resnet_{rocm,sycl}.{json,csv}` - ResNet-18/34/50 at 224x224.
- `scripts/dcu_build_rocm.sh` - isolated ROCm-enabled TVM build.
- `scripts/dcu_sweep_paired.sh` - compact sweep, both backends.
- `scripts/dcu_sweep_resnet.sh` - ResNet sweep, both backends.
- `scripts/summarize_dcu_paired.py` - pairing check and ratio table.
- `scripts/dcu_bisect_nonfinite.py` - VM instrumentation that localizes the
  first kernel producing a nonfinite output.
- `scripts/dcu_store_sentinel.py` - sentinel pre-fill separating a miscomputed
  value from a lane that was never stored.
- `scripts/dcu_isa_walk.py` - walks the module tree and dumps the AMDGPU ISA and
  LLVM IR of one device kernel.
- `scripts/dcu_allreduce_minimal.py` - standalone cross-thread allreduce, passes.
- `scripts/dcu_layernorm_stages.py` - exports the reduced sum, sum of squares
  and reciprocal-sqrt as per-lane buffers; shows the reduction is correct and
  the reciprocal-sqrt is not.
- `scripts/dcu_fdiv_matrix.py` - primitive matrix separating sqrt and log from
  fdiv, fmul and exp on one operand.
- `scripts/dcu_sqrt_mechanism.py` - five operand routes to the same value,
  isolating the VALU-producer hazard, with the ISA that shows it.
