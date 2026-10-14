# Experiments of v2.0

Run, build and analysis scripts for the v2.0 backend (`source_overlay/`, change table in
`source_overlay/README.md`). Their outputs, as collected on the evaluation hosts, are in
`results/round2/`. In the scripts and logs, "published" refers to v1 of this repository
(commit `472a43e`), and "r2" or "overlay_r2" to the v2.0 overlay.

The scripts keep the absolute paths of the evaluation hosts: on the RTX 4090 host, this repository
was checked out under `.../round2_20261012/repo/`, with `revision_round2/experiments` and
`revision_round2/code/overlay_r2` corresponding to `experiments/round2` and `source_overlay`; on the
DCU host, the scripts run inside a container with the overlay built in `/root/r2/tvm`. Adjust the
paths at the top of each script to rerun them elsewhere.

## Measurement protocol (RTX 4090)

Timings are taken in timing windows with exclusive use of device 1; each window script aborts if
another process uses that card. Functional checks ran on device 0 and are not used for timing. Each configuration runs in three processes with 10 warm-ups and 50
synchronized measurements; summaries report the median of the three per-process p50 values.

## Environment and builds

| Script | Purpose |
|---|---|
| `env_4090.sh` | Environment for the published build, the v2.0 build (`r2`) and the v2.0 build against DPC++ v7.1.1 (`v711`) |
| `build_r2_4090.sh`, `sync_r2_4090.sh` | Build the overlay into a separate TVM tree; incremental rebuild after overlay changes |
| `build_dpcpp_v711.sh`, `build_tvm_v711.sh` | Build the intel/llvm v7.1.1 release with the CUDA plugin from source, and TVM against it |
| `run_round2_4090.sh`, `chain_4090.sh` | Early staged runner (preflight, build, stages) and window chaining |

## Timing windows (RTX 4090) and their results

| Script | Content | Results |
|---|---|---|
| `run_window1.sh`, `run_window1b.sh` | Asynchronous dispatch and command graphs; then C8, C9, C10 with CUDA's schedules on SYCL | `results/round2/4090_window1`, `4090_window1_e1`, `4090_windows_1_3/window1b` |
| `run_window2.sh` | Launch-overhead microbenchmark, `discard_events`, cross-replay of the v1 databases | `4090_windows_1_3/window2` |
| `run_window3.sh` | D1; equal-budget seeded retunes of ResNet-18 B1 for both targets with the T1 runner and T2 timer (the databases used by all later tuned results) | `4090_windows_1_3/window3` |
| `run_window4.sh` | Seven-model batch sweep, E11 compute-bound shapes, TVM-OpenCL comparison | `4090_windows_4_5/window4` |
| `run_window5.sh` | Reduction lowerings: barriers, sub-group shuffles, `reduce_over_group` | `4090_windows_4_5/window5` |
| `run_window6.sh`, `6d`, `6e`, `6g`, `6h` | After M1, J1, F1: batch sweep, ablation, index-width variants (N1), graph-replay cost (6g), long runs and graph sweep with R2 (6h) | `4090_window6`, `4090_windows_6` |
| `run_window7.sh` | Retune of ResNet-18 with N1 | `4090_window7` |
| `run_window8.sh` | Final backend (after F2): single-switch rows, seven-model sweep, retuned ResNet-18, traces | `4090_window8` |
| `run_window9.sh` | The same sources built against DPC++ 646ab08 and DPC++ v7.1.1, alternating; launch and graph-replay microbenchmarks; L1 on/off | `4090_window9` |
| `run_window10.sh` | Retune of ResNet-18 with DPC++ v7.1.1; traces of both builds | `4090_window10` |

Each window directory holds the per-process JSON records (`<config>_run<N>.json`), the window log,
and where present Nsight Systems CSV exports (`e1/`) and a `README.md` with the summarized numbers.

## Other experiments

| Script | Purpose | Results |
|---|---|---|
| `run_e10_4090.sh`, `run_e10_final2.sh`, `run_e10_dcu.sh`, `tvm_sycl_extended_probe_r2.py` | Operator correctness against float64 references (masked relative error, ULP) | `4090_e10*`, `dcu_e10_r2` |
| `run_dcu_r2.sh`, `run_dcu_r2b.sh`, `run_dcu_d1.sh`, `run_dcu_final.sh`, `dcu_chain.sh` | DCU (gfx906, DPC++ 2022 HIP plugin) sweeps of the seven models against the v1 library and TVM's HIP backend | `dcu_20261012`, `dcu_r2_20261013`, `dcu_final_20261013` |
| `run_jit_compile_test.sh`, `jit_split_bench.sh`, `compile_profile.py` | JIT compile time and the JIT cache (B1, B2, B2') | `4090_aux/jit` |
| `sycl_launch_overhead.cpp`, `cuda_launch_overhead.cu`, `hip_launch_overhead.cpp` | Host cost of a kernel launch (queue::submit, queue shortcut, `khr::launch_grouped`) | windows 2 and 9 |
| `sycl_graph_replay_overhead.cpp`, `cuda_graph_replay_overhead.cu` | Host cost of a graph replay against the number of earlier replays | windows 6g and 9 |
| `sycl_grid_limit.cpp`, `sycl_rsqrt_ulp.cpp`, `onednn_large_shapes.cpp`, `sycl_graph_probe.cpp` | Grid limits, rsqrt accuracy, oneDNN shapes, graph extension probe | `4090_aux/e8e9`, `4090_aux/e11`, window 4 |
| `kcmp.sh`, `tcmp_toolchains.sh`, `dump_*_src.py` | Machine-code comparison of one kernel or module (registers, spills, SASS histogram) across code generators or DPC++ versions | `4090_aux/kcmp`, `4090_aux/tcmp_*`, window 9 and 10 READMEs |
| `run_audit_4090.sh`, `audit_warp_sync.py`, `diag_timer.sh`, `diag_nsys.sh` | Warp-sync audit of generated sources; diagnosis of the v1 tuning timer | `4090_aux/audit`, `4090_aux/tune_diag*` |
| `run_func_checks*.sh`, `smoke_r2.py`, `run_r2_tests_nopytest.py` | Functional checks on device 0 and the regression tests | `4090_aux/func*` |

Probes: `compact_probe_r2.py` (MLP, CNN, Transformer, ViT-style), `resnet_relax_probe_r2.py`
(ResNets, MetaSchedule tuning with `--runner-recycle-on-error`, database replay, Nsight markers),
`big_ops_probe_r2.py` (compute-bound shapes).

## Analysis scripts (`analysis/`)

Summaries of the window directories: `analyze_window.py`, `summarize_variants.py`,
`summarize_window4.py`, `summarize_window9.py`, `ablation_4090.py`, `dispersion.py`; tables:
`batch_sweep_tex.py`, `dcu_table_tex.py`, `table3_rows.py`, `tabulate_dcu.py`; Nsight traces:
`analyze_e1_nsys.py`, `compare_by_order.py`, `compare_kernels.py`, `e1_compact_kernels.py`; the
19 ResNet-18 workloads of the v1 databases: `m2_workloads.py`, `m2_workload_table.py`;
`filter_db_no_async.py`, `collect_existing_metrics.py`.

`lineage_compare.py` counts the code lines (comments and blank lines removed, whitespace ignored)
of the backend that are identical to the RELOAD22/tvm `sycl` branch (commit `5b36163df4`):

```bash
# fork files from the branch, TVM files from the apache-tvm-src-v0.21.0 release
F="src/target/source/codegen_sycl.cc src/target/source/codegen_sycl.h src/runtime/sycl/sycl_common.h \
   src/runtime/sycl/sycl_module.cc src/runtime/sycl/sycl_module.h src/runtime/sycl/sycl_device_api.cc \
   cmake/modules/SYCL.cmake"
python3 lineage_compare.py <fork files root> source_overlay $F
```

For v2.0: 1,765 lines in these seven files, 937 of them identical to the branch; 278 lines in new
files; 134 changed lines in TVM v0.21 files (same script, TVM release as the old root). Of the
2,177 backend lines, 937 (43%) are identical to the branch, and 937 of the branch's 1,224 lines
remain.
