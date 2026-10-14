# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Regression tests for the round-2 changes of the SYCL backend (see code/README_overlay_r2.md).

Each test pins one fix: grid mapping beyond 65,535 work-groups (C3), element-wise three-lane
vector accesses (C8), rejection of warp-level syncs (C1) and of over-budget local memory (C2),
uninitialized local memory (C9), launch bounds (C10), float min/max (F2), plugin-specific math (M1),
the opt-in 32-bit indices (N1) and the kernel submission form (L1). They need a SYCL device and DPC++ (tvm.device("sycl", 0)).
"""
import os

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tir as T


def _sycl_device():
    try:
        dev = tvm.device("sycl", 0)
        return dev if dev.exist else None
    except Exception:  # pylint: disable=broad-except
        return None


DEV = _sycl_device()
pytestmark = pytest.mark.skipif(DEV is None, reason="no SYCL device")


def _target():
    return tvm.target.Target.from_device(DEV)


def _sycl_source(rt_mod):
    stack, seen = [rt_mod], set()
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if m.type_key == "sycl":
            return m.get_source("sycl")
        stack.extend(m.imported_modules)
    raise AssertionError("no SYCL module in the build")


def _build(func, target=None):
    return tvm.tir.build(tvm.IRModule({"main": func}), target=target or _target())


def test_grid_beyond_65535_work_groups():
    """blockIdx.x maps to the backend's x axis (C3); the inherited mapping put it on the z axis,
    which NVIDIA limits to 65,535 blocks."""
    nblocks, nthreads = 1 << 20, 32

    @T.prim_func
    def main(A: T.Buffer((nblocks * nthreads,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
            for tx in T.thread_binding(nthreads, thread="threadIdx.x"):
                A[bx * nthreads + tx] = T.Cast("float32", bx)

    f = _build(main)
    a = tvm.nd.array(np.zeros(nblocks * nthreads, "float32"), DEV)
    f(a)
    expected = np.repeat(np.arange(nblocks, dtype="float32"), nthreads)
    np.testing.assert_array_equal(a.numpy(), expected)


def test_three_lane_vector_access_is_elementwise():
    """Three-lane loads and stores are emitted element by element (C8): sycl::vec<T,3> has a
    16-byte layout, so the cast form misaligned 12-byte strided accesses and touched a fourth
    element."""
    n = 64 * 64

    @T.prim_func
    def main(A: T.Buffer((n * 3,), "float32"), B: T.Buffer((n * 3,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(n // 64, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                for v in T.vectorized(3):
                    B[(bx * 64 + tx) * 3 + v] = A[(bx * 64 + tx) * 3 + v] + T.float32(1)

    f = _build(main)
    a_np = np.random.uniform(-1, 1, n * 3).astype("float32")
    a, b = tvm.nd.array(a_np, DEV), tvm.nd.array(np.zeros(n * 3, "float32"), DEV)
    f(a, b)
    np.testing.assert_allclose(b.numpy(), a_np + 1, rtol=0, atol=0)
    assert "vec<float, 3>*" not in _sycl_source(f)


def test_warp_storage_sync_is_rejected():
    """A remaining storage_sync("warp") is a compile-time error (C1), not a silent no-op."""

    @T.prim_func
    def main(A: T.Buffer((32,), "float32")):
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            A[tx] = A[tx] + T.float32(1)
            T.evaluate(T.tvm_storage_sync("warp"))

    with pytest.raises(Exception):
        _build(main)


def test_local_memory_over_budget_is_rejected():
    """Group-local allocations above the target budget (at most 48 KB) fail to build (C2)
    instead of being trimmed with a warning."""
    n = 16384  # 64 KB of float32

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
        for bx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                S = T.decl_buffer((n,), "float32", scope="shared")
                for i in range(n // 256):
                    S[i * 256 + tx] = A[i * 256 + tx]
                for i in range(n // 256):
                    B[i * 256 + tx] = S[n - 1 - (i * 256 + tx)]

    with pytest.raises(Exception):
        _build(main)


def test_local_memory_and_launch_bounds_in_source():
    """Shared buffers use group_local_memory_for_overwrite (C9) and kernels carry
    [[intel::max_work_group_size]] from their thread extents (C10)."""
    n = 1024

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(n // 64, thread="blockIdx.x"):
            for tx in T.thread_binding(64, thread="threadIdx.x"):
                S = T.decl_buffer((64,), "float32", scope="shared")
                S[tx] = A[bx * 64 + tx]
                T.evaluate(T.tvm_storage_sync("shared"))
                B[bx * 64 + tx] = S[63 - tx]

    f = _build(main)
    src = _sycl_source(f)
    assert "group_local_memory_for_overwrite" in src
    assert "[[intel::max_work_group_size(1, 1, 64)]]" in src
    a_np = np.arange(n, dtype="float32")
    a, b = tvm.nd.array(a_np, DEV), tvm.nd.array(np.zeros(n, "float32"), DEV)
    f(a, b)
    np.testing.assert_array_equal(b.numpy(), a_np.reshape(-1, 64)[:, ::-1].reshape(-1))


def test_float_min_max_use_fmin_fmax():
    """Float min/max are sycl::fmin/fmax (F2), as CUDA's fminf/fmaxf."""
    n = 4096

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(n // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                i = bx * 128 + tx
                B[i] = T.min(T.max(A[i], T.float32(0)), T.float32(0.5))

    f = _build(main)
    src = _sycl_source(f)
    assert "sycl::fmax(" in src and "sycl::fmin(" in src
    a_np = np.random.uniform(-1, 1, n).astype("float32")
    a, b = tvm.nd.array(a_np, DEV), tvm.nd.array(np.zeros(n, "float32"), DEV)
    f(a, b)
    np.testing.assert_array_equal(b.numpy(), np.clip(a_np, 0, 0.5))


def test_erf_form_follows_the_plugin():
    """Behind the CUDA plugin (arch sm_*) erf is the native float32 overload (M1); otherwise it is
    evaluated through the double overload. Both stay within 1e-6 of the float64 reference."""
    n = 1 << 16

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(n // 256, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                B[bx * 256 + tx] = T.erf(A[bx * 256 + tx])

    target = _target()
    f = _build(main, target)
    src = _sycl_source(f)
    arch = str(target.attrs.get("arch", ""))
    if arch.startswith("sm_"):
        assert "sycl::erf(" in src and "(double)" not in src
    a_np = np.linspace(-4, 4, n).astype("float32")
    a, b = tvm.nd.array(a_np, DEV), tvm.nd.array(np.zeros(n, "float32"), DEV)
    f(a, b)
    import math  # pylint: disable=import-outside-toplevel

    ref = np.array([math.erf(float(x)) for x in a_np])
    assert np.max(np.abs(b.numpy() - ref)) < 1e-6


def test_native_indices_option():
    """TVM_SYCL_NATIVE_IDS=1 (N1) reads the 32-bit PTX registers on NVPTX; results are unchanged.
    The default keeps nd_item's indices."""
    nblocks, nthreads = 300, 96

    @T.prim_func
    def main(A: T.Buffer((nblocks * nthreads,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(nblocks, thread="blockIdx.x"):
            for tx in T.thread_binding(nthreads, thread="threadIdx.x"):
                A[bx * nthreads + tx] = T.Cast("float32", (bx * 7 + tx) % 13)

    expected = np.array([(b * 7 + t) % 13 for b in range(nblocks) for t in range(nthreads)], "float32")
    for mode in ("0", "1"):
        old = os.environ.get("TVM_SYCL_NATIVE_IDS")
        os.environ["TVM_SYCL_NATIVE_IDS"] = mode
        try:
            f = _build(main)
        finally:
            if old is None:
                del os.environ["TVM_SYCL_NATIVE_IDS"]
            else:
                os.environ["TVM_SYCL_NATIVE_IDS"] = old
        src = _sycl_source(f)
        assert ("#define TVM_SYCL_PORTABLE_IDS" in src) == (mode == "0")
        a = tvm.nd.array(np.zeros(nblocks * nthreads, "float32"), DEV)
        f(a)
        np.testing.assert_array_equal(a.numpy(), expected)


def test_kernel_submission_forms():
    """Kernels are submitted through TVM_SYCL_LAUNCH (L1): handler- and event-free where the DPC++
    headers provide sycl_khr_free_function_commands, queue::submit otherwise or with
    TVM_SYCL_DIRECT_SUBMIT=0. Two dependent kernels per call, 50 calls without a host wait: the
    in-order queue alone must order them."""
    n = 4096

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32")):
        T.func_attr({"tir.noalias": True})
        for bx in T.thread_binding(n // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                A[bx * 128 + tx] = A[bx * 128 + tx] + T.float32(1)
        for bx in T.thread_binding(n // 128, thread="blockIdx.x"):
            for tx in T.thread_binding(128, thread="threadIdx.x"):
                B[bx * 128 + tx] = A[bx * 128 + tx] * T.float32(2)

    a_np = np.random.uniform(-1, 1, n).astype("float32")
    for mode in ("1", "0"):
        old = os.environ.get("TVM_SYCL_DIRECT_SUBMIT")
        os.environ["TVM_SYCL_DIRECT_SUBMIT"] = mode
        try:
            f = _build(main)
        finally:
            if old is None:
                del os.environ["TVM_SYCL_DIRECT_SUBMIT"]
            else:
                os.environ["TVM_SYCL_DIRECT_SUBMIT"] = old
        src = _sycl_source(f)
        assert src.count("TVM_SYCL_LAUNCH(Qh,") == 2
        assert ("#define TVM_SYCL_HANDLER_SUBMIT" in src) == (mode == "0")
        a, b = tvm.nd.array(a_np, DEV), tvm.nd.array(np.zeros(n, "float32"), DEV)
        for _ in range(50):
            f(a, b)
        np.testing.assert_allclose(a.numpy(), a_np + 50, rtol=1e-6)
        np.testing.assert_allclose(b.numpy(), (a_np + 50) * 2, rtol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
