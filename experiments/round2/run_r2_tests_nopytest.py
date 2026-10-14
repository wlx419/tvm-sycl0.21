"""Run test_target_codegen_sycl_r2.py without pytest (the DCU container has none): pytest is
replaced by a stub with a working pytest.raises; the module-level skip marker and tvm.testing's
imports only touch attributes.

usage (DCU container, 2026-10-13: 9 passed):
  python3 run_r2_tests_nopytest.py tvm/tests/python/codegen/test_target_codegen_sycl_r2.py
"""
import importlib.util
import sys
import traceback
from unittest import mock


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        return issubclass(et, self.exc)


stub = mock.MagicMock()
stub.raises = _Raises
sys.modules.setdefault("pytest", stub)
spec = importlib.util.spec_from_file_location("t", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
print("device:", t.DEV, flush=True)
ok = fail = 0
for name in sorted(n for n in dir(t) if n.startswith("test_")):
    try:
        getattr(t, name)()
        ok += 1
        print("PASS", name, flush=True)
    except Exception:  # report every failure
        fail += 1
        print("FAIL", name, traceback.format_exc()[-1500:], flush=True)
print(f"{ok} passed, {fail} failed")
