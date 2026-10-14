"""Dump the device source of the E11 shapes (4096^3 matmul, ResNet stage-1 convolution) built with the
default (DLight) pipeline, as big_ops_probe_r2.py builds them.

usage: dump_big_ops_src.py <matmul|conv2d> <sycl|cuda> <target string> <out prefix>
CUDA: <out>.cu; SYCL: <out>.cpp (the translation unit given to DPC++).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tvm  # noqa: E402

from big_ops_probe_r2 import ConvMod, MatmulMod, default_pipeline  # noqa: E402


def walk(m, kind, seen):
    if id(m) in seen:
        return []
    seen.add(id(m))
    found = [m] if m.type_key == kind else []
    for sub in m.imported_modules:
        found += walk(sub, kind, seen)
    return found


def main():
    op, kind, ts, out = sys.argv[1:5]
    target = tvm.target.Target(ts)
    mod = MatmulMod if op == "matmul" else ConvMod
    ex = tvm.compile(mod, target=target, relax_pipeline=default_pipeline(target))
    mods = walk(ex.mod, kind, set())
    if kind == "cuda":
        Path(out + ".cu").write_text("\n".join(m.get_source("cu") for m in mods))
    else:
        Path(out + ".cpp").write_text("\n".join(m.get_source("sycl") for m in mods))
    print(op, kind, "modules:", len(mods))


if __name__ == "__main__":
    main()
