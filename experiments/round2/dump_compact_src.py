"""Dump the device source of a compact model built with the default (DLight) pipeline.

usage: dump_compact_src.py <model> <sycl|cuda> <target string> <out prefix>
CUDA: <out>.cu and <out>.ptx; SYCL: <out>.cpp (the translation unit given to DPC++).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "experiments"))
import relax_model_probe as base  # noqa: E402

import tvm  # noqa: E402
from tvm.relax.frontend import nn  # noqa: E402

from compact_probe_r2 import default_pipeline  # noqa: E402


def walk(m, kind, seen):
    if id(m) in seen:
        return []
    seen.add(id(m))
    found = [m] if m.type_key == kind else []
    for sub in m.imported_modules:
        found += walk(sub, kind, seen)
    return found


def main():
    model_name, kind, ts, out = sys.argv[1:5]
    target = tvm.target.Target(ts)
    model, input_shape, _ = base.make_case(model_name, 1)
    mod, _ = model.export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
    ex = tvm.compile(mod, target=target, relax_pipeline=default_pipeline(target))
    mods = walk(ex.mod, kind, set())
    if kind == "cuda":
        Path(out + ".cu").write_text("\n".join(m.get_source("cu") for m in mods))
        Path(out + ".ptx").write_text("\n".join(m.get_source("ptx") for m in mods))
    else:
        Path(out + ".cpp").write_text("\n".join(m.get_source("sycl") for m in mods))
    print(kind, "modules:", len(mods))


if __name__ == "__main__":
    main()
