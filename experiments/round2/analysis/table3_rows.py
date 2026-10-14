#!/usr/bin/env python3
"""Table 3 rows (max over the RTX 4090 run and both DCU cards) from E10 JSON files."""
import json
import sys

PICK = [("matmul", "256x256x256"), ("batch_matmul", "8x32x32x64"), ("row_sum", "1024x1024"),
        ("layer_norm", "128x768"), ("conv2d_nhwc", "8x32x32x16x32x3x3"),
        ("max_pool2d_nhwc", "8x32x32x16x2x2x2"), ("gelu", "256x1024"), ("transpose_3d", "16x128x64"),
        ("softmax", "512x512")]


def rows(path):
    d = json.load(open(path))
    rs = d["results"] if isinstance(d, dict) and "results" in d else d
    return {(r["op"], r["shape"]): r for r in rs}


def main():
    runs = [rows(p) for p in sys.argv[1:]]
    for op, shape in PICK:
        rs = [r[(op, shape)] for r in runs]
        status = "pass" if all(r["status"] == "pass" for r in rs) else "FAIL"
        mabs = max(r.get("max_abs_err_vs_f64") or r.get("max_abs_err") or 0 for r in rs)
        mrel = max(r.get("max_rel_err_masked") or 0 for r in rs)
        med = max(r.get("median_ulp") or 0 for r in rs)
        excl = max(r.get("rel_masked_excluded_fraction") or 0 for r in rs)
        print(f"{op:16s} {shape:20s} {status} abs={mabs:.2e} relm={mrel:.2e} medulp={med:g} excluded={excl:.2%}")


if __name__ == "__main__":
    main()
