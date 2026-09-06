#!/usr/bin/env python3
"""Summarize the paired DCU SYCL/ROCm sweep.

Reads the four probe JSONs produced by tmp/dcu_sweep_paired.sh and reports, per
model and batch, the SYCL and ROCm p50 plus the SYCL/ROCm ratio. Pairing is
verified on model, batch, input shape, and tolerance; a mismatch is reported
rather than silently ratioed.
"""
import argparse
import json
import pathlib
import sys

MODEL_ORDER = ["resnet18", "resnet34", "resnet50", "mlp", "cnn", "transformer", "vit"]
BATCHES = [1, 8, 16]


def load(path: pathlib.Path):
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return {}
    rows = json.load(path.open(encoding="utf-8")).get("results", [])
    return {(r.get("model"), int(r.get("batch"))): r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tmp/dcu_paired")
    args = ap.parse_args()
    root = pathlib.Path(args.root)

    sycl = {}
    rocm = {}
    for name, sink in (("compact_sycl", sycl), ("resnet_sycl", sycl),
                       ("compact_rocm", rocm), ("resnet_rocm", rocm)):
        sink.update(load(root / f"{name}.json"))

    print(f"{'model':<12}{'B':>4}{'ROCm p50':>12}{'SYCL p50':>12}"
          f"{'SYCL/ROCm':>12}{'ROCm':>16}{'SYCL':>16}")
    print("-" * 84)

    pairs = mismatches = 0
    both_pass = 0
    for model in MODEL_ORDER:
        for batch in BATCHES:
            r = rocm.get((model, batch))
            s = sycl.get((model, batch))
            if r is None and s is None:
                continue
            pairs += 1

            def fmt(rec):
                if rec is None:
                    return "absent", None
                st = rec.get("status")
                p50 = rec.get("latency_p50_ms")
                return st, p50

            r_st, r_p50 = fmt(r)
            s_st, s_p50 = fmt(s)

            # Pairing check: same shape and tolerance on both sides.
            if r is not None and s is not None:
                for key in ("input_shape", "atol", "rtol"):
                    if r.get(key) != s.get(key):
                        mismatches += 1
                        print(f"  !! {model} B{batch}: {key} differs "
                              f"({r.get(key)} vs {s.get(key)})")

            ratio = ""
            if r_st == "pass" and s_st == "pass" and r_p50 and s_p50:
                ratio = f"{s_p50 / r_p50:.2f}x"
                both_pass += 1

            print(f"{model:<12}{batch:>4}"
                  f"{(f'{r_p50:.4f}' if r_p50 else 'NA'):>12}"
                  f"{(f'{s_p50:.4f}' if s_p50 else 'NA'):>12}"
                  f"{ratio:>12}{r_st:>16}{s_st:>16}")

    print("-" * 84)
    print(f"cases: {pairs}   both-pass pairs: {both_pass}   pairing mismatches: {mismatches}")

    # Errors worth reporting explicitly.
    print("\n=== non-pass cases ===")
    for label, table in (("rocm", rocm), ("sycl", sycl)):
        for (model, batch), rec in sorted(table.items()):
            if rec.get("status") != "pass":
                print(f"  {label:<5} {model:<12} B{batch:<3} {rec.get('status')}: "
                      f"abs={rec.get('max_abs_err')} err={rec.get('error')}")


if __name__ == "__main__":
    main()
