"""Line-level lineage comparison: strip comments and blank lines,
ignore whitespace, diff; count lines of the new file that are identical to the old file.

usage: compare.py <old root> <new root> <relpath>...      (prints one row per file and a total)
"""
import difflib
import os
import re
import sys


def code_lines(path):
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8", errors="replace").read()
    ext = os.path.splitext(path)[1]
    if ext in (".py", ".cmake", ".pxi", ".txt") or path.endswith("CMakeLists.txt.patch"):
        text = re.sub(r'(?m)^\s*#.*$', "", text)
        text = re.sub(r'"""[\s\S]*?"""', "", text)
    else:
        # C/C++: block comments, then line comments (not inside string literals, approximately)
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
        text = re.sub(r'(?m)(^|[^:"\\])//.*$', r"\1", text)
    out = []
    for line in text.splitlines():
        norm = " ".join(line.split())
        if norm:
            out.append(norm)
    return out


def compare(old, new):
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    same = sum(b.size for b in sm.get_matching_blocks())
    return same


def main():
    old_root, new_root, rels = sys.argv[1], sys.argv[2], sys.argv[3:]
    tot = [0, 0, 0]
    print(f"{'file':42s} {'old':>6s} {'new':>6s} {'same':>6s} {'new%same':>8s}")
    for rel in rels:
        o, n = code_lines(os.path.join(old_root, rel)), code_lines(os.path.join(new_root, rel))
        s = compare(o, n)
        tot[0] += len(o); tot[1] += len(n); tot[2] += s
        pct = 100.0 * s / len(n) if n else 0.0
        print(f"{rel:42s} {len(o):6d} {len(n):6d} {s:6d} {pct:7.1f}%")
    print(f"{'TOTAL':42s} {tot[0]:6d} {tot[1]:6d} {tot[2]:6d} {100.0 * tot[2] / tot[1]:7.1f}%")


if __name__ == "__main__":
    main()
