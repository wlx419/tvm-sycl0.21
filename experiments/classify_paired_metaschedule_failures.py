#!/usr/bin/env python3
"""Classify current MetaSchedule measurement failures from per-task logs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TRIAL_ERROR = re.compile(
    r"\[Task #(?P<task>\d+): (?P<name>[^\]]+)\] Trial #(?P<trial>\d+): Error in (?P<stage>[^:]+):"
)
POSTPROC = re.compile(r"Postproc #\d+ \[(?P<name>[^\(]+)\([^\]]+\)\]: (?P<count>\d+) failure\(s\)")


def classify(stage: str, block: str) -> str:
    lower = block.lower()
    if "timeout" in lower:
        return "timeout"
    if "device allocation failed" in lower or "out of memory" in lower:
        return "device_allocation"
    if any(
        token in lower
        for token in ("kernel wait failed", "invalid work-group", "invalid work group", "misaligned address")
    ):
        return "device_execution"
    if stage.strip().lower() == "building" or any(
        token in lower
        for token in ("compilation error", "compile error", "fatal error:", "buildererror", "build error")
    ):
        return "build_or_compile"
    if stage.strip().lower() == "running":
        return "other_runtime"
    return "other"


def analyze_target(work_dir: Path) -> dict[str, object]:
    errors = Counter()
    by_task = defaultdict(Counter)
    postproc = Counter()
    examples = {}
    files = sorted((work_dir / "logs").glob("tvm.meta_schedule.logging.task_*.log"))
    files = [path for path in files if "task_scheduler" not in path.name]

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(TRIAL_ERROR.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[match.start() : end]
            category = classify(match.group("stage"), block)
            errors[category] += 1
            by_task[match.group("name")][category] += 1
            examples.setdefault(category, "\n".join(block.splitlines()[:8]))
        for match in POSTPROC.finditer(text):
            postproc[match.group("name").strip()] += int(match.group("count"))

    return {
        "task_log_files": len(files),
        "measurement_error_blocks": sum(errors.values()),
        "measurement_errors": dict(errors),
        "measurement_errors_by_task": {name: dict(counts) for name, counts in sorted(by_task.items())},
        "postprocessor_rejections": dict(postproc),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Experiment root containing tune/cuda and tune/sycl")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {target: analyze_target(root / "tune" / target) for target in ("cuda", "sycl")}
    out = (args.out or root / "paired_metaschedule_failure_classes.json").resolve()
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
