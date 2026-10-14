#!/usr/bin/env python3
"""Copy a MetaSchedule JSON database, dropping tuning records whose trace uses async-copy software
pipelining (`software_pipeline_async_stages`, sm_80+ CUDA only). ApplyDatabase then picks, per
workload, the best remaining record: the best CUDA schedule that the SYCL backend can lower.

usage: filter_db_no_async.py <src_db_dir> <dst_db_dir>
"""
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "database_workload.json", dst / "database_workload.json")
kept, dropped = 0, 0
best_all, best_kept = {}, {}
with open(src / "database_tuning_record.json") as fin, open(dst / "database_tuning_record.json", "w") as fout:
    for line in fin:
        rec = json.loads(line)
        wl, (trace, run_secs, *_rest) = rec[0], rec[1]
        finite = run_secs and all(isinstance(v, (int, float)) and math.isfinite(v) and v < 1e9 for v in run_secs)
        cost = sum(run_secs) / len(run_secs) if finite else math.inf
        best_all[wl] = min(best_all.get(wl, math.inf), cost)
        if "software_pipeline_async_stages" in json.dumps(trace):
            dropped += 1
            continue
        kept += 1
        best_kept[wl] = min(best_kept.get(wl, math.inf), cost)
        fout.write(line)
changed = [wl for wl in best_all if best_all[wl] < math.inf and best_kept.get(wl, math.inf) != best_all[wl]]
summary = {"kept": kept, "dropped_async": dropped,
           "workloads_whose_best_changed": len(changed),
           "workloads_without_finite_record_after_filter": sum(1 for wl in best_all if best_all[wl] < math.inf and best_kept.get(wl, math.inf) == math.inf),
           "best_ms_before_after": {str(wl): [best_all[wl] * 1e3, best_kept.get(wl, math.inf) * 1e3] for wl in changed}}
(dst / "filter_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
