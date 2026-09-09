#!/usr/bin/env python3
"""Run the fixed 30-unit C0/C1/C2 frozen Ridge diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.c0_transfer import run_probe_group_task
from src.training.finetune.scheduler import ScheduledUnit, run_subprocess_scheduler

CHECKPOINTS = {
    group: ROOT / f"results/mts_glt_distill_repair_control/{group}/student/student_deploy_020k.pt"
    for group in ("c0", "c1", "c2")
}


def valid_task(root, group, task):
    complete = root / "units" / group / task / "complete.json"
    if not complete.is_file():
        return False
    try:
        for fold in range(5):
            unit = root / "units" / group / task / f"fold_{fold}"
            record = json.loads((unit / "metrics.json").read_text())
            if record.get("group") != group or record.get("task") != task or int(record.get("fold", -1)) != fold:
                return False
            import numpy as np
            with np.load(unit / "predictions.npz", allow_pickle=False) as payload:
                if not np.isfinite(payload["y_pred"]).all():
                    return False
        return True
    except Exception:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["worker", "all", "smoke"], default="all")
    parser.add_argument("--group", choices=["c0", "c1", "c2"])
    parser.add_argument("--task", choices=["xc", "eps"])
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--output-root", default="results/mts_c0_transfer_optimization/probes")
    parser.add_argument("--logs-root", default="logs/mts_c0_transfer_optimization/probes")
    args = parser.parse_args(argv)
    output = (ROOT / args.output_root).resolve()
    if args.mode in {"worker", "smoke"}:
        group, task = args.group or "c0", args.task or "xc"
        folds = [0] if args.mode == "smoke" else range(5)
        run_probe_group_task(group, task, CHECKPOINTS[group], output, folds=folds)
        return 0
    units = [ScheduledUnit(seed=index, task=task, fold=0) for index, group in enumerate(("c0", "c1", "c2")) for task in ("xc", "eps")]
    groups = {0: "c0", 1: "c1", 2: "c2"}
    report = run_subprocess_scheduler(
        units, gpu_ids=args.gpu_ids.split(","), cwd=ROOT,
        log_dir=(ROOT / args.logs_root).resolve(),
        should_skip=lambda unit: valid_task(output, groups[unit.seed], unit.task),
        command_factory=lambda unit, gpu: [
            sys.executable, __file__, "--mode", "worker", "--group", groups[unit.seed],
            "--task", unit.task, "--output-root", str(output),
        ],
    )
    report.update({"schema": "mts-c0-transfer-probe-scheduler-v1", "probe_units": 30})
    Path(args.logs_root).mkdir(parents=True, exist_ok=True)
    (ROOT / args.logs_root / "scheduler_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
