#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.pretrain.glt_distill_engine import run_stage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("teacher", "student"), required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--result-root")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--resume")
    args = parser.parse_args()
    run_stage(
        args.config, args.stage,
        stop_after=args.stop_after, result_root=args.result_root,
        teacher_checkpoint=args.teacher_checkpoint,
        resume=args.resume,
    )
