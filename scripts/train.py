#!/usr/bin/env python3
"""Thin command-line entry point for one MTS fine-tune job.

The real engine result row intentionally retains the operational fields
``'eval_batch_size': int(args.eval_batch_size)`` and ``'amp_dtype': args.amp_dtype``;
this wrapper contains no prediction or checkpoint hash identity logic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.finetune import engine as _engine

parse_arguments = _engine.parse_arguments
run_finetune_job = _engine.run_finetune_job


def __getattr__(name):
    """Expose legacy inspection helpers from the single implementation."""
    return getattr(_engine, name)


def main():
    return run_finetune_job(parse_arguments())


if __name__ == "__main__":
    main()
