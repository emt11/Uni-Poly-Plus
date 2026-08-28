#!/usr/bin/env python3
"""Thin command-line entry point for MTS pretraining."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.pretrain import engine as _engine

parse_arguments = _engine.parse_arguments
run_pretrain = _engine.run_pretrain


def main():
    return run_pretrain(parse_arguments())


if __name__ == "__main__":
    main()
