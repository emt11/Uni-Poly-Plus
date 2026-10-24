#!/usr/bin/env python3
"""Finite single-arm entry: V1 attention scale 64**-0.5, physical GPU2/3 only.

Run inside tmux session Uni-Poly.  The module refuses to start whenever the
result root already exists, so a failed cycle is never silently reused.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['UA2D_PRETRAIN_FP_CAMPAIGN'] = 'attnscale_r1'


def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    from src.training.ua2d_attnscale import controller
    controller()


if __name__ == '__main__':
    main()
