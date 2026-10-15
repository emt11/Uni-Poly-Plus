#!/usr/bin/env python3
"""One authorized FT generalization unit, reusing the original B0 recipe."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    import scripts.run_glt_fusion_unit as unit
    old=unit.OUT
    unit.OUT=ROOT/'results/glt_ft_generalization_20261008/r1'
    unit.parent=lambda arm: old/'pretrain/B0/deploy_05000.pt'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',required=True,choices=('FT_PARTIAL','FT_LOW_LR','FT_REG'))
    p.add_argument('--task',required=True,choices=unit.TASKS)
    p.add_argument('--fold',required=True,type=int,choices=range(5))
    p.add_argument('--formal',action='store_true')
    args=p.parse_args()
    unit.ft(args.arm,args.task,args.fold,args.formal,ft_candidate=args.arm,reference_root=old)
