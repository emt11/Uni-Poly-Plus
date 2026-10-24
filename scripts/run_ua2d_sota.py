#!/usr/bin/env python3
"""UA2D-SOTA-20261018-01 r1 execution; explicit development/formal gates."""
import argparse
import os
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.ua2d_sota_context import configure
bootstrap=argparse.ArgumentParser(add_help=False)
bootstrap.add_argument('--campaign',default=os.environ.get('UA2D_SOTA_CAMPAIGN','r1'))
bootstrap.add_argument('--cycle-root');bootstrap.add_argument('--log-root')
bound,_=bootstrap.parse_known_args()
configure(bound.campaign,bound.cycle_root,bound.log_root)
from src.training.ua2d_sota import *

def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',default=bound.campaign,choices=['r1','r1_retest1'])
    parser.add_argument('--cycle-root',default=str(OUT));parser.add_argument('--log-root',default=str(LOG))
    parser.add_argument('--plan',required=True,choices=['UA2D-SOTA-20261018-01'])
    parser.add_argument('--revision',required=True,choices=['r1'])
    parser.add_argument('--stage',required=True,choices=['development','formal'])
    parser.add_argument('--unit',choices=['F0','F0_recovery1','F1','ft','pt','neural_formal','kernel_formal'])
    parser.add_argument('--round');parser.add_argument('--strategy',choices=STRATEGIES)
    parser.add_argument('--arm',default='UA');parser.add_argument('--task',choices=TASKS);parser.add_argument('--fold',type=int,choices=range(5))
    args=parser.parse_args()
    if configure(args.campaign,args.cycle_root,args.log_root)!=(OUT,LOG) or args.campaign!=CAMPAIGN: raise ValueError('late campaign change forbidden')
    try:
        if args.stage=='development' and args.unit in ('F0','F0_recovery1','F1'): kernel_round(args.unit,args.arm)
        elif args.stage=='development' and args.unit=='ft': run_ft(args.arm,args.task,args.fold)
        elif args.stage=='development' and args.unit=='pt':
            from src.training.ua2d_sota_pt import run_pt
            run_pt(args.arm)
        elif args.stage=='formal' and args.unit=='neural_formal': neural_formal(args.arm,args.task,args.fold)
        elif args.stage=='formal' and args.unit=='kernel_formal': kernel_formal(args.round,args.strategy,args.task,args.fold)
        else: raise ValueError('controller stage not yet dispatched')
    except BaseException as error:
        write(OUT/'failures'/f'{args.unit}_{args.arm}_{args.task}_{args.fold}.json',dict(status='阻断',error=repr(error),traceback=traceback.format_exc(),time=now(),retry=False))
        raise
if __name__=='__main__': main()
