#!/usr/bin/env python3
"""用户明确授权的T0_A提前正式评估；复用35个best，不训练、不选路线。"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_glt_fusion_r1 as campaign

def main():
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continue-after-dispatch-fix',action='store_true')
    args=parser.parse_args()
    attempt='early_T0_A_test_02' if args.continue_after_dispatch_fix else 'early_T0_A_test_01'
    if args.continue_after_dispatch_fix and campaign.read(campaign.OUT/'early_T0_A_test_01/execution_status.json')['status']!='阻断':
        raise ValueError('authorized dispatch continuation requires stopped earlier attempt')
    campaign.LOG=campaign.BASE_LOG/attempt
    campaign.MD_EXITS=[campaign.BASE_LOG/'repair_01/prepare_pi1m/exit.txt',campaign.BASE_LOG/'repair_02/prepare_downstream/exit.txt']
    folder=campaign.OUT/attempt
    status=folder/'execution_status.json'
    if status.exists():raise FileExistsError('early evaluation already recorded; no restart')
    control=campaign.read(folder/'controller_pause.json')
    campaign.write(status,dict(status='执行中',scope='T0_A',started=campaign.now()))
    try:
        campaign.finetune('T0_A',formal=True)
        references={}
        for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
            parts=[part.strip() for part in line.split('|')]
            if len(parts)==5 and parts[1].lower() in campaign.TASKS and '**' in parts[3]:
                references[parts[1].lower()]=dict(model=parts[2],r2=float(parts[3].replace('**','').split('±')[0]))
        if set(references)!=set(campaign.TASKS):raise ValueError('incomplete paper references')
        rows=[]
        for task in campaign.TASKS:
            folds=[]
            for fold in range(5):
                unit=campaign.OUT/'formal/T0_A'/task/f'fold{fold}'
                metric=campaign.read(unit/'metrics.json')
                folds.append(dict(fold=fold,r2=metric['r2'],rmse=metric['rmse'],prediction=str(unit/'test_predictions.npz'),metrics=str(unit/'metrics.json')))
            row=dict(arm='T0_A',task=task,folds=folds,paper_model=references[task]['model'],paper_r2=references[task]['r2'])
            for key in ('r2','rmse'):
                values=[f[key] for f in folds]
                row[key+'_mean']=float(np.mean(values));row[key+'_std']=float(np.std(values,ddof=0))
            row['difference']=row['r2_mean']-row['paper_r2'];rows.append(row)
        campaign.write(folder/'seven_task_test_results.json',dict(rows=rows,protocol='paper5_outer; seed42+fold; ddof0; user-authorized early test before final engineering test; not blind test; paper protocol differs/unverified',selection_used_test=False,engineering_test='尚未执行',macro_r2=float(np.mean([r['r2_mean'] for r in rows]))))
        campaign.write(status,dict(status='已完成',completed=campaign.now(),formal_units=35,selection_used_test=False,engineering_test='尚未执行'))
        campaign.write(campaign.OUT/'consumption.json',campaign.consumption())
        expected='scripts/run_glt_fusion_r1.py'
        if expected not in Path(f'/proc/{control["pid"]}/cmdline').read_bytes().decode().replace('\x00',' '):
            raise ValueError('paused controller identity changed')
        os.kill(control['pid'],signal.SIGCONT)
        campaign.write(folder/'controller_resumed.json',dict(pid=control['pid'],time=campaign.now()))
    except BaseException as error:
        campaign.write(status,dict(status='阻断',time=campaign.now(),error=repr(error),traceback=traceback.format_exc(),no_retry=True,controller='paused; preserve evidence'))
        campaign.write(campaign.OUT/'consumption.json',campaign.consumption())
        raise

if __name__=='__main__':main()
