#!/usr/bin/env python3
"""Freeze validation-only FT choice; summarize existing formal predictions."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OLD=ROOT/'results/glt_fusion_20261007/r1'
OUT=ROOT/'results/glt_ft_generalization_20261008/r1'
ARMS=('FT_PARTIAL','FT_LOW_LR','FT_REG')
TASKS=('eea','egb','egc','ei','eps','nc','xc')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def metric(arm,task,fold):return read((OLD if arm=='B0' else OUT)/'finetune'/arm/task/f'fold{fold}/metrics.json')
def compare(arm,reference):
    from scripts.select_glt_fusion_r1 import historical
    pairs={}
    for task in TASKS:
        values=[]
        for fold in range(5):
            ref=read(historical(task,fold))['best_validation_rmse'] if reference=='I' else metric('B0',task,fold)['rmse']
            value=metric(arm,task,fold)['rmse']
            if not math.isfinite(ref) or ref<=0 or not math.isfinite(value):raise ValueError('invalid comparison RMSE')
            values.append((ref-value)/ref)
        pairs[task]=values
    means={task:float(np.mean(v)) for task,v in pairs.items()}
    macro=float(np.mean(list(means.values())));positive=sum(v>0 for v in means.values());worst=min(means.values())
    return dict(reference=reference,fold_gains=pairs,task_means=means,macro=macro,positive_tasks=positive,worst_task=worst,passed=macro>=.01 and positive>=5 and worst>=-.02)
def freeze():
    comparisons={arm:compare(arm,'B0') for arm in ARMS}
    qualified=[arm for arm in ARMS if comparisons[arm]['passed']]
    chosen=max(qualified,key=lambda a:comparisons[a]['macro']) if qualified else 'B0'
    try:
        incumbent=compare(chosen,'I');deployment=chosen if incumbent['passed'] else 'I';decision='晋级' if deployment!='I' else '保留I'
    except (FileNotFoundError,KeyError,ValueError) as error:
        incumbent=dict(status='证据不足',error=str(error));deployment='I';decision='部署晋级不可判定；保留I'
    units=[]
    for arm in ARMS:
        for task in TASKS:
            for fold in range(5):
                folder=OUT/'finetune'/arm/task/f'fold{fold}';m=metric(arm,task,fold)
                if m['status']!='PASS' or not (folder/'best.pt').exists():raise ValueError('incomplete FT validation')
                units.append(dict(arm=arm,task=task,fold=fold,best=str(folder/'best.pt'),scaler=read(folder/'scaler.json'),validation_metrics=m))
    write(OUT/'selection_before_test.json',dict(plan='GLT-FT-GENERALIZATION-20261008-01/r1',validation_units=105,reference='B0',
        reference_selection=str(OLD/'selection_before_test.json'),chosen_scientific=chosen,deployment=deployment,deployment_decision=decision,
        comparisons=comparisons,incumbent=incumbent,qualified=qualified,units=units,configs={arm:read(OUT/'configs'/f'{arm}.json') for arm in ARMS},
        rule='mean per-fold relative original-unit validation RMSE; macro>=1%, positive>=5/7, worst>=-2%; exact tie fixed PARTIAL/LOW_LR/REG order',
        test_used=False,old_test_exposure=True,post_hoc_PT_not_used_for_selection=True))
def summarize():
    paper={}
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        parts=[s.strip() for s in line.split('|')]
        if len(parts)==5 and parts[1].lower() in TASKS and '**' in parts[3]:paper[parts[1].lower()]=(parts[2],float(parts[3].replace('**','').split('±')[0]))
    if set(paper)!=set(TASKS):raise ValueError('incomplete paper references')
    rows=[]
    for arm in ARMS:
        for task in TASKS:
            folds=[]
            for fold in range(5):
                p=OUT/'formal'/arm/task/f'fold{fold}/test_predictions.npz'
                with np.load(p,allow_pickle=False) as d:truth=d['y_true'];pred=d['y_pred']
                if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise FloatingPointError('nonfinite formal predictions')
                folds.append(dict(fold=fold,rmse=float(np.sqrt(np.mean((truth-pred)**2))),r2=1-float(np.sum((truth-pred)**2)/np.sum((truth-truth.mean())**2)),prediction=str(p)))
            r2=np.array([x['r2'] for x in folds]);rmse=np.array([x['rmse'] for x in folds]);model,ref=paper[task]
            rows.append(dict(arm=arm,task=task,folds=folds,r2_mean=float(r2.mean()),r2_std=float(r2.std(ddof=0)),rmse_mean=float(rmse.mean()),rmse_std=float(rmse.std(ddof=0)),paper_model=model,paper_r2=ref,difference=float(r2.mean())-ref))
    reference=[r for r in read(OLD/'final_review/seven_task_results.json')['rows'] if r['arm']=='B0']
    folder=OUT/'final_review';folder.mkdir(exist_ok=True)
    write(folder/'seven_task_results.json',dict(rows=rows,reference_B0=reference,selection=read(OUT/'selection_before_test.json'),
        protocol='five shared folds; seed42+fold; ddof0; old test exposed during development; controlled project comparison only; paper Egc/Xc cohort and Xc scaling differ; not blind/SOTA/multiseed'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('freeze','summarize'));a=p.parse_args()
    {'freeze':freeze,'summarize':summarize}[a.action]()
