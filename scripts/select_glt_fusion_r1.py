#!/usr/bin/env python3
"""完整validation选择和正式预测汇总；不调用模型，不根据test选路线。"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.initialize_glt_fusion_r1 import OUT, ROOT, TASKS, CORE

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x') as handle:json.dump(data,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
def metrics(arm,task,fold):return read(OUT/'finetune'/arm/task/f'fold{fold}'/'metrics.json')
def historical(task,fold):
    if task=='xc':return ROOT/f'results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/fold{fold}/metrics.json'
    route='m_xattn' if task=='eea' else 'glt_ref'
    return ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/{route}/{task}/fold{fold}/metrics.json'
def comparison(candidate,reference):
    folds=[]
    for task in TASKS:
        values=[]
        for fold in range(5):
            current=metrics(candidate,task,fold)['rmse']
            baseline=read(historical(task,fold))['best_validation_rmse'] if reference=='I' else metrics(reference,task,fold)['rmse']
            if not math.isfinite(baseline) or baseline<=0 or not math.isfinite(current):raise ValueError('invalid validation RMSE')
            values.append((baseline-current)/baseline)
        folds.append(values)
    means=np.mean(folds,axis=1);macro=float(means.mean());positives=int((means>0).sum());worst=float(means.min())
    return dict(reference=reference,fold_gains=dict(zip(TASKS,folds)),task_means=dict(zip(TASKS,means.tolist())),macro=macro,positive_tasks=positives,worst_task=worst,passed=macro>=.01 and positives>=5 and worst>=-.02)
def ordering(arm,reference):
    modules={'concat':0,'kfuse':1,'token':1,'tk':2}
    cfg=read(OUT/'initialization'/f'{arm}.json')
    return (-comparison(arm,reference)['macro'],modules[cfg['fusion_mode']],int(cfg['align_enabled'] or cfg['local_enabled']))
def core_selection():
    candidates={arm:comparison(arm,'B0') for arm in CORE if arm!='B0'}
    eligible=[arm for arm in candidates if candidates[arm]['passed']]
    chosen=sorted(eligible,key=lambda a:ordering(a,'B0'))[0] if eligible else 'B0'
    direct={'T0':'T_RAW','T0_A':'T0','K_MD':'K0','K_MD_A':'K_MD','TK_MD':'K_MD','TK_MD_A':'TK_MD','TK_MD_L':'TK_MD'}
    write(OUT/'selection_after_core_validation.json',dict(plan='GLT-FUSION-20261007-01/r1',validation_units=350,S=chosen,candidates=candidates,mechanisms={arm:comparison(arm,ref) for arm,ref in direct.items()},rule='macro>=1%, >=5/7 positive, min>=-2%; exact ties fewer modules then no alignment',test_used=False))
def prepare_objectives():
    import torch
    from src.modules.glt_dual_pretrain import DualPretrainer
    core=read(OUT/'selection_after_core_validation.json');stats=read(OUT/'geometry_statistics.json')['statistics']
    base=read(OUT/'initialization'/f"{core['S']}.json")
    initial=torch.load(OUT/'initialization/common_step0.pt',map_location='cpu',weights_only=False)['model']
    for arm in ('G_NORM','G_FGR'):
        config=dict(base,arm=arm,geometry_scales={name:stats[name]['std'] for name in ('length','angle')},
                    third_task='fgr' if arm=='G_FGR' else 'none',fgr_mu=stats['fgr_log1p']['mean'],fgr_sigma=stats['fgr_log1p']['std'],fgr_max_pairs=32,
                    initial_state=str(OUT/'initialization'/f'{arm}_step0.pt'),loss_weights=[1.,1.,.1 if arm=='G_FGR' else 0.])
        kwargs={key:config[key] for key in ('fusion_mode','fusion_revision','use_md200','atom_ce_input','third_task','align_enabled','local_enabled','geometry_head_norm','opt_mode','geometry_scales','fgr_mu','fgr_sigma')}
        model=DualPretrainer(**kwargs);current=model.state_dict()
        for name,value in current.items():
            if name not in initial or initial[name].shape!=value.shape:raise ValueError('objective step0 architecture mismatch '+name)
            current[name]=initial[name].clone()
        model.load_state_dict(current,strict=True)
        if Path(config['initial_state']).exists():raise FileExistsError('objective step0 already exists')
        torch.save(dict(model=current,opt_mode='GLT_FUSION_R1',step=0,arm=arm),config['initial_state'])
        write(OUT/'initialization'/f'{arm}.json',config)
def final_selection():
    core=read(OUT/'selection_after_core_validation.json');s=core['S']
    norm_s=comparison('G_NORM',s);norm_b=comparison('G_NORM','B0')
    fgr_norm=comparison('G_FGR','G_NORM');fgr_b=comparison('G_FGR','B0')
    norm_ok=norm_s['passed'] and norm_b['passed'];fgr_ok=norm_ok and fgr_norm['passed'] and fgr_b['passed']
    eligible=[s]+(['G_NORM'] if norm_ok else [])+(['G_FGR'] if fgr_ok else [])
    chosen=sorted(eligible,key=lambda a:ordering(a,'B0'))[0]
    try:
        incumbent=comparison(chosen,'I');deploy=chosen if incumbent['passed'] else 'I';decision='晋级' if deploy!='I' else '保留I'
    except (FileNotFoundError,KeyError,ValueError) as error:
        incumbent=dict(status='证据不足',error=str(error));deploy='I';decision='部署晋级不可判定；保留I'
    arms=(*CORE,'G_NORM','G_FGR');units=[]
    for arm in arms:
        for task in TASKS:
            for fold in range(5):
                folder=OUT/'finetune'/arm/task/f'fold{fold}';m=metrics(arm,task,fold)
                if m['status']!='PASS':raise ValueError('incomplete validation unit')
                units.append(dict(arm=arm,task=task,fold=fold,best=str(folder/'best.pt'),scaler=read(folder/'scaler.json'),validation_metrics=m))
    early=OUT/'selection_before_T0_A_test.json'
    prior_test_exposure=dict(arm='T0_A',units=35,authorization='用户明确要求提前正式TEST',frozen_validation_selection=str(early),used_for_selection=False) if early.exists() else None
    write(OUT/'selection_before_test.json',dict(plan='GLT-FUSION-20261007-01/r1',validation_units=420,S=s,chosen_scientific=chosen,deployment=deploy,deployment_decision=decision,incumbent=incumbent,objective_comparisons=dict(norm_s=norm_s,norm_b0=norm_b,fgr_norm=fgr_norm,fgr_b0=fgr_b),eligible=eligible,units=units,configs={arm:read(OUT/'initialization'/f'{arm}.json') for arm in arms},test_used=False,prior_test_exposure=prior_test_exposure))
def summarize():
    selection=read(OUT/'selection_before_test.json');rows=[]
    paper={}
    # Read the actual static table; do not infer from historical results.
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        parts=[part.strip() for part in line.split('|')]
        if len(parts)==5 and parts[1].lower() in TASKS and '**' in parts[3]:
            paper[parts[1].lower()]=(parts[2],float(parts[3].replace('**','').split('±')[0]))
    if set(paper)!=set(TASKS):raise ValueError('paper reference table incomplete')
    for arm in (*CORE,'G_NORM','G_FGR'):
        for task in TASKS:
            folds=[]
            for fold in range(5):
                path=OUT/'formal'/arm/task/f'fold{fold}'/'test_predictions.npz'
                with np.load(path,allow_pickle=False) as archive:
                    truth=archive['y_true'];pred=archive['y_pred'];rmse=float(np.sqrt(np.mean((truth-pred)**2)));r2=1-float(np.sum((truth-pred)**2)/np.sum((truth-truth.mean())**2))
                folds.append(dict(fold=fold,r2=r2,rmse=rmse,prediction=str(path)))
            r2s=[f['r2'] for f in folds];errors=[f['rmse'] for f in folds];model,ref=paper[task]
            rows.append(dict(arm=arm,task=task,folds=folds,r2_mean=float(np.mean(r2s)),r2_std=float(np.std(r2s,ddof=0)),rmse_mean=float(np.mean(errors)),rmse_std=float(np.std(errors,ddof=0)),paper_model=model,paper_r2=ref,difference=float(np.mean(r2s))-ref))
    review=OUT/'final_review';review.mkdir(exist_ok=True)
    write(review/'seven_task_results.json',dict(rows=rows,selection=selection,protocol='paper5_outer; seed42+fold; shared split; ddof0; not independent blind test; paper protocol equivalence unverified; no multi-seed stability claim'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('core','objectives','final','summarize'));args=p.parse_args()
    if args.action=='objectives':
        from src.training.glt_dual_runtime import require_tmux
        require_tmux()
    {'core':core_selection,'objectives':prepare_objectives,'final':final_selection,'summarize':summarize}[args.action]()
