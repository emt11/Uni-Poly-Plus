"""Validation-only selection, frozen bindings, and official paired reporting."""
import csv
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import r2_score
from scripts.run_fce_param_unit import (OUT,BASE,SPLIT,ARMS,TASKS,PARENT,RUNTIME,
    read,write,head_path,config_for,validate_config,epochs)


def choose(macros):
    if set(macros)!=set(ARMS) or not all(math.isfinite(v) for v in macros.values()):
        raise ValueError('incomplete/nonfinite validation arms')
    return max(ARMS[1:],key=lambda a:macros[a])


def prediction_metrics(path):
    with np.load(path,allow_pickle=False) as f:
        p={k:f[k].copy() for k in ('indices','sample_keys','y_true','y_pred','scaler_mean','scaler_scale','best_epoch')}
    if p['y_true'].shape!=p['y_pred'].shape or not np.isfinite(p['y_true']).all() or not np.isfinite(p['y_pred']).all():
        raise ValueError('invalid saved predictions')
    return dict(r2=float(r2_score(p['y_true'],p['y_pred'])),
        rmse=float(np.sqrt(np.mean((p['y_true']-p['y_pred'])**2)))),p


def freeze():
    target=OUT/'selection_before_test.json'
    if target.exists():raise FileExistsError('selection already frozen')
    if any((OUT/'formal').glob('**/test_predictions.npz')):raise ValueError('TEST exists before freeze')
    units=[];scores={};rows=[]
    for arm in ARMS:
        tasks={}
        for task in TASKS:
            cfg=validate_config(read(OUT/'initialization'/arm/(task+'.json')),arm,task)
            entry=read(SPLIT/(task+'.json'))
            values=[]
            for fold in range(5):
                path=OUT/'finetune'/arm/task/f'fold{fold}'
                m=read(path/'metrics.json');s=read(path/'scaler.json');progress=read(path/'progress.json')['history']
                count=epochs(task)*math.ceil(len(entry['folds'][fold]['train_indices'])/24)
                if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(task) or m['updates']!={'attempted':count,'completed':count}:
                    raise ValueError('incomplete/mismatched FT budget')
                if len(progress)!=epochs(task) or any(r['recipe']!=cfg['recipe'] for r in progress):raise ValueError('recipe/history mismatch')
                chosen=min(progress,key=lambda r:r['validation_rmse'])
                if chosen['epoch']!=m['best_epoch'] or chosen['validation_r2']!=m['r2'] or chosen['validation_rmse']!=m['rmse']:
                    raise ValueError('validation best mismatch')
                pm,p=prediction_metrics(path/'validation_predictions.npz')
                if not np.array_equal(p['indices'],entry['folds'][fold]['validation_indices']):raise ValueError('validation indices mismatch')
                for k in ('r2','rmse'):
                    if not math.isclose(pm[k],m[k],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('saved validation metric mismatch')
                best=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
                meta=dict(config=cfg,task=task,fold=fold,seed=42+fold,parent=str(PARENT),head_initial=str(head_path(task,fold)))
                if any(best.get(k)!=v for k,v in meta.items()) or best['scaler']!=s or best['best_epoch']!=m['best_epoch'] or best['optimizer_updates']!=count:
                    raise ValueError('best metadata binding mismatch')
                if any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('nonfinite best')
                units.append(dict(arm=arm,**meta,best=str(path/'best.pt'),best_epoch=m['best_epoch'],scaler=s,
                    test_indices=sorted(entry['folds'][fold]['test_indices'])))
                values.append(m['r2']);rows.append(dict(arm=arm,task=task,fold=fold,r2=m['r2'],rmse=m['rmse'],best_epoch=m['best_epoch']))
                del best
            tasks[task]=float(np.mean(values))
        scores[arm]=dict(tasks=tasks,macro_r2=float(np.mean(list(tasks.values()))))
    candidate=choose({a:scores[a]['macro_r2'] for a in ARMS})
    changes={}
    for arm in ARMS[1:]:
        changes[arm]={}
        for task in TASKS:
            a=[r for r in rows if r['arm']==arm and r['task']==task];b=[r for r in rows if r['arm']=='R0' and r['task']==task]
            changes[arm][task]=dict(delta_r2=scores[arm]['tasks'][task]-scores['R0']['tasks'][task],
                r2_improved_folds=sum(x['r2']>y['r2'] for x,y in zip(a,b)),
                rmse_improved_folds=sum(x['rmse']<y['rmse'] for x,y in zip(a,b)))
    write(target,dict(validation_units=140,candidate=candidate,units=units,scores=scores,changes=changes,
        validation_rows=rows,runtime_revision=RUNTIME,outer_test='NOT_RUN',
        selection_rule='five-fold means then seven equal-weight task means; ties H_LR,B_LR,S_COS',
        frozen=True,test_arms=['R0',candidate]))
    with (OUT/'validation_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return candidate


def papers():
    result={}
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        p=[v.strip() for v in line.split('|')]
        if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):
            result[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
    if set(result)!=set(TASKS):raise ValueError('incomplete paper references')
    return result


def summarize():
    frozen=read(OUT/'selection_before_test.json');candidate=frozen['candidate'];paper=papers()
    review=OUT/'final_review';review.mkdir(exist_ok=False)
    arms={};raw={}
    for arm in ('R0',candidate):
        tasks={}
        for task in TASKS:
            rows=[]
            for fold in range(5):
                folder=OUT/'formal'/arm/task/f'fold{fold}'
                m=read(folder/'metrics.json');pm,p=prediction_metrics(folder/'test_predictions.npz')
                binding=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold))
                if m['status']!='PASS' or m['outer_test']!='RUN' or m['config']!=binding['config'] or m['best_epoch']!=binding['best_epoch']:
                    raise ValueError('formal frozen metadata mismatch')
                if p['indices'].tolist()!=binding['test_indices']:raise ValueError('formal test indices mismatch')
                for k in ('r2','rmse'):
                    if not math.isclose(pm[k],m[k],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('formal saved metric mismatch')
                raw[arm,task,fold]=p;rows.append(dict(fold=fold,**pm))
            tasks[task]=dict(r2_mean=float(np.mean([r['r2'] for r in rows])),
                r2_std=float(np.std([r['r2'] for r in rows])),rmse_mean=float(np.mean([r['rmse'] for r in rows])),
                rmse_std=float(np.std([r['rmse'] for r in rows])),folds=rows)
        arms[arm]=dict(tasks=tasks,macro_r2=float(np.mean([tasks[t]['r2_mean'] for t in TASKS])))
    rows=[]
    for task in TASKS:
        for fold in range(5):
            a=raw[candidate,task,fold];b=raw['R0',task,fold]
            for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
                if not np.array_equal(a[key],b[key]):raise ValueError('paired formal mismatch '+key)
        r=arms['R0']['tasks'][task];c=arms[candidate]['tasks'][task]
        rows.append(dict(task=task,reference=r,candidate=c,delta_r2=c['r2_mean']-r['r2_mean'],
            delta_rmse=c['rmse_mean']-r['rmse_mean'],paper=paper[task],candidate_minus_paper=c['r2_mean']-paper[task]['r2']))
    delta=arms[candidate]['macro_r2']-arms['R0']['macro_r2']
    historical=read(BASE/'final_review/seven_task_results.json')
    historical_macro=float(np.mean([historical['arms']['F_CE'][t]['r2']['mean'] for t in TASKS]))
    write(review/'seven_task_results.json',dict(candidate=candidate,arms=arms,rows=rows,delta_macro_r2=delta,
        effective=delta>=0.010,threshold=0.010,complete=True,independent_review='NOT_PERFORMED',
        aggregation='mean of five folds per task; mean of seven tasks; std ddof=0',
        historical_reference=historical,historical_fce_macro_r2=historical_macro,
        candidate_minus_historical_fce=arms[candidate]['macro_r2']-historical_macro,
        r0_minus_historical_fce=arms['R0']['macro_r2']-historical_macro,protocol='shared TEST used in development; paper split equivalence unverified; numerical comparison only'))
    lines=[f'候选：{candidate}；状态：待审查（独立审查未进行）。',
        f"R0 macro R² {arms['R0']['macro_r2']:.3f}；候选 {arms[candidate]['macro_r2']:.3f}；差值 {delta:+.3f}；"+('达到唯一有效门槛。' if delta>=0.010 else '本轮未证实有效。'),
        '原精度值与判定保存seven_task_results.json；先五折均值、后七任务等权均值，std ddof=0，seed42+fold。',
        '', '|任务|R0 R²±std|候选R²±std|候选−R0|论文模型|论文R²|候选−论文|',
        '|---|---:|---:|---:|---|---:|---:|']
    for x in rows:
        r=x['reference'];c=x['candidate'];p=x['paper']
        lines.append(f"|{x['task']}|{r['r2_mean']:.3f}±{r['r2_std']:.3f}|{c['r2_mean']:.3f}±{c['r2_std']:.3f}|{x['delta_r2']:+.3f}|{p['model']}|{p['r2']:.3f}|{x['candidate_minus_paper']:+.3f}|")
    lines+=['', f"历史F_CE宏R² {historical_macro:.3f}；候选−历史 {arms[candidate]['macro_r2']-historical_macro:+.3f}；R0−历史 {arms['R0']['macro_r2']-historical_macro:+.3f}（仅额外数值参照，不用于有效门槛）。", '', '共享TEST已用于历史开发，不是独立盲测；Egc／Xc数据规模与论文记录不同，精确split等价性未核实，仅数值对照，不宣称公平比较、多seed稳定或统计显著。',
        'checkpoint：finetune/<arm>/<task>/foldN/best.pt；预测：formal/<arm>/<task>/foldN/test_predictions.npz；冻结selection_before_test.json；日志logs/fce_param_20261014/r2/。']
    (review/'delivery.md').write_text('\n'.join(lines)+'\n')
    return '\n'.join(lines)
