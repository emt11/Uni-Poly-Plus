"""Validation-only selection, frozen bindings, and official paired reporting."""
import csv
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import r2_score
from scripts.run_fce_tune_t2_unit import (OUT,BASE,SPLIT,ARMS,TASKS,PARENT,RUNTIME,
    read,write,head_path,config_for,validate_config,epochs)


def fixed_formal_units(units):
    expected={(a,t,f) for a in ARMS for t in TASKS for f in range(5)}
    keys=[(u['arm'],u['task'],u['fold']) for u in units]
    if len(keys)!=70 or len(set(keys))!=70 or set(keys)!=expected:
        raise ValueError('incomplete/duplicate fixed two-arm FT matrix')
    return [[a,t,f] for a in ARMS for t in TASKS for f in range(5)]


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
                if m['config']!=cfg or m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(task) or m['updates']!={'attempted':count,'completed':count}:
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
                    train_indices=entry['folds'][fold]['train_indices'],validation_indices=entry['folds'][fold]['validation_indices'],
                    test_indices=sorted(entry['folds'][fold]['test_indices'])))
                values.append(m['r2']);rows.append(dict(arm=arm,task=task,fold=fold,r2=m['r2'],rmse=m['rmse'],best_epoch=m['best_epoch']))
                del best
            tasks[task]=float(np.mean(values))
        scores[arm]=dict(tasks=tasks,macro_r2=float(np.mean(list(tasks.values()))))
    formal=fixed_formal_units(units)
    from datetime import datetime,timezone
    write(target,dict(validation_units=70,units=units,scores=scores,validation_rows=rows,
        runtime_revision=RUNTIME,outer_test='NOT_RUN',frozen=True,
        frozen_at=datetime.now(timezone.utc).isoformat(),formal_units=formal,
        source_used=str(OUT/'source_used'),test_arms=list(ARMS),
        selection_rule='own fold validation RMSE chooses best epoch within each arm; both fixed arms evaluated; no arm selection'))
    with (OUT/'validation_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return formal


def papers():
    result={}
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        p=[v.strip() for v in line.split('|')]
        if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):
            result[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
    if set(result)!=set(TASKS):raise ValueError('incomplete paper references')
    return result


def summarize():
    frozen=read(OUT/'selection_before_test.json');paper=papers()
    if frozen['formal_units']!=fixed_formal_units(frozen['units']):raise ValueError('fixed formal matrix mismatch')
    review=OUT/'final_review';review.mkdir(exist_ok=False)
    arms={};raw={};details=[]
    for arm in ARMS:
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
                raw[arm,task,fold]=p
                val=next(r for r in frozen['validation_rows'] if (r['arm'],r['task'],r['fold'])==(arm,task,fold))
                row=dict(fold=fold,arm=arm,best_epoch=binding['best_epoch'],**pm,
                    validation_r2=val['r2'],validation_rmse=val['rmse'],
                    test_minus_validation_r2=pm['r2']-val['r2'],test_minus_validation_rmse=pm['rmse']-val['rmse'])
                rows.append(row);details.append(dict(task=task,**row))
            tasks[task]={}
            for metric in ('r2','rmse','validation_r2','validation_rmse'):
                tasks[task][metric+'_mean']=float(np.mean([r[metric] for r in rows]))
                tasks[task][metric+'_std']=float(np.std([r[metric] for r in rows]))
            tasks[task]['folds']=rows
        arms[arm]=dict(tasks=tasks,macro_r2=float(np.mean([tasks[t]['r2_mean'] for t in TASKS])))
    rows=[];paired=[]
    for task in TASKS:
        for fold in range(5):
            a=raw['T2_G_LR',task,fold];b=raw['T2_R0',task,fold]
            for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
                if not np.array_equal(a[key],b[key]):raise ValueError('paired formal mismatch '+key)
        r=arms['T2_R0']['tasks'][task];c=arms['T2_G_LR']['tasks'][task]
        for b,a in zip(r['folds'],c['folds']):
            paired.append(dict(task=task,fold=a['fold'],r0_best_epoch=b['best_epoch'],g_lr_best_epoch=a['best_epoch'],
                r0_r2=b['r2'],g_lr_r2=a['r2'],delta_r2=a['r2']-b['r2'],r0_rmse=b['rmse'],g_lr_rmse=a['rmse'],delta_rmse=a['rmse']-b['rmse']))
        rows.append(dict(task=task,reference=r,candidate=c,delta_r2=c['r2_mean']-r['r2_mean'],
            delta_rmse=c['rmse_mean']-r['rmse_mean'],paper=paper[task],
            reference_minus_paper=r['r2_mean']-paper[task]['r2'],candidate_minus_paper=c['r2_mean']-paper[task]['r2'],
            r2_improved_folds=sum(a['r2']>b['r2'] for a,b in zip(c['folds'],r['folds'])),
            rmse_improved_folds=sum(a['rmse']<b['rmse'] for a,b in zip(c['folds'],r['folds']))))
    delta=arms['T2_G_LR']['macro_r2']-arms['T2_R0']['macro_r2']
    reached={a:sum(arms[a]['tasks'][t]['r2_mean']>=paper[t]['r2'] for t in TASKS) for a in ARMS}
    verdict='达到宏差进展里程碑' if delta>=.010 else ('小幅数值改善' if delta>0 else '本轮未观察到整体改善')
    write(review/'seven_task_results.json',dict(arms=arms,rows=rows,delta_macro_r2=delta,
        milestone=delta>=.010,threshold=.010,paper_targets_reached=reached,complete=True,independent_review='NOT_PERFORMED',
        aggregation='five folds seed42+fold; std ddof=0; seven equal-weight tasks',
        protocol='shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only'))
    for name,data in [('formal_fold_details.csv',details),('paired_fold_differences.csv',paired)]:
        with (review/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['状态：待审查；独立审查未进行。两臂均为预先固定的完整训练流程，无臂间选择。',
        f"T2_R0 macro R² {arms['T2_R0']['macro_r2']:.3f}；T2_G_LR {arms['T2_G_LR']['macro_r2']:.3f}；差 {delta:+.3f}；{verdict}。",
        f"论文数值目标：T2_R0达到{reached['T2_R0']}/7，T2_G_LR达到{reached['T2_G_LR']}/7。",
        '五折 seed42+fold，std ddof=0；先用原精度计算差值再格式化；R²差正值较好，RMSE差负值较好。',
        '', '|任务|R0 R²±std|G_LR R²±std|ΔR²|R0 RMSE±std|G_LR RMSE±std|ΔRMSE|论文模型|论文R²|R0−论文|G_LR−论文|',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    labels=dict(eea='Eea',egb='Egb',egc='Egc',ei='Ei',eps='EPS',nc='Nc',xc='Xc')
    for x in rows:
        r=x['reference'];c=x['candidate'];p=x['paper']
        lines.append(f"|{labels[x['task']]}|{r['r2_mean']:.3f}±{r['r2_std']:.3f}|{c['r2_mean']:.3f}±{c['r2_std']:.3f}|{x['delta_r2']:+.3f}|{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}|{c['rmse_mean']:.3f}±{c['rmse_std']:.3f}|{x['delta_rmse']:+.3f}|{p['model']}|{p['r2']:.3f}|{x['reference_minus_paper']:+.3f}|{x['candidate_minus_paper']:+.3f}|")
    lines+=['', '|任务|R²改善折数|RMSE改善折数|R0 validation R²|R0 TEST−val|G_LR validation R²|G_LR TEST−val|','|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        r=x['reference'];c=x['candidate']
        lines.append(f"|{labels[x['task']]}|{x['r2_improved_folds']}/5|{x['rmse_improved_folds']}/5|{r['validation_r2_mean']:.3f}|{r['r2_mean']-r['validation_r2_mean']:+.3f}|{c['validation_r2_mean']:.3f}|{c['r2_mean']-c['validation_r2_mean']:+.3f}|")
    lines+=['', '|任务|fold|R0 best epoch|G_LR best epoch|ΔR²|ΔRMSE|','|---|---:|---:|---:|---:|---:|']
    for p in paired:lines.append(f"|{labels[p['task']]}|{p['fold']}|{p['r0_best_epoch']}|{p['g_lr_best_epoch']}|{p['delta_r2']:+.3f}|{p['delta_rmse']:+.3f}|")
    lines+=['', '共享validation／TEST已参与历史开发，不是独立盲测。项目Egc／Xc数量3380／432，论文历史记录4124／430；论文原数据与split等价性未核实。仅数值对照，不宣称公平超过论文、稳定优势或统计显著。',
        '冻结文件selection_before_test.json绑定70份best；两臂各35次正式预测；没有task／fold臂间赢家或混合流程。',
        '结果results/fce_tune_20261015/t2/；日志logs/fce_tune_20261015/t2/；tmux Uni-Poly:fce_t2_*。']
    delivery='\n'.join(lines)+'\n';(review/'delivery.md').write_text(delivery)
    return delivery
