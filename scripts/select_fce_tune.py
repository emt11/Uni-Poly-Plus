"""Validation-only selection, frozen bindings, and official paired reporting."""
import csv
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import r2_score
from scripts.run_fce_tune_unit import (OUT,BASE,SPLIT,ARMS,TASKS,PARENT,RUNTIME,
    read,write,head_path,config_for,validate_config,epochs)


def choose(rmses):
    if set(rmses)!=set(ARMS) or not all(math.isfinite(v) for v in rmses.values()):
        raise ValueError('incomplete/nonfinite validation arms')
    return min(ARMS,key=lambda a:rmses[a])


def select_folds(rows):
    if len(rows)!=140 or len({(r['arm'],r['task'],r['fold']) for r in rows})!=140:
        raise ValueError('incomplete validation matrix')
    selections=[]
    for task in TASKS:
        for fold in range(5):
            local={r['arm']:r['rmse'] for r in rows if r['task']==task and r['fold']==fold}
            selections.append(dict(task=task,fold=fold,arm=choose(local),validation_rmse=local))
    return selections


def formal_units(selections):
    if len(selections)!=35 or {(s['task'],s['fold']) for s in selections}!={(t,f) for t in TASKS for f in range(5)}:
        raise ValueError('incomplete fold selections')
    units=[]
    for s in selections:
        if s['arm'] not in ARMS:raise ValueError('unknown selected arm')
        for arm in ('T1_R0',s['arm']):
            unit=[arm,s['task'],s['fold']]
            if unit not in units:units.append(unit)
    return units


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
                    train_indices=entry['folds'][fold]['train_indices'],validation_indices=entry['folds'][fold]['validation_indices'],
                    test_indices=sorted(entry['folds'][fold]['test_indices'])))
                values.append(m['r2']);rows.append(dict(arm=arm,task=task,fold=fold,r2=m['r2'],rmse=m['rmse'],best_epoch=m['best_epoch']))
                del best
            tasks[task]=float(np.mean(values))
        scores[arm]=dict(tasks=tasks,macro_r2=float(np.mean(list(tasks.values()))))
    selections=select_folds(rows)
    interactions=[]
    for task in TASKS:
        for fold in range(5):
            local={r['arm']:r for r in rows if r['task']==task and r['fold']==fold}
            interactions.append(dict(task=task,fold=fold,**{metric:local['T1_DW'][metric]-local['T1_D'][metric]-local['T1_W'][metric]+local['T1_R0'][metric] for metric in ('r2','rmse')}))
    write(target,dict(validation_units=140,selections=selections,units=units,scores=scores,
        validation_rows=rows,runtime_revision=RUNTIME,outer_test='NOT_RUN',
        selection_rule='each task/fold own validation RMSE minimum; exact ties T1_R0,T1_D,T1_W,T1_DW',
        frozen=True,formal_units=formal_units(selections),paired_validation_combination_differences=interactions))
    with (OUT/'validation_results.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return formal_units(selections)


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
    review=OUT/'final_review';review.mkdir(exist_ok=False)
    arms={};raw={};details=[]
    for label in ('T1_R0','T1_TUNED'):
        tasks={}
        for task in TASKS:
            rows=[]
            for fold in range(5):
                chosen=next(s for s in frozen['selections'] if s['task']==task and s['fold']==fold)
                arm='T1_R0' if label=='T1_R0' else chosen['arm']
                folder=OUT/'formal'/arm/task/f'fold{fold}'
                m=read(folder/'metrics.json');pm,p=prediction_metrics(folder/'test_predictions.npz')
                binding=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold))
                if m['status']!='PASS' or m['outer_test']!='RUN' or m['config']!=binding['config'] or m['best_epoch']!=binding['best_epoch']:
                    raise ValueError('formal frozen metadata mismatch')
                if p['indices'].tolist()!=binding['test_indices']:raise ValueError('formal test indices mismatch')
                for k in ('r2','rmse'):
                    if not math.isclose(pm[k],m[k],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('formal saved metric mismatch')
                raw[label,task,fold]=p
                row=dict(fold=fold,arm=arm,best_epoch=binding['best_epoch'],**pm)
                rows.append(row);details.append(dict(flow=label,task=task,**row))
            tasks[task]=dict(r2_mean=float(np.mean([r['r2'] for r in rows])),
                r2_std=float(np.std([r['r2'] for r in rows])),rmse_mean=float(np.mean([r['rmse'] for r in rows])),
                rmse_std=float(np.std([r['rmse'] for r in rows])),folds=rows)
        arms[label]=dict(tasks=tasks,macro_r2=float(np.mean([tasks[t]['r2_mean'] for t in TASKS])))
    rows=[]
    for task in TASKS:
        for fold in range(5):
            a=raw['T1_TUNED',task,fold];b=raw['T1_R0',task,fold]
            for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
                if not np.array_equal(a[key],b[key]):raise ValueError('paired formal mismatch '+key)
        r=arms['T1_R0']['tasks'][task];c=arms['T1_TUNED']['tasks'][task]
        counts={a:sum(s['arm']==a for s in frozen['selections'] if s['task']==task) for a in ARMS}
        val=[next(x for x in frozen['validation_rows'] if (x['arm'],x['task'],x['fold'])==(s['arm'],task,s['fold'])) for s in frozen['selections'] if s['task']==task]
        rows.append(dict(task=task,reference=r,candidate=c,delta_r2=c['r2_mean']-r['r2_mean'],
            delta_rmse=c['rmse_mean']-r['rmse_mean'],paper=paper[task],candidate_minus_paper=c['r2_mean']-paper[task]['r2'],
            selection_counts=counts,selected_validation_r2=float(np.mean([x['r2'] for x in val])),
            selected_validation_rmse=float(np.mean([x['rmse'] for x in val]))))
    delta=arms['T1_TUNED']['macro_r2']-arms['T1_R0']['macro_r2']
    reached=sum(x['candidate_minus_paper']>=0 for x in rows)
    verdict='达到宏差进展里程碑' if delta>=.010 else ('小幅数值改善' if delta>0 else '本轮未观察到整体改善')
    write(review/'seven_task_results.json',dict(candidate='T1_TUNED',arms=arms,rows=rows,delta_macro_r2=delta,
        milestone=delta>=.010,threshold=.010,paper_targets_reached=reached,complete=True,independent_review='NOT_PERFORMED',
        selections=frozen['selections'],aggregation='five folds seed42+fold; std ddof=0; seven equal-weight tasks',
        protocol='shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only'))
    with (review/'formal_fold_details.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    lines=['状态：待审查；独立审查未进行。T1_TUNED 为本 fold validation 选择正则化的训练流程。',
        f"T1_R0 macro R² {arms['T1_R0']['macro_r2']:.3f}；T1_TUNED {arms['T1_TUNED']['macro_r2']:.3f}；差 {delta:+.3f}；{verdict}。七任务论文数值目标达到 {reached}/7。",
        '五折 seed42+fold，std ddof=0；原精度先算差再格式化；R²差正值较好，RMSE差负值较好。',
        '', '|任务|R0 R²±std|TUNED R²±std|ΔR²|R0 RMSE±std|TUNED RMSE±std|ΔRMSE|论文模型|论文R²|TUNED−论文|',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for x in rows:
        r=x['reference'];c=x['candidate'];p=x['paper']
        lines.append(f"|{x['task']}|{r['r2_mean']:.3f}±{r['r2_std']:.3f}|{c['r2_mean']:.3f}±{c['r2_std']:.3f}|{x['delta_r2']:+.3f}|{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}|{c['rmse_mean']:.3f}±{c['rmse_std']:.3f}|{x['delta_rmse']:+.3f}|{p['model']}|{p['r2']:.3f}|{x['candidate_minus_paper']:+.3f}|")
    lines+=['', '|任务|R0/D/W/DW选择次数|选中validation R²|TEST−validation R²|','|---|---|---:|---:|']
    for x in rows:
        lines.append(f"|{x['task']}|{'/'.join(str(x['selection_counts'][a]) for a in ARMS)}|{x['selected_validation_r2']:.3f}|{x['candidate']['r2_mean']-x['selected_validation_r2']:+.3f}|")
    lines+=['', '|任务|fold|选择|best epoch|','|---|---:|---|---:|']
    for d in details:
        if d['flow']=='T1_TUNED':lines.append(f"|{d['task']}|{d['fold']}|{d['arm']}|{d['best_epoch']}|")
    lines+=['', '共享 validation／TEST 已参与历史开发，不是独立盲测。项目 Egc／Xc 数量3380／432，论文历史记录4124／430，论文原数据与split等价性未核实；仅数值对照，不宣称公平超过论文或稳定优势。',
        'validation 组合差分保存在 selection_before_test.json；不是稳定交互机制证据。未选候选不读TEST，R0重复项复用。',
        '结果 results/fce_tune_20261015/t1/；日志 logs/fce_tune_20261015/t1/；真实退出见每个 execution.json／exit.txt；独立审查待进行。']
    delivery='\n'.join(lines)+'\n';(review/'delivery.md').write_text(delivery)
    return delivery
