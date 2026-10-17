"""Validation-only selection, frozen bindings, and official paired reporting."""
import csv
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import r2_score
from scripts.run_fce_tune_t3b_unit import (OUT,BASE,SPLIT,ARMS,TASKS,parent,RUNTIME,
    read,write,head_path,config_for,validate_config,epochs)

REFERENCE=ROOT/'results/fce_tune_20261015/t3'
COMPARE_ARMS=('T3_REF','T3B_G05')


def reference_bindings():
    saved=read(REFERENCE/'selection_before_test.json')
    units=[u for u in saved['units'] if u['arm']=='T3_REF']
    rows=[u for u in saved['validation_rows'] if u['arm']=='T3_REF']
    expected={(t,f) for t in TASKS for f in range(5)}
    if len(units)!=35 or {(u['task'],u['fold']) for u in units}!=expected or len(rows)!=35:
        raise ValueError('incomplete reused T3_REF bindings')
    if any(not (REFERENCE/'formal'/'T3_REF'/t/f'fold{f}'/'test_predictions.npz').is_file() for t,f in expected):
        raise ValueError('missing reused T3_REF formal result')
    return units,rows


def fixed_formal_units(units):
    expected={(a,t,f) for a in ARMS for t in TASKS for f in range(5)}
    keys=[(u['arm'],u['task'],u['fold']) for u in units]
    if len(keys)!=35 or len(set(keys))!=35 or set(keys)!=expected:
        raise ValueError('incomplete/duplicate fixed candidate FT matrix')
    return [[a,t,f] for a in ARMS for t in TASKS for f in range(5)]


def prediction_metrics(path):
    with np.load(path,allow_pickle=False) as f:
        p={k:f[k].copy() for k in ('indices','sample_keys','y_true','y_pred','scaler_mean','scaler_scale','best_epoch')}
    if p['y_true'].shape!=p['y_pred'].shape or not np.isfinite(p['y_true']).all() or not np.isfinite(p['y_pred']).all():
        raise ValueError('invalid saved predictions')
    return dict(r2=float(r2_score(p['y_true'],p['y_pred'])),
        rmse=float(np.sqrt(np.mean((p['y_true']-p['y_pred'])**2)))),p


def freeze():
    if not (OUT/'pt_completion.json').exists():raise ValueError('both PT packages must complete before freeze')
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
                meta=dict(config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),head_initial=str(head_path(task,fold)))
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
    reference_units,reference_rows=reference_bindings()
    from datetime import datetime,timezone
    write(target,dict(validation_units=35,units=units,reference_units=reference_units,reference_root=str(REFERENCE),reference_already_development_exposed=True,scores=scores,validation_rows=rows+reference_rows,
        runtime_revision=RUNTIME,outer_test='NOT_RUN',frozen=True,
        frozen_at=datetime.now(timezone.utc).isoformat(),formal_units=formal,pt_parents={a:dict(path=str(parent(a)),config=config_for(a,TASKS[0])['training_config']) for a in ARMS},
        source_used=str(OUT/'source_used'),test_arms=list(ARMS),
        selection_rule='own fold validation RMSE chooses best epoch within each arm; fixed candidate evaluated versus reused T3_REF; no arm selection'))
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
    for arm in COMPARE_ARMS:
        tasks={}
        for task in TASKS:
            rows=[]
            for fold in range(5):
                folder=(REFERENCE if arm=='T3_REF' else OUT)/'formal'/arm/task/f'fold{fold}'
                m=read(folder/'metrics.json');pm,p=prediction_metrics(folder/'test_predictions.npz')
                binding=next(u for u in frozen['units']+frozen['reference_units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold))
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
            a=raw['T3B_G05',task,fold];b=raw['T3_REF',task,fold]
            for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
                if not np.array_equal(a[key],b[key]):raise ValueError('paired formal mismatch '+key)
        r=arms['T3_REF']['tasks'][task];c=arms['T3B_G05']['tasks'][task]
        for b,a in zip(r['folds'],c['folds']):
            paired.append(dict(task=task,fold=a['fold'],ref_best_epoch=b['best_epoch'],g05_best_epoch=a['best_epoch'],
                ref_r2=b['r2'],g05_r2=a['r2'],delta_r2=a['r2']-b['r2'],ref_rmse=b['rmse'],g05_rmse=a['rmse'],delta_rmse=a['rmse']-b['rmse']))
        rows.append(dict(task=task,reference=r,candidate=c,delta_r2=c['r2_mean']-r['r2_mean'],
            delta_rmse=c['rmse_mean']-r['rmse_mean'],paper=paper[task],
            reference_minus_paper=r['r2_mean']-paper[task]['r2'],candidate_minus_paper=c['r2_mean']-paper[task]['r2'],
            r2_improved_folds=sum(a['r2']>b['r2'] for a,b in zip(c['folds'],r['folds'])),
            rmse_improved_folds=sum(a['rmse']<b['rmse'] for a,b in zip(c['folds'],r['folds']))))
    delta=arms['T3B_G05']['macro_r2']-arms['T3_REF']['macro_r2']
    reached={a:sum(arms[a]['tasks'][t]['r2_mean']>=paper[t]['r2'] for t in TASKS) for a in COMPARE_ARMS}
    verdict='达到宏差进展里程碑' if delta>=.010 else ('小幅数值改善' if delta>0 else '本轮未观察到整体改善')
    historical=read(BASE/'final_review/seven_task_results.json')
    historical_macro=float(np.mean([historical['arms']['F_CE'][t]['r2']['mean'] for t in TASKS]))
    t2=read(ROOT/'results/fce_tune_20261015/t2/final_review/seven_task_results.json')
    write(review/'seven_task_results.json',dict(historical_fce_macro=historical_macro,t2_macro={a:v['macro_r2'] for a,v in t2['arms'].items()},arms=arms,rows=rows,delta_macro_r2=delta,
        milestone=delta>=.010,threshold=.010,paper_targets_reached=reached,complete=True,reused_reference=str(REFERENCE),new_formal_predictions=35,independent_review='NOT_PERFORMED',
        aggregation='five folds seed42+fold; std ddof=0; seven equal-weight tasks',
        protocol='shared validation/TEST used in development; paper split equivalence unverified; numerical comparison only'))
    for name,data in [('formal_fold_details.csv',details),('paired_fold_differences.csv',paired)]:
        with (review/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['状态：待审查；独立审查未进行。T3_REF复用既有完整结果；本轮只新增T3B_G05，无臂间选择。',
        f"T3_REF macro R² {arms['T3_REF']['macro_r2']:.3f}；T3B_G05 {arms['T3B_G05']['macro_r2']:.3f}；差 {delta:+.3f}；{verdict}。",
        f"论文数值目标：T3_REF达到{reached['T3_REF']}/7，T3B_G05达到{reached['T3B_G05']}/7。",
        '五折 seed42+fold，std ddof=0；先用原精度计算差值再格式化；R²差正值较好，RMSE差负值较好。',
        '', '|任务|REF R²±std|G05 R²±std|ΔR²|REF RMSE±std|G05 RMSE±std|ΔRMSE|论文模型|论文R²|REF−论文|G05−论文|',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    labels=dict(eea='Eea',egb='Egb',egc='Egc',ei='Ei',eps='EPS',nc='Nc',xc='Xc')
    for x in rows:
        r=x['reference'];c=x['candidate'];p=x['paper']
        lines.append(f"|{labels[x['task']]}|{r['r2_mean']:.3f}±{r['r2_std']:.3f}|{c['r2_mean']:.3f}±{c['r2_std']:.3f}|{x['delta_r2']:+.3f}|{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}|{c['rmse_mean']:.3f}±{c['rmse_std']:.3f}|{x['delta_rmse']:+.3f}|{p['model']}|{p['r2']:.3f}|{x['reference_minus_paper']:+.3f}|{x['candidate_minus_paper']:+.3f}|")
    lines+=['', '|任务|R²改善折数|RMSE改善折数|REF validation R²|REF TEST−val|G05 validation R²|G05 TEST−val|','|---|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        r=x['reference'];c=x['candidate']
        lines.append(f"|{labels[x['task']]}|{x['r2_improved_folds']}/5|{x['rmse_improved_folds']}/5|{r['validation_r2_mean']:.3f}|{r['r2_mean']-r['validation_r2_mean']:+.3f}|{c['validation_r2_mean']:.3f}|{c['r2_mean']-c['validation_r2_mean']:+.3f}|")
    lines+=['', '|任务|fold|REF best epoch|G05 best epoch|ΔR²|ΔRMSE|','|---|---:|---:|---:|---:|---:|']
    for p in paired:lines.append(f"|{labels[p['task']]}|{p['fold']}|{p['ref_best_epoch']}|{p['g05_best_epoch']}|{p['delta_r2']:+.3f}|{p['delta_rmse']:+.3f}|")
    lines += ['',f"额外数值背景：历史四rank F_CE macro R² {historical_macro:.3f}；T2_R0 {t2['arms']['T2_R0']['macro_r2']:.3f}；T2_G_LR {t2['arms']['T2_G_LR']['macro_r2']:.3f}。这些结果不替代本轮匹配参考，不将rank差异归因于schedule。"]
    lines+=['', '共享validation／TEST已参与历史开发，不是独立盲测。项目Egc／Xc数量3380／432，论文历史记录4124／430；论文原数据与split等价性未核实。仅数值对照，不宣称公平超过论文、稳定优势或统计显著。',
        '本轮selection_before_test.json冻结35份候选best并绑定35份既有参考；新增正式预测35次，参考35份预测仅复用。参考TEST早已开发暴露，不称双方首次盲测。没有task／fold赢家或混合流程。',
        '结果results/fce_tune_20261015/t3b/；日志logs/fce_tune_20261015/t3b/；tmux Uni-Poly:fce_t3b_*。']
    delivery='\n'.join(lines)+'\n';(review/'delivery.md').write_text(delivery)
    return delivery
