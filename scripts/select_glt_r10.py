"""Freeze complete r10 FT matrices and summarize saved official predictions."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import r2_score
from scripts.run_glt_r10_unit import (OUT,BASE,SPLIT,TASKS,FT_ARMS,read,write,parent,
                                      head_path,ft_config,validate_ft)
from src.modules.glt_r10 import REFERENCE,PT_ARMS,RUNTIME


def consumption(cycle):
    folder=OUT/cycle;logs=ROOT/'logs/glt_optimization_20261011/r10'/cycle
    result=dict(pt_starts=len(list(logs.glob('pt_*/command.json'))),pt_attempted_updates=0,pt_completed_updates=0,
        pt_validation_completed=0,pt_validation_presentations=0,ft_starts=len(list(logs.glob('ft_*/command.json'))),
        ft_completed=0,ft_epochs=0,ft_attempted_updates=0,ft_completed_updates=0,
        formal_starts=len(list(logs.glob('test_*/command.json'))),formal_completed=0,
        cpu_test_starts=len(list(logs.glob('final_cpu/command.json'))),geometry_derivations=0)
    for path in (folder/'pretrain').glob('*/consumption.json'):
        value=read(path);result['pt_attempted_updates']+=value['attempted_updates'];result['pt_completed_updates']+=value['completed_updates']
    for path in (folder/'pt_validation').glob('*/*/metrics.json'):
        result['pt_validation_completed']+=1;result['pt_validation_presentations']+=read(path)['presented']
    for arm in FT_ARMS[cycle]:
        for task in TASKS:
            for fold in range(5):
                unit=folder/'finetune'/arm/task/f'fold{fold}'
                if (unit/'progress.json').exists():result['ft_epochs']+=len(read(unit/'progress.json')['history'])
                if (unit/'updates.json').exists():
                    updates=read(unit/'updates.json');result['ft_attempted_updates']+=updates['attempted'];result['ft_completed_updates']+=updates['completed']
                result['ft_completed']+=int((unit/'metrics.json').exists())
                result['formal_completed']+=int((folder/'formal'/arm/task/f'fold{fold}'/'metrics.json').exists())
    result['pt_train_presentations']=1008*result['pt_completed_updates']
    pt=len(PT_ARMS[cycle]);ft=35*len(FT_ARMS[cycle])
    limits=dict(pt_starts=pt,pt_attempted_updates=5000*pt,pt_completed_updates=5000*pt,
        pt_validation_completed=3*pt,pt_validation_presentations=3072*pt,ft_starts=ft,ft_completed=ft,
        ft_epochs=70*ft,ft_attempted_updates=1600*ft,ft_completed_updates=1600*ft,
        formal_starts=ft,formal_completed=ft,cpu_test_starts=1)
    if any(result[k]>v for k,v in limits.items()):raise RuntimeError('r10 hard budget exceeded: '+str(result))
    return result


def freeze(cycle):
    folder=OUT/cycle;target=folder/'evaluation_before_test.json'
    if target.exists():raise FileExistsError('matrix already frozen')
    c=consumption(cycle);expected=35*len(FT_ARMS[cycle])
    if (c['ft_completed'],c['ft_epochs'],c['ft_completed_updates'])!=(expected,70*expected,1600*expected):raise ValueError('incomplete FT matrix')
    configs={arm:validate_ft(read(folder/'initialization'/(arm+'.json')),cycle,arm) for arm in FT_ARMS[cycle]}
    units=[]
    for arm in FT_ARMS[cycle]:
        for task in TASKS:
            split=read(SPLIT/(task+'.json'))
            for fold in range(5):
                path=folder/'finetune'/arm/task/f'fold{fold}';m=read(path/'metrics.json');scaler=read(path/'scaler.json')
                best=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
                count=70*math.ceil(len(split['folds'][fold]['train_indices'])/24)
                expected_metadata=dict(config=configs[arm],task=task,fold=fold,seed=42+fold,
                    parent=str(parent(cycle,arm)),head_initial=str(head_path(cycle,task,fold)))
                if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=70 or m['updates']!={'attempted':count,'completed':count}:
                    raise ValueError('FT budget/status mismatch')
                if any(best.get(k)!=v for k,v in expected_metadata.items()) or best['scaler']!=scaler or best['optimizer_updates']!=count:
                    raise ValueError('best parent/head/config/scaler binding mismatch')
                if not 1<=best['best_epoch']<=70 or any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('invalid best tensors/epoch')
                units.append(dict(arm=arm,**expected_metadata,best=str(path/'best.pt'),best_epoch=best['best_epoch'],
                    scaler=scaler,test_indices=sorted(split['folds'][fold]['test_indices'])))
                del best
    write(target,dict(validation_units=expected,units=units,configs=configs,selection_by_validation=True,runtime_revision=RUNTIME,outer_test='NOT_RUN'))


def prediction_metrics(path):
    with np.load(path,allow_pickle=False) as value:
        p={k:value[k].copy() for k in ('indices','sample_keys','y_true','y_pred','scaler_mean','scaler_scale','best_epoch')}
    if p['y_true'].shape!=p['y_pred'].shape or not np.isfinite(p['y_true']).all() or not np.isfinite(p['y_pred']).all():raise ValueError('corrupt saved prediction')
    return dict(r2=float(r2_score(p['y_true'],p['y_pred'])),rmse=float(np.sqrt(np.mean((p['y_true']-p['y_pred'])**2)))),p


def paired_metrics(candidate,reference):
    a,pa=prediction_metrics(candidate);b,pb=prediction_metrics(reference)
    for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
        if not np.array_equal(pa[key],pb[key]):raise ValueError('paired prediction identity/labels/scaler mismatch '+key)
    return dict(r2=a['r2']-b['r2'],rmse=a['rmse']-b['rmse'])


def papers():
    result={}
    for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
        p=[v.strip() for v in line.split('|')]
        if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):
            result[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
    if set(result)!=set(TASKS):raise ValueError('incomplete seven-task paper references')
    return result


def summarize(cycle):
    folder=OUT/cycle;review=folder/'final_review';review.mkdir(exist_ok=False)
    frozen=read(folder/'evaluation_before_test.json');paper=papers();rows=[];arms={}
    historical=read(BASE/'final_review/seven_task_results.json')['arms']['F_CE']
    historical_macro=float(np.mean([historical[t]['r2']['mean'] for t in TASKS]))
    for arm in FT_ARMS[cycle]:
        tasks={}
        for task in TASKS:
            folds=[]
            for fold in range(5):
                unit=folder/'formal'/arm/task/f'fold{fold}';m=read(unit/'metrics.json')
                metrics,p=prediction_metrics(unit/'test_predictions.npz')
                binding=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold))
                if m['status']!='PASS' or m['outer_test']!='RUN' or p['indices'].tolist()!=binding['test_indices'] or int(p['best_epoch'])!=binding['best_epoch']:
                    raise ValueError('formal frozen binding mismatch')
                if any(abs(metrics[k]-m[k])>1e-10 for k in ('r2','rmse')):raise ValueError('formal metric/prediction mismatch')
                row=dict(arm=arm,task=task,fold=fold,**metrics,best=binding['best'],predictions=str(unit/'test_predictions.npz'))
                folds.append(row);rows.append(row)
            tasks[task]={key:dict(mean=float(np.mean([r[key] for r in folds])),std=float(np.std([r[key] for r in folds],ddof=0))) for key in ('r2','rmse')}
            tasks[task].update(folds=folds,paper=dict(paper[task],delta=tasks[task]['r2']['mean']-paper[task]['r2']),historical_delta=tasks[task]['r2']['mean']-historical[task]['r2']['mean'])
        macro=float(np.mean([tasks[t]['r2']['mean'] for t in TASKS]))
        arms[arm]=dict(tasks=tasks,macro_test_r2=macro,historical_delta=macro-historical_macro,qualifies=macro-historical_macro>.01,config=frozen['configs'][arm])
    ref_cycle,ref_arm=REFERENCE[cycle];paired={}
    if ref_cycle!=cycle:
        reference=read(OUT/ref_cycle/'final_review/seven_task_results.json')['arms'][ref_arm]
        if reference['config']['runtime_revision']!=RUNTIME:raise ValueError('reference runtime mismatch')
    else:reference=arms[ref_arm]
    for arm in FT_ARMS[cycle]:
        effects={}
        for task in TASKS:
            test=[];validation=[]
            for fold in range(5):
                suffix=Path(task)/f'fold{fold}'
                test.append(paired_metrics(folder/'formal'/arm/suffix/'test_predictions.npz',OUT/ref_cycle/'formal'/ref_arm/suffix/'test_predictions.npz'))
                validation.append(paired_metrics(folder/'finetune'/arm/suffix/'validation_predictions.npz',OUT/ref_cycle/'finetune'/ref_arm/suffix/'validation_predictions.npz'))
            effects[task]=dict(test_folds=test,validation_folds=validation,
                test_mean={k:float(np.mean([r[k] for r in test])) for k in ('r2','rmse')},
                validation_mean={k:float(np.mean([r[k] for r in validation])) for k in ('r2','rmse')},
                macro_test_contribution=float(np.mean([r['r2'] for r in test]))/7,
                macro_validation_contribution=float(np.mean([r['r2'] for r in validation]))/7)
        paired[arm]=dict(reference_cycle=ref_cycle,reference_arm=ref_arm,tasks=effects,
            macro_test_delta=arms[arm]['macro_test_r2']-reference['macro_test_r2'],
            macro_validation_delta=float(np.mean([effects[t]['validation_mean']['r2'] for t in TASKS])))
    write(review/'seven_task_results.json',dict(arms=arms,paired=paired,consumption=consumption(cycle),historical_macro=historical_macro,
        protocol='seven tasks, five folds, seed42+fold, ddof0; shared TEST used in development; numeric paper comparison only',deployment='I unchanged',independent_review='NOT_RUN'))
    with (review/'fold_metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f'# {cycle} 正式结果（待独立审查）','','七任务各五折；seed42+fold，std ddof0。共享TEST已用于开发；论文协议等价性未核实，仅数值对照。',
        '','|条件|宏R²|相对历史F_CE|相对匹配参考|validation宏差|性能门槛|','|---|---:|---:|---:|---:|---|']
    for arm,v in arms.items():lines.append(f"|{arm}|{v['macro_test_r2']:.3f}|{v['historical_delta']:+.3f}|{paired[arm]['macro_test_delta']:+.3f}|{paired[arm]['macro_validation_delta']:+.3f}|{v['qualifies']}|")
    lines+=['','|任务|论文最佳模型|论文R²|条件|项目R² mean±std|原单位RMSE mean±std|项目−论文|相对匹配参考R²|','|---|---|---:|---|---:|---:|---:|---:|']
    for task in TASKS:
        for arm,v in arms.items():
            t=v['tasks'][task]
            lines.append(f"|{task}|{paper[task]['model']}|{paper[task]['r2']:.3f}|{arm}|{t['r2']['mean']:.3f}±{t['r2']['std']:.3f}|{t['rmse']['mean']:.3f}±{t['rmse']['std']:.3f}|{t['paper']['delta']:+.3f}|{paired[arm]['tasks'][task]['test_mean']['r2']:+.3f}|")
    lines+=['',f'匹配参考：{ref_cycle}/{ref_arm}。完整逐折TEST／validation配对差保留于seven_task_results.json；R²差正值较好，RMSE差负值较好。',
        '',f'产物：{folder}；日志：logs/glt_optimization_20261011/r10/{cycle}/。不按TEST追加实验，不替换F_CE或部署I。']
    (review/'formal_results.md').write_text('\n'.join(lines)+'\n')


def handoff(cycle):
    folder=OUT/cycle;log=ROOT/'logs/glt_optimization_20261011/r10'/cycle/'final_cpu'
    if int((log/'exit.txt').read_text())!=0:raise ValueError('cannot mark failed engineering suite passed')
    c=consumption(cycle);write(folder/'consumption.json',c)
    write(folder/'execution.json',dict(status='待审查',engineering='PASS',independent_review='NOT_RUN',consumption=c))
    plan=ROOT/'CURRENT_PLAN.md';text=plan.read_text();lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('|'+cycle+'|') and line.endswith('|待执行|'):
            lines[i]=line[:-len('待执行|')]+'待审查|'
    text='\n'.join(lines)+'\n'
    text+=f'\n\n## r10 {cycle} 实际执行结果（待审查）\n\n'+(folder/'final_review/formal_results.md').read_text()
    text+='\n工程最终测试：'+(log/'stdout.log').read_text().strip()+'\n\n实际预算：\n\n```json\n'+json.dumps(c,ensure_ascii=False,indent=2)+'\n```\n'
    text+='\n独立审查未执行；原始命令、环境、起止时间、stdout／stderr及真实exit在上述日志根，各阶段独立tmux window。\n'
    plan.write_text(text)


if __name__=='__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux();parser=argparse.ArgumentParser();parser.add_argument('--cycle',choices=tuple(FT_ARMS),required=True)
    parser.add_argument('--action',choices=('freeze','summarize','handoff'),required=True);args=parser.parse_args()
    globals()[args.action](args.cycle)
