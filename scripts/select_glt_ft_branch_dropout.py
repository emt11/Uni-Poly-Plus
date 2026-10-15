#!/usr/bin/env python3
"""Freeze all FT bests, summarize each saved formal prediction exactly once."""
import csv,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_ft_branch_dropout_unit import OUT,BASE,PARENT,ARMS,TASKS,read,write,ft_folder,epochs
LOG=ROOT/'logs/glt_optimization_20261011/r1/S1'
REFERENCE=.7855913245367517

def macro_r2(tasks):
 if set(tasks)!=set(TASKS):raise ValueError('complete seven-task matrix required')
 values=[np.asarray(tasks[t],dtype=float) for t in TASKS]
 if any(v.shape!=(5,) or not np.isfinite(v).all() for v in values):raise ValueError('five finite folds required')
 return float(np.mean([v.mean() for v in values]))

def qualifies(macro):
 if not math.isfinite(macro):raise ValueError('nonfinite macro')
 return Decimal(str(macro))-Decimal(str(REFERENCE))>Decimal('.01')

def conditional_effects(b,d,f,r):
 return dict(F_minus_F_CE=f-b,R_minus_F_CE=r-b,D_minus_R=d-r,D_minus_F=d-f,interaction=d-f-r+b)

def freeze():
 import torch
 if (OUT/'evaluation_before_test.json').exists():raise FileExistsError('matrix already frozen')
 configs={a:read(OUT/'initialization'/f'{a}.json') for a in ARMS};units=[]
 for a in ARMS:
  for t in TASKS:
   split=read(ROOT/'data/splits/mips_paper5_outer'/f'{t}.json')
   for f in range(5):
    folder=ft_folder(a,t,f);m=read(folder/'metrics.json');scaler=read(folder/'scaler.json');entry=split['folds'][f]
    count=epochs(t)*((len(entry['train_indices'])+23)//24)
    if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(t) or m['updates']!={'attempted':count,'completed':count}:raise ValueError('incomplete FT budget')
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    if best['config']!=configs[a] or best['parent']!=str(PARENT) or best['task']!=t or best['fold']!=f or best['seed']!=42+f or best['scaler_mean']!=scaler['mean'] or best['scaler_scale']!=scaler['scale']:raise ValueError('best metadata mismatch')
    units.append(dict(arm=a,task=t,fold=f,best=str(folder/'best.pt'),parent=str(PARENT),best_epoch=best['best_epoch'],scaler=scaler,test_indices=sorted(entry['test_indices'])))
 if len(units)!=70:raise ValueError('incomplete frozen matrix')
 write(OUT/'evaluation_before_test.json',dict(validation_units=70,units=units,configs=configs,parent=str(PARENT),parent_fusion_gate='none',reference_macro=REFERENCE,outer_test='NOT_RUN',selection_by_validation=True))

def summarize():
 from sklearn.metrics import r2_score,mean_squared_error
 frozen=read(OUT/'evaluation_before_test.json');d_results=read(ROOT/'results/glt_ft_robustness_20261011/r1/final_review/seven_task_results.json')['arms']['D'];baseline=read(BASE/'final_review/seven_task_results.json')['arms']['F_CE']
 if float(np.mean([baseline[t]['r2']['mean'] for t in TASKS]))!=REFERENCE:raise ValueError('F_CE macro mismatch')
 papers={}
 for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
  p=[v.strip() for v in line.split('|')]
  if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):papers[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
 if set(papers)!=set(TASKS):raise ValueError('paper table incomplete')
 original_selection=read(BASE/'selection_before_test.json')
 reference_predictions={}
 for u in original_selection['units']:
  if u['arm']=='F_CE':
   predictions=Path(baseline[u['task']]['prediction_paths'][u['fold']])
   if not predictions.exists():raise FileNotFoundError('frozen baseline prediction path '+str(predictions))
   with np.load(predictions,allow_pickle=False) as p:reference_predictions[(u['task'],u['fold'])]={k:p[k].copy() for k in ('indices','sample_keys','y_true','scaler_mean','scaler_scale')}
 d_values={}
 for t in TASKS:
  d_values[t]=[]
  for f in range(5):
   path=ROOT/'results/glt_ft_robustness_20261011/r1/formal/D'/t/f'fold{f}'/'test_predictions.npz'
   with np.load(path,allow_pickle=False) as p:
    for key,value in reference_predictions[(t,f)].items():
     if not np.array_equal(p[key],value):raise ValueError('D reference identity mismatch '+key)
    if not np.isfinite(p['y_pred']).all():raise FloatingPointError('D prediction nonfinite')
    d_values[t].append(float(r2_score(p['y_true'],p['y_pred'])))
   if d_values[t][-1]!=d_results['tasks'][t]['folds'][f]['r2']:raise ValueError('saved D R2 mismatch')
 details={};rows=[]
 for a in ARMS:
  tasks={};values={}
  for t in TASKS:
   folds=[]
   for f in range(5):
    folder=OUT/'formal'/a/t/f'fold{f}';m=read(folder/'metrics.json')
    binding=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(a,t,f))
    if m['status']!='PASS' or m['outer_test']!='RUN':raise ValueError('formal unit incomplete')
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
     if p['indices'].tolist()!=binding['test_indices'] or int(p['best_epoch'])!=binding['best_epoch']:raise ValueError('frozen prediction identity mismatch')
     for key,value in reference_predictions[(t,f)].items():
      if not np.array_equal(p[key],value):raise ValueError('F_CE prediction rows/labels/scaler mismatch '+key)
     truth=p['y_true'];prediction=p['y_pred']
     if truth.shape!=prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():raise FloatingPointError('formal prediction corrupt')
     r2=float(r2_score(truth,prediction));rmse=float(np.sqrt(mean_squared_error(truth,prediction)))
     if not math.isfinite(r2) or not math.isfinite(rmse):raise FloatingPointError('nonfinite metric')
     source_valid=int(p['geometry_valid'].sum())
    old=read(BASE/'formal/F_CE'/t/f'fold{f}'/'metrics.json')
    row=dict(arm=a,task=t,fold=f,r2=r2,rmse=rmse,reference_r2=old['r2'],r2_delta=r2-old['r2'],reference_rmse=old['rmse'],rmse_delta=rmse-old['rmse'],best=binding['best'],predictions=str(folder/'test_predictions.npz'),original_valid_graphs=source_valid)
    folds.append(row);rows.append(row)
   tasks[t]={k:dict(mean=float(np.mean([x[k] for x in folds])),std=float(np.std([x[k] for x in folds],ddof=0))) for k in ('r2','rmse')}
   tasks[t].update(folds=folds,r2_delta=tasks[t]['r2']['mean']-baseline[t]['r2']['mean'],paper=dict(papers[t],delta=tasks[t]['r2']['mean']-papers[t]['r2']))
   values[t]=[r['r2'] for r in folds]
  macro=macro_r2(values);details[a]=dict(tasks=tasks,macro_test_r2=macro,macro_delta=macro-REFERENCE,qualifies=qualifies(macro))
 effects={}
 for t in TASKS:
  folds=[conditional_effects(details['F']['tasks'][t]['folds'][f]['reference_r2'],d_values[t][f],*[details[a]['tasks'][t]['folds'][f]['r2'] for a in ARMS]) for f in range(5)]
  effects[t]=dict(folds=folds,mean={k:float(np.mean([v[k] for v in folds])) for k in folds[0]},std={k:float(np.std([v[k] for v in folds],ddof=0)) for k in folds[0]})
 macro_effects=conditional_effects(REFERENCE,macro_r2(d_values),*[details[a]['macro_test_r2'] for a in ARMS])
 passed=[a for a in ARMS if details[a]['qualifies']];proposed=max(passed,key=lambda a:details[a]['macro_test_r2']) if passed else 'F_CE'
 write(OUT/'final_review/seven_task_results.json',dict(arms=details,reference_macro=REFERENCE,proposed_baseline=proposed,conditional_effects=effects,macro_effects=macro_effects,baseline='F_CE unchanged',deployment='I unchanged',protocol='seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only'))
 with (OUT/'final_review/formal_fold_details.csv').open('x',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# GLT-OPTIMIZATION S1 正式结果（待独立审查）','','F_CE参考宏值0.7855913245367517，唯一门槛为原精度宏差严格>0.01。seed42+fold；每task五折mean/std，std ddof0，再七task等权宏平均。TEST已用于开发选择，非独立盲测。','','|候选|宏R²|相对F_CE|通过|','|---|---:|---:|---|']
 for a in ARMS:lines.append(f"|{a}|{details[a]['macro_test_r2']:.9f}|{details[a]['macro_delta']:+.9f}|{details[a]['qualifies']}|")
 lines+=['','建议基线：'+proposed+'；不自动替换F_CE或部署I。','','|任务|F_CE|F|R|','|---|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+f"{baseline[t]['r2']['mean']:.9f}±{baseline[t]['r2']['std']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['r2']['mean']:.9f}±{details[a]['tasks'][t]['r2']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|任务|论文最佳模型|论文R²|F−论文|R−论文|','|---|---|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+papers[t]['model']+f"|{papers[t]['r2']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['paper']['delta']:+.9f}" for a in ARMS)+'|')
 lines+=['','|任务|F 原单位RMSE|R 原单位RMSE|','|---|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{details[a]['tasks'][t]['rmse']['mean']:.9f}±{details[a]['tasks'][t]['rmse']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|条件差|宏差|','|---|---:|']
 for k,v in macro_effects.items():lines.append(f'|{k}|{v:+.9f}|')
 lines+=['','完整逐fold配对差、四个条件效应及组合差分的task/fold/宏值在seven_task_results.json和formal_fold_details.csv。R²差正值较高，RMSE差负值较好。正组合差分不等于达标或多seed稳定互补。论文与项目数据规模、Egc/Xc版本、标签处理和精确split等价性未核实，仅作数值对照，不宣称公平超过论文。EAT/Eib/Tg不纳入七任务。']
 (OUT/'final_review/formal_results.md').write_text('\n'.join(lines)+'\n')

def handoff():
 if int((LOG/'final/exit.txt').read_text())!=0 or read(OUT/'execution.json')['status']!='待审查':raise ValueError('final engineering test not successful')
 report='\n\n## S1 实际执行交接（待审查）\n\n仅新增授权四文件并更新本文；共享编码器、训练条件、数据及既有参考未修改。F/R 在 joint 按同一独立 CPU Generator(seed=10000+42+fold)逐图 p0.20 分别关闭融合／读出；norm3 后置零，包括 bias，保留路径不 detach；不复用其他 mask 的 fused 缓存。head-only、validation、正式预测禁用抽样。\n\n实际预算：\n\n~~~json\n'+json.dumps(read(OUT/'consumption.json'),indent=2)+'\n~~~\n\n唯一最终 CPU 测试（空 CUDA_VISIBLE_DEVICES，300秒上限）：\n\n~~~text\n'+(LOG/'final/stdout.log').read_text().strip()+'\n~~~\n\n'+(OUT/'final_review/formal_results.md').read_text()+'\n\n窗口 Uni-Poly:glt_opt_r1_S1_controller；物理 GPU0/1/2/3，每卡最多一GPU单元，总并发≤4，内部cuda:0。日志 logs/glt_optimization_20261011/r1/S1/，checkpoint finetune/<arm>/<task>/foldN/best.pt，预测 formal/<arm>/<task>/foldN/test_predictions.npz，产物根 results/glt_optimization_20261011/r1/S1/。日志保留 command/cwd/environment/start/end/stdout/stderr/真实exit。工程正确性与性能门槛分开；尚未独立审查，不归档、不替换F_CE或部署I、不启动后续周期。\n'
 (OUT/'final_review/delivery.md').write_text(report)
 plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('总体状态：**执行中**（仅 S1 获授权，其余周期待授权）','总体状态：**待审查**（S1 执行结束，其余周期待授权）').replace('## 4. S1：分路径 dropout 对照（执行中）','## 4. S1：分路径 dropout 对照（待审查）').replace('状态：**执行中**。','状态：**待审查**。');plan.write_text(s+report)

if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif not sys.argv[1:]:freeze()
 else:raise ValueError('unexpected arguments')
