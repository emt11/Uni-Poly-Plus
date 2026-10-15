#!/usr/bin/env python3
"""Freeze validation selection, then summarize single formal predictions."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_periodic_seam_unit import OUT,ARMS,TASKS,read,write,ft_folder
BASE=ROOT/'results/glt_fusion_20261007/r1'

def select():
 if (OUT/'selection_before_test.json').exists():raise FileExistsError('selection already frozen')
 units=[];scores={};configs={a:read(OUT/'initialization'/f'{a}.json') for a in ARMS}
 for a in ARMS:
  task_gains={}
  for t in TASKS:
   gains=[]
   for f in range(5):
    folder=ft_folder(a,t,f);m=read(folder/'metrics.json');b=read(BASE/'finetune/B0'/t/f'fold{f}'/'metrics.json')
    if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=(60 if t=='egc' else 70):raise ValueError('FT incomplete')
    split=read(ROOT/'data/splits/mips_paper5_outer'/f'{t}.json')['folds'][f]
    if m['updates']['completed']!=m['epochs']*((len(split['train_indices'])+23)//24):raise ValueError('FT budget differs')
    gains.append((b['rmse']-m['rmse'])/b['rmse'])
    units.append(dict(arm=a,task=t,fold=f,best=str(folder/'best.pt'),scaler=read(folder/'scaler.json'),metrics=m))
   task_gains[t]=float(np.mean(gains))
  macro=float(np.mean(list(task_gains.values())));improved=sum(v>0 for v in task_gains.values());worst=min(task_gains.values())
  scores[a]=dict(macro=macro,strictly_improved_tasks=improved,worst_task=worst,task_gains=task_gains,qualifies=macro>=.01 and improved>=5 and worst>=-.02)
 eligible=[a for a in ARMS if scores[a]['qualifies']]
 chosen=min(eligible,key=lambda a:(-scores[a]['macro'],ARMS.index(a))) if eligible else 'B0'
 write(OUT/'selection_before_test.json',dict(validation_units=len(units),selected=chosen,deploy='I unchanged',configs=configs,units=units,scores=scores,rule='macro>=.01; >=5 strict improvements; worst>=-.02; equal parameters finite-first fixed-order',outer_test='NOT_RUN'))

def summarize():
 selection=read(OUT/'selection_before_test.json');detail={}
 for a in ARMS:
  detail[a]={}
  for t in TASKS:
   rows=[read(OUT/'formal'/a/t/f'fold{f}'/'metrics.json') for f in range(5)]
   if any(r['status']!='PASS' or r['outer_test']!='RUN' for r in rows):raise ValueError('formal incomplete')
   detail[a][t]={k:dict(mean=float(np.mean([r[k] for r in rows])),std=float(np.std([r[k] for r in rows],ddof=0))) for k in ('r2','rmse')}
   detail[a][t]['prediction_paths']=[str(OUT/'formal'/a/t/f'fold{f}'/'test_predictions.npz') for f in range(5)]
 combinations={}
 for t in TASKS:
  values=[]
  for f in range(5):
   rms={a:read(OUT/'formal'/a/t/f'fold{f}'/'metrics.json')['rmse'] for a in ARMS};b=read(BASE/'formal/B0'/t/f'fold{f}'/'metrics.json')['rmse']
   values.append(((rms['P_CE']-rms['P_RAW'])-(rms['F_CE']-rms['F_RAW']))/b)
  combinations[t]=dict(folds=values,mean=float(np.mean(values)),std=float(np.std(values)))
 paired={}
 for a,reference in (('F_RAW','T_RAW'),('F_CE','T0'),('P_RAW','F_RAW'),('P_CE','F_CE'),('F_CE','F_RAW'),('P_CE','P_RAW')):
  paired[a+'-'+reference]={}
  refroot=BASE if reference in ('T_RAW','T0') else OUT
  for t in TASKS:
   folds=[]
   for f in range(5):
    cand=read(OUT/'formal'/a/t/f'fold{f}'/'metrics.json')
    ref=read(refroot/'formal'/reference/t/f'fold{f}'/'metrics.json')
    folds.append(dict(fold=f,rmse_delta=cand['rmse']-ref['rmse'],relative_rmse_improvement=(ref['rmse']-cand['rmse'])/ref['rmse'],r2_delta=cand['r2']-ref['r2']))
   paired[a+'-'+reference][t]=dict(folds=folds,mean_rmse_delta=float(np.mean([v['rmse_delta'] for v in folds])),mean_r2_delta=float(np.mean([v['r2_delta'] for v in folds])))
 chosen=selection['selected'];papers={'eea':.934,'egb':.923,'egc':.911,'ei':.839,'eps':.787,'nc':.856,'xc':.423}
 table={}
 for t,p in papers.items():
  if chosen=='B0':
   vals=[read(BASE/'formal/B0'/t/f'fold{f}'/'metrics.json')['r2'] for f in range(5)]
   m=float(np.mean(vals));std=float(np.std(vals))
  else:m=detail[chosen][t]['r2']['mean'];std=detail[chosen][t]['r2']['std']
  table[t]=dict(project_r2=m,std=std,paper_r2=p,delta=m-p,paper_model='Periodic-TDL (+Chem,+RC)' if t in ('egb','egc') else 'Periodic-TDL')
 write(OUT/'final_review'/'seven_task_results.json',dict(selected=chosen,arms=detail,paper_comparison=table,paired=paired,combination= combinations,macro_combination=float(np.mean([v['mean'] for v in combinations.values()])),aggregation='paper5_outer seed42+fold five folds ddof0',protocol='shared development folds; Egc3380 vs paper4124 Xc432 vs430; split/scaling equivalence unverified; numeric comparison only'))


def handoff():
 log=ROOT/'logs/glt_periodic_seam_20261009/r1'
 execution=read(OUT/'execution.json');consumption=read(OUT/'consumption.json')
 if execution['status']!='待审查' or int((log/'final/exit.txt').read_text())!=0:
  raise ValueError('handoff requires completed execution and the sole successful pytest')
 results=read(OUT/'final_review/seven_task_results.json')
 text=['# GLT-PERIODIC-SEAM-20261009-01/r1 执行交付（待审查）','',
  '四臂 PT、固定 validation、140 FT 和 140 正式预测已完成；唯一 CPU pytest 的真实 exit 为 0。独立只读审查尚未执行，不标记已完成，不归档模型历史。',
  '', '## 实际预算', '', '```json',json.dumps(consumption,ensure_ascii=False,indent=2),'```','',
  '## 唯一工程测试', '', '`timeout 300s python -m pytest -q tests/test_glt_periodic_seam.py`', '',
  '日志：'+str(log/'final/stdout.log')+'；stderr：'+str(log/'final/stderr.log')+'；exit：'+str(log/'final/exit.txt')+'.', '',
  '```text',(log/'final/stdout.log').read_text().strip(),'```','',
  '## 七任务论文数值对照', '', '选定 S='+results['selected']+'；部署 I 保留。R² 越高越好，差值为本项目减论文参考。', '',
  '|任务|S R² mean±std|论文最佳模型|论文 R²|差值|','|---|---:|---|---:|---:|']
 for task in TASKS:
  row=results['paper_comparison'][task]
  text.append(f"|{task}|{row['project_r2']:.3f}±{row['std']:.3f}|{row['paper_model']}|{row['paper_r2']:.3f}|{row['delta']:+.3f}|")
 text.extend(['', '口径：paper5_outer，seed42+fold，五折 mean/std(ddof0)。共享 validation/test 已参与开发；项目 Egc3380／Xc432 与论文表参考4124／430不同，精确 split 和 Xc scaling 等价性未核实，仅做数值对照。', '',
  '四臂、逐折正式预测、配对差异及组合差分：'+str(OUT/'final_review/seven_task_results.json')+'；选择冻结：'+str(OUT/'selection_before_test.json')+'.', '',
  '## 路径与未完成事项', '', '日志根：'+str(log)+'；产物根：'+str(OUT)+'.', '',
  '控制器 window：Uni-Poly:glt_ps_r1_controller；各单元 command/cwd/start/end/stdout/stderr/exit 保留在独立日志目录。原始数据、旧产物及部署 I 保留；旧实验、旧测试、方案三、MD、ALIGN、FGR、KFuse、torsion、额外 seed／调参均未执行。', '',
  '未完成：独立只读审查。审查者只核对实际 diff、上述唯一测试和既有日志／预测，不追加运行。通过后才能更新已完成并归档 PROJECT_HISTORY.md。'])
 report='\n'.join(text)+'\n'
 path=OUT/'final_review/delivery.md'
 with path.open('x') as stream:stream.write(report)
 plan=ROOT/'CURRENT_PLAN.md';current=plan.read_text()
 current=current.replace('状态：**执行中**。','状态：**待审查**。',1)
 current+='\n## 本轮实际执行交付（独立审查尚未执行）\n\n'+report
 plan.write_text(current)
 pipeline=ROOT/'PIPELINE.md';current=pipeline.read_text().replace('## GLT 周期接缝周期（2026-10-09，执行中）','## GLT 周期接缝周期（2026-10-09，待审查）')
 current+='\n本周期授权执行已完成、唯一 pytest exit=0；独立审查尚未执行。实际预算、七任务结果及预测路径见 `results/glt_periodic_seam_20261009/r1/final_review/delivery.md`，状态为待审查。\n'
 pipeline.write_text(current)

if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 else:select()
