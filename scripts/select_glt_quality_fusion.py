#!/usr/bin/env python3
"""Freeze all evaluations, then select once by the authorized macro TEST rule."""
import csv,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_quality_fusion_unit import OUT,ARMS,TASKS,read,write,ft_folder,parent
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
LOG=ROOT/'logs/glt_quality_fusion_20261010/r1'
REFERENCE=0.7855913245367517

def macro_r2(tasks):
 if set(tasks)!=set(TASKS):raise ValueError('macro requires all seven tasks')
 means=[]
 for task in TASKS:
  values=np.asarray(tasks[task],dtype=float)
  if values.shape!=(5,) or not np.isfinite(values).all():raise ValueError('five finite folds required')
  means.append(float(values.mean()))
 return float(np.mean(means))

def qualifies(macro,reference=REFERENCE):
 if not math.isfinite(macro) or not math.isfinite(reference):raise ValueError('nonfinite macro')
 return Decimal(str(macro))-Decimal(str(reference))>Decimal('0.01')

def select():
 if (OUT/'evaluation_before_test.json').exists():raise FileExistsError('evaluation already frozen')
 units=[];configs={a:read(OUT/'initialization'/f'{a}.json') for a in ARMS}
 for a in ARMS:
  for t in TASKS:
   split=read(ROOT/'data/splits/mips_paper5_outer'/f'{t}.json')
   for f in range(5):
    folder=ft_folder(a,t,f);m=read(folder/'metrics.json');entry=split['folds'][f]
    expected_epochs=60 if t=='egc' else 70
    if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=expected_epochs:raise ValueError('FT incomplete')
    if m['updates']!={'attempted':expected_epochs*((len(entry['train_indices'])+23)//24),'completed':expected_epochs*((len(entry['train_indices'])+23)//24)}:raise ValueError('FT budget differs')
    if not (folder/'best.pt').is_file():raise FileNotFoundError('best missing')
    units.append(dict(arm=a,task=t,fold=f,best=str(folder/'best.pt'),parent=str(parent(a)),scaler=read(folder/'scaler.json'),metrics=m,test_indices=sorted(entry['test_indices'])))
 if len(units)!=140:raise ValueError('evaluation matrix incomplete')
 write(OUT/'evaluation_before_test.json',dict(validation_units=140,units=units,configs=configs,quality=read(OUT/'quality_completed/summary.json'),outer_test='NOT_RUN',selection_by_validation=False,reference_macro=REFERENCE,rule='seven-task equal-weight macro TEST R2 delta strictly >0.01',deployment='I unchanged'))

def summarize():
 from sklearn.metrics import r2_score
 frozen=read(OUT/'evaluation_before_test.json');details={};fold_rows=[];paired={};papers={}
 # Read the seven-task best envelope directly from the existing paper table.
 for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
  parts=[x.strip() for x in line.split('|')]
  if len(parts)==5 and parts[1].lower() in TASKS and parts[3].startswith('**'):
   papers[parts[1].lower()]=dict(model=parts[2],r2=float(parts[3].strip('*').split('±')[0]))
 if set(papers)!=set(TASKS):raise ValueError('paper reference incomplete')
 base=read(BASE/'final_review/seven_task_results.json')['arms']['F_CE']
 ref=float(np.mean([base[t]['r2']['mean'] for t in TASKS]))
 if ref!=REFERENCE:raise ValueError('F_CE reference macro mismatch')
 quality=read(OUT/'quality_completed/downstream_quality.json')
 for a in ARMS:
  tasks={};task_values={};paired[a]={}
  for t in TASKS:
   folds=[];quality_groups={True:([],[]),False:([],[])}
   for f in range(5):
    folder=OUT/'formal'/a/t/f'fold{f}';m=read(folder/'metrics.json');entry=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(a,t,f))
    if m['status']!='PASS' or m['outer_test']!='RUN':raise ValueError('formal incomplete')
    with np.load(folder/'test_predictions.npz',allow_pickle=False) as pred:
     if pred['indices'].tolist()!=entry['test_indices'] or pred['scaler_mean'].tolist()!=entry['scaler']['mean'] or pred['scaler_scale'].tolist()!=entry['scaler']['scale']:raise ValueError('formal row/scaler mismatch')
     truth=pred['y_true'];prediction=pred['y_pred'];flags=pred['quality_eligible'].astype(bool)
     expected=[quality[str(k)]['quality_eligible'] for k in pred['sample_keys']]
     if flags.tolist()!=expected:raise ValueError('formal quality flags mismatch')
     if not np.isfinite(truth).all() or not np.isfinite(prediction).all():raise FloatingPointError('formal nonfinite')
     for flag in (True,False):
      quality_groups[flag][0].extend(truth[flags==flag].tolist());quality_groups[flag][1].extend(prediction[flags==flag].tolist())
    old=read(BASE/'formal/F_CE'/t/f'fold{f}'/'metrics.json')
    row=dict(arm=a,task=t,fold=f,r2=m['r2'],rmse=m['rmse'],reference_r2=old['r2'],r2_delta=m['r2']-old['r2'],reference_rmse=old['rmse'],rmse_delta=m['rmse']-old['rmse'],best=entry['best'],predictions=str(folder/'test_predictions.npz'),quality_eligible=int(flags.sum()),test_rows=len(flags))
    folds.append(row);fold_rows.append(row)
   task_values[t]=[r['r2'] for r in folds]
   tasks[t]={k:dict(mean=float(np.mean([r[k] for r in folds])),std=float(np.std([r[k] for r in folds],ddof=0))) for k in ('r2','rmse')}
   tasks[t].update(folds=folds,r2_delta=tasks[t]['r2']['mean']-base[t]['r2']['mean'],paper= dict(papers[t],delta=tasks[t]['r2']['mean']-papers[t]['r2']),quality_groups={})
   for flag,(truth,prediction) in quality_groups.items():
    tasks[t]['quality_groups'][str(flag)]=dict(n=len(truth),pooled_rmse=float(np.sqrt(np.mean((np.array(truth)-np.array(prediction))**2))) if truth else None,pooled_r2=float(r2_score(truth,prediction)) if len(truth)>1 else None,role='descriptive only; no extra inference; not selection')
   paired[a][t]=dict(folds=[r['r2_delta'] for r in folds],mean=tasks[t]['r2_delta'])
  macro=macro_r2(task_values)
  details[a]=dict(tasks=tasks,macro_test_r2=macro,macro_delta=macro-ref,qualifies=qualifies(macro,ref))
 order=('Q','Q_G','Q_A','Q_GA');passed=[a for a in order if details[a]['qualifies']]
 proposed=max(passed,key=lambda a:details[a]['macro_test_r2']) if passed else 'F_CE'
 combination={t:dict(folds=[(details['Q_GA']['tasks'][t]['folds'][f]['r2']-details['Q_G']['tasks'][t]['folds'][f]['r2'])-(details['Q_A']['tasks'][t]['folds'][f]['r2']-details['Q']['tasks'][t]['folds'][f]['r2']) for f in range(5)]) for t in TASKS}
 for v in combination.values():v['mean']=float(np.mean(v['folds']));v['std']=float(np.std(v['folds']))
 write(OUT/'final_review/seven_task_results.json',dict(arms=details,reference_macro=ref,proposed_baseline=proposed,baseline='F_CE unchanged pending independent review',deployment='I unchanged',paired=paired,combination=combination,macro_combination=float(np.mean([v['mean'] for v in combination.values()])),protocol='five folds seed42+fold ddof0; shared TEST used for development selection; numeric paper comparison only'))
 with (OUT/'final_review/formal_fold_details.csv').open('x',newline='') as out:
  writer=csv.DictWriter(out,fieldnames=list(fold_rows[0]));writer.writeheader();writer.writerows(fold_rows)
 lines=['# 四候选正式结果（待独立审查）','','宏值先求各任务五折 R² 均值，再七任务等权平均。唯一性能条件为相对 F_CE 宏 R² 绝对差严格 >0.01。TEST 用于本轮开发选择，非独立盲测。','','|候选|宏 TEST R²|相对 F_CE|通过|','|---|---:|---:|---|']
 for a in ARMS:
  r=details[a];lines.append(f"|{a}|{r['macro_test_r2']:.3f}|{r['macro_delta']:+.3f}|{r['qualifies']}|")
 lines+=['','建议基线：'+proposed+'；当前基线与部署保持原样，待独立审查。','','|任务|F_CE|Q|Q_A|Q_G|Q_GA|','|---|---:|---:|---:|---:|---:|']
 for t in TASKS:
  vals=[f"{details[a]['tasks'][t]['r2']['mean']:.3f}±{details[a]['tasks'][t]['r2']['std']:.3f}" for a in ARMS]
  lines.append('|'+t+f"|{base[t]['r2']['mean']:.3f}±{base[t]['r2']['std']:.3f}|"+'|'.join(vals)+'|')
 lines+=['','## 七任务论文数值对照','','|任务|论文最佳模型|论文 R²|Q−论文|Q_A−论文|Q_G−论文|Q_GA−论文|','|---|---|---:|---:|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+papers[t]['model']+f"|{papers[t]['r2']:.3f}|"+'|'.join(f"{details[a]['tasks'][t]['paper']['delta']:+.3f}" for a in ARMS)+'|')
 lines+=['','Egc 数据量不同；论文与项目标签缩放不同，七任务精确 split 等价性未核实，只报告数值对照，不宣称公平超过论文。全部完整精度、逐fold、质量分层及路径见 seven_task_results.json 和 formal_fold_details.csv。']
 (OUT/'final_review/formal_results.md').write_text('\n'.join(lines)+'\n')

def handoff():
 execution=read(OUT/'execution.json')
 if execution['status']!='待审查' or int((LOG/'final/exit.txt').read_text())!=0:raise ValueError('handoff requires sole successful final test')
 report=['# GLT-QUALITY-FUSION-20261010-01/r1 执行交付（待审查）','','已完成授权质量派生、四臂 PT、140 FT、140 正式预测和唯一工程测试。自检不替代独立审查；未归档新模型周期，未替换 F_CE 或部署 I。','','## 质量保留率','','```json',json.dumps(read(OUT/'quality_completed/summary.json'),ensure_ascii=False,indent=2),'```','','## 实际预算','','```json',json.dumps(read(OUT/'consumption.json'),indent=2),'```','','## 唯一最终 CPU 测试','','```text',(LOG/'final/stdout.log').read_text().strip(),'```','','## 实际结果','',(OUT/'final_review/formal_results.md').read_text(),'','日志根：'+str(LOG)+'；产物根：'+str(OUT)+'。每个单元保留 command/cwd/start/end/stdout/stderr/真实exit。tmux controller：Uni-Poly:glt_qf_r1_controller；quality、initialization、pt_*、ft_*、test_*、summary、final 为独立窗口。','','未完成事项：Codex 独立只读审查实际 diff、唯一测试、预算与既有预测产物；不得追加测试或实验。']
 text='\n'.join(report)+'\n';(OUT/'final_review/delivery.md').write_text(text)
 plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('状态：**执行中**。','状态：**待审查**。',1);plan.write_text(s+'\n\n'+text)

if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif not sys.argv[1:]:select()
 else:raise ValueError('unknown arguments')
