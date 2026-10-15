#!/usr/bin/env python3
"""F1 complete validation freeze and one saved-prediction summary."""
import csv,difflib,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_ft_graph_interaction_unit import OUT,BASE,U1,ARMS,TASKS,PROFILE,PARAMETERS,parent,head_path,epochs,ft_folder,read,write,validate_config,validate_best,interaction_config
LOG=ROOT/'logs/glt_optimization_20261011/r5/F1'
REFERENCE=.7855913245367517
MECHANISM_REFERENCE=.7868789263757971

def macro_r2(tasks):
 if set(tasks)!=set(TASKS):raise ValueError('seven tasks required')
 values=[np.asarray(tasks[t],dtype=float) for t in TASKS]
 if any(x.shape!=(5,) or not np.isfinite(x).all() for x in values):raise ValueError('five finite folds required')
 return float(np.mean([x.mean() for x in values]))
def qualifies(value):return Decimal(str(value))-Decimal(str(REFERENCE))>Decimal('.01')
def freeze():
 import torch
 if (OUT/'evaluation_before_test.json').exists():raise FileExistsError('matrix already frozen')
 cfg=validate_config(read(OUT/'initialization/F1_MUL.json'),'F1_MUL');units=[]
 for task in TASKS:
  split=read(ROOT/'data/splits/mips_paper5_outer'/f'{task}.json')
  for fold in range(5):
   folder=ft_folder('F1_MUL',task,fold);m=read(folder/'metrics.json');scaler=read(folder/'scaler.json');entry=split['folds'][fold]
   count=70*((len(entry['train_indices'])+23)//24)
   if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=70 or m['updates']!={'attempted':count,'completed':count}:raise ValueError('FT matrix incomplete')
   best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);validate_best(best,'F1_MUL',task,fold,cfg)
   if best['scaler_mean']!=scaler['mean'] or best['scaler_scale']!=scaler['scale'] or not 1<=best['best_epoch']<=70 or any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('best/scaler/epoch invalid')
   units.append(dict(arm='F1_MUL',task=task,fold=fold,best=str(folder/'best.pt'),parent=str(parent('F1_MUL')),best_epoch=best['best_epoch'],head_initial=str(head_path('F1_MUL',task,fold)),scaler=scaler,test_indices=sorted(entry['test_indices']),**interaction_config()))
 if len(units)!=35:raise ValueError('35 complete units required')
 write(OUT/'evaluation_before_test.json',dict(validation_units=35,units=units,configs={'F1_MUL':cfg},parent=str(parent('F1_MUL')),outer_test='NOT_RUN',selection_by_validation=True))

def summarize():
 from sklearn.metrics import r2_score,mean_squared_error
 frozen=read(OUT/'evaluation_before_test.json');u=read(U1/'final_review/recovery1/seven_task_results.json')['arms']['U0_FUSED'];b=read(BASE/'final_review/seven_task_results.json')['arms']['F_CE']
 if u['macro_test_r2']!=MECHANISM_REFERENCE or float(np.mean([b[t]['r2']['mean'] for t in TASKS]))!=REFERENCE:raise ValueError('reference macro differs')
 papers={}
 for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
  p=[v.strip() for v in line.split('|')]
  if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):papers[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
 if set(papers)!=set(TASKS):raise ValueError('paper matrix incomplete')
 rows=[];tasks={};values={}
 for task in TASKS:
  folds=[]
  for fold in range(5):
   folder=OUT/'formal/F1_MUL'/task/f'fold{fold}';m=read(folder/'metrics.json');binding=next(x for x in frozen['units'] if x['task']==task and x['fold']==fold)
   if m['status']!='PASS' or m['outer_test']!='RUN' or m['parameters']!=PARAMETERS:raise ValueError('formal unit/parameters invalid')
   with np.load(folder/'test_predictions.npz',allow_pickle=False) as p:
    if p['indices'].tolist()!=binding['test_indices'] or int(p['best_epoch'])!=binding['best_epoch']:raise ValueError('formal binding mismatch')
    truth=p['y_true'];pred=p['y_pred']
    if truth.shape!=pred.shape or not np.isfinite(truth).all() or not np.isfinite(pred).all():raise FloatingPointError('nonfinite prediction')
    references={}
    for name,path in [('U0_FUSED',Path(u['tasks'][task]['folds'][fold]['predictions'])),('F_CE',Path(b[task]['prediction_paths'][fold]))]:
     with np.load(path,allow_pickle=False) as ref:
      for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
       if not np.array_equal(p[key],ref[key]):raise ValueError('reference identity/label/scaler mismatch '+name+' '+key)
      references[name]=dict(r2=float(r2_score(ref['y_true'],ref['y_pred'])),rmse=float(np.sqrt(mean_squared_error(ref['y_true'],ref['y_pred']))),predictions=str(path))
    r2=float(r2_score(truth,pred));rmse=float(np.sqrt(mean_squared_error(truth,pred)))
   if not math.isfinite(r2) or not math.isfinite(rmse):raise FloatingPointError('nonfinite metric')
   row=dict(task=task,fold=fold,r2=r2,rmse=rmse,U0_FUSED_r2=references['U0_FUSED']['r2'],F_CE_r2=references['F_CE']['r2'],
    r2_minus_U0_FUSED=r2-references['U0_FUSED']['r2'],r2_minus_F_CE=r2-references['F_CE']['r2'],
    rmse_minus_U0_FUSED=rmse-references['U0_FUSED']['rmse'],rmse_minus_F_CE=rmse-references['F_CE']['rmse'],
    best=binding['best'],predictions=str(folder/'test_predictions.npz'),reference_predictions=references)
   rows.append(row);folds.append(row)
  stats={k:dict(mean=float(np.mean([x[k] for x in folds])),std=float(np.std([x[k] for x in folds],ddof=0))) for k in ('r2','rmse','r2_minus_U0_FUSED','r2_minus_F_CE','rmse_minus_U0_FUSED','rmse_minus_F_CE')}
  stats.update(folds=folds,paper=dict(papers[task],delta=stats['r2']['mean']-papers[task]['r2']))
  tasks[task]=stats;values[task]=[x['r2'] for x in folds]
 macro=macro_r2(values);details=dict(tasks=tasks,macro_test_r2=macro,macro_delta=macro-REFERENCE,macro_minus_U0_FUSED=macro-MECHANISM_REFERENCE,qualifies=qualifies(macro),parameters=PARAMETERS,configuration=frozen['configs']['F1_MUL'])
 write(OUT/'final_review/seven_task_results.json',dict(arms={'F1_MUL':details},reference_macro=REFERENCE,mechanism_reference_macro=MECHANISM_REFERENCE,consumption=read(OUT/'consumption.json'),protocol='seven tasks, five folds seed42+fold, std ddof0; shared TEST development results, paper numeric comparison only'))
 with (OUT/'final_review/formal_fold_details.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# F1 正式结果（待独立审查）','',f'F1_MUL宏R² {macro:.12f}，相对U0_FUSED {macro-MECHANISM_REFERENCE:+.12f}，相对历史F_CE {macro-REFERENCE:+.12f}；历史门槛严格>0.01：{qualifies(macro)}。',
  '', '|任务|F1 R² mean±std|原单位RMSE mean±std|F1−U0_FUSED|F1−F_CE|论文最佳模型|论文R²|F1−论文|','|---|---:|---:|---:|---:|---|---:|---:|']
 for t in TASKS:
  x=tasks[t];lines.append(f"|{t}|{x['r2']['mean']:.9f}±{x['r2']['std']:.9f}|{x['rmse']['mean']:.9f}±{x['rmse']['std']:.9f}|{x['r2_minus_U0_FUSED']['mean']:+.9f}|{x['r2_minus_F_CE']['mean']:+.9f}|{x['paper']['model']}|{x['paper']['r2']:.9f}|{x['paper']['delta']:+.9f}|")
 lines+=['','每任务五折seed42+fold、std ddof0，七任务等权宏值。R²差正值较好，RMSE差负值较好；逐fold/task配对差、路径及scaler/标签/样本一致性见JSON/CSV。共享TEST已用于开发，非独立盲测；论文数据/split/统计协议等价性未完全核实，仅数值对照。',
  '',f'总FT参数{PARAMETERS}，新增98944；交互容量增加也是受控候选的一部分，无等容量对照。h₂已融合三维信息，不能称纯二维；正均值不自动证明稳定优势。工程验收与性能分别报告，F_CE/部署I不替换。']
 (OUT/'final_review/formal_results.md').write_text('\n'.join(lines)+'\n')

def capture_diff():
 names=['CURRENT_PLAN.md','scripts/run_glt_ft_graph_interaction_unit.py','scripts/run_glt_ft_graph_interaction.py','scripts/select_glt_ft_graph_interaction.py','tests/test_glt_ft_graph_interaction.py'];diff=[]
 for name in names:
  p=LOG/'source_before'/name;before=p.read_text() if p.exists() else ''
  diff.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
 destination=OUT/'final_review/actual_changes.diff';destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x') as f:f.write(''.join(diff))
def handoff():
 if int((LOG/'final/exit.txt').read_text())!=0 or read(OUT/'execution.json')['status']!='待审查':raise ValueError('final test not successful')
 report='\n\n## F1 r5 实际执行交接（待审查）\n\n完整35 FT、冻结、35正式预测、一次汇总及唯一最终CPU测试已执行。只新增四个入口/测试文件并更新本文，共享源码与旧产物未修改。新wrapper独立CPU初始化、输出零初态、最终有效性输出后清零，无detach或源样本修改。\n\n实耗：\n\n```json\n'+json.dumps(read(OUT/'consumption.json'),indent=2)+'\n```\n\n唯一CPU测试（空CUDA_VISIBLE_DEVICES、timeout300秒、exit0）：\n\n```text\n'+(LOG/'final/stdout.log').read_text().strip()+'\n```\n\n'+(OUT/'final_review/formal_results.md').read_text()+'\n\n产物results/glt_optimization_20261011/r5/F1/；日志logs/glt_optimization_20261011/r5/F1/；窗口Uni-Poly:glt_opt_r5_F1_controller。物理0/1/2/3单卡四路、内部cuda:0。best位于finetune/F1_MUL/<task>/foldN/best.pt，预测formal/F1_MUL/<task>/foldN/test_predictions.npz，绑定evaluation_before_test.json，完整配置initialization/F1_MUL.json。实际增量final_review/actual_changes.diff，逐fold formal_fold_details.csv。工程PASS与性能门槛分开报告；自检不代替独立审查。历史、SOTA、F_CE与部署I保持，全部未实施路线保留，其他周期未启动。\n'
 (OUT/'final_review/delivery.md').write_text(report)
 p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('**F1，执行中**','**F1，待审查**').replace('（r5，执行中）','（r5，待审查）');p.write_text(s+report)
if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif sys.argv[1:]==['--capture-diff']:capture_diff()
 elif not sys.argv[1:]:freeze()
 else:raise ValueError('invalid selection arguments')
