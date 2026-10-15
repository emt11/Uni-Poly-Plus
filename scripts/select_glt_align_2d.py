#!/usr/bin/env python3
"""Freeze all FT bests, summarize each saved formal prediction exactly once."""
import csv,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_align_2d_unit import OUT,RECOVERY,REVIEW,BASE,parent,prediction_config,head_path,PROFILE,ARMS,TASKS,read,write,ft_folder,epochs,validate_config
ORIGINAL_LOG=ROOT/'logs/glt_optimization_20261011/r4/U1'
LOG=ORIGINAL_LOG/'recovery1'
REFERENCE=.7855913245367517

def macro_r2(tasks):
 if set(tasks)!=set(TASKS):raise ValueError('complete seven-task matrix required')
 values=[np.asarray(tasks[t],dtype=float) for t in TASKS]
 if any(v.shape!=(5,) or not np.isfinite(v).all() for v in values):raise ValueError('five finite folds required')
 return float(np.mean([v.mean() for v in values]))

def qualifies(macro):
 if not math.isfinite(macro):raise ValueError('nonfinite macro')
 return Decimal(str(macro))-Decimal(str(REFERENCE))>Decimal('.01')

def conditional_effects(b,z,a,f,af):
 return dict(UA_2D_minus_U0_2D=a-z,UA_FUSED_minus_U0_FUSED=af-f,UA_2D_minus_UA_FUSED=a-af,
  U0_2D_minus_U0_FUSED=z-f,U0_FUSED_minus_F_CE=f-b,U0_2D_minus_F_CE=z-b,UA_2D_minus_F_CE=a-b,
  UA_FUSED_minus_F_CE=af-b,ALIGN_path_difference=(af-f)-(a-z))

def freeze():
 import torch
 if (OUT/'evaluation_before_test.json').exists():raise FileExistsError('matrix already frozen')
 configs={a:validate_config(read(OUT/'initialization'/f'{a}.json'),a) for a in ARMS};units=[]
 for a in ARMS:
  for t in TASKS:
   split=read(ROOT/'data/splits/mips_paper5_outer'/f'{t}.json')
   for f in range(5):
    folder=ft_folder(a,t,f);m=read(folder/'metrics.json');scaler=read(folder/'scaler.json');entry=split['folds'][f]
    count=epochs(t)*((len(entry['train_indices'])+23)//24)
    if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=epochs(t) or m['updates']!={'attempted':count,'completed':count}:raise ValueError('incomplete FT budget')
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    if best['config']!=configs[a] or best['parent']!=str(parent(a)) or best['task']!=t or best['fold']!=f or best['seed']!=42+f or best['scaler_mean']!=scaler['mean'] or best['scaler_scale']!=scaler['scale']:raise ValueError('best metadata mismatch')
    if best.get('prediction_path')!=configs[a]['prediction_path'] or best.get('input_dim')!=configs[a]['input_dim'] or best.get('head_initial')!=str(head_path(a,t,f)):raise ValueError('best path/head binding mismatch')
    if not 1<=best['best_epoch']<=epochs(t) or any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('best epoch/tensor invalid')
    units.append(dict(arm=a,task=t,fold=f,best=str(folder/'best.pt'),parent=str(parent(a)),best_epoch=best['best_epoch'],head_initial=str(head_path(a,t,f)),**prediction_config(a),scaler=scaler,test_indices=sorted(entry['test_indices'])))
 if len(units)!=140:raise ValueError('incomplete frozen matrix')
 write(OUT/'evaluation_before_test.json',dict(validation_units=140,units=units,configs=configs,parents={a:str(parent(a)) for a in ARMS},parent_fusion_gate='none',reference_macro=REFERENCE,outer_test='NOT_RUN',selection_by_validation=True))

def summarize():
 from sklearn.metrics import r2_score,mean_squared_error
 frozen=read(OUT/'evaluation_before_test.json');baseline=read(BASE/'final_review/seven_task_results.json')['arms']['F_CE']
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
  macro=macro_r2(values);details[a]=dict(parameters=read(OUT/'formal'/a/'eea'/'fold0'/'metrics.json')['parameters'],pt_parameters=read(OUT/'initialization'/'summary.json')['pt_parameters']-(0 if a.startswith('UA_') else 328448),configuration=frozen['configs'][a],parent=str(parent(a)),tasks=tasks,macro_test_r2=macro,macro_delta=macro-REFERENCE,qualifies=qualifies(macro))
 effects={};rmse_effects={}
 for t in TASKS:
  folds=[conditional_effects(details['U0_2D']['tasks'][t]['folds'][f]['reference_r2'],*[details[a]['tasks'][t]['folds'][f]['r2'] for a in ARMS]) for f in range(5)]
  effects[t]=dict(folds=folds,mean={k:float(np.mean([v[k] for v in folds])) for k in folds[0]},std={k:float(np.std([v[k] for v in folds],ddof=0)) for k in folds[0]})
  rmse_folds=[conditional_effects(details['U0_2D']['tasks'][t]['folds'][f]['reference_rmse'],*[details[a]['tasks'][t]['folds'][f]['rmse'] for a in ARMS]) for f in range(5)]
  rmse_effects[t]=dict(folds=rmse_folds,mean={k:float(np.mean([v[k] for v in rmse_folds])) for k in rmse_folds[0]},std={k:float(np.std([v[k] for v in rmse_folds],ddof=0)) for k in rmse_folds[0]})
 macro_effects=conditional_effects(REFERENCE,*[details[a]['macro_test_r2'] for a in ARMS])
 passed=[a for a in ARMS if details[a]['qualifies']];proposed=max(passed,key=lambda a:details[a]['macro_test_r2']) if passed else 'F_CE'
 write(REVIEW/'seven_task_results.json',dict(arms=details,reference_macro=REFERENCE,proposed_baseline=proposed,conditional_effects=effects,rmse_conditional_effects=rmse_effects,macro_effects=macro_effects,baseline='F_CE unchanged',deployment='I unchanged',protocol='seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only'))
 with (REVIEW/'formal_fold_details.csv').open('x',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# U1 正式结果（待独立审查）','','共享TEST已用于开发；论文协议等价性未完全核实，仅数值对照。七任务各五折seed42+fold，std ddof0，七任务等权宏平均。','','|条件|宏R²|相对历史F_CE|严格>0.01|','|---|---:|---:|---|']
 for a in ARMS:lines.append(f"|{a}|{details[a]['macro_test_r2']:.9f}|{details[a]['macro_delta']:+.9f}|{details[a]['qualifies']}|")
 lines+=['','|任务|历史F_CE|'+ '|'.join(ARMS)+'|','|---|'+'---:|'*5]
 for t in TASKS:lines.append('|'+t+'|'+f"{baseline[t]['r2']['mean']:.9f}±{baseline[t]['r2']['std']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['r2']['mean']:.9f}±{details[a]['tasks'][t]['r2']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|任务|论文最佳模型|论文R²|'+ '|'.join(a+'−论文' for a in ARMS)+'|','|---|---|'+'---:|'*5]
 for t in TASKS:lines.append('|'+t+'|'+papers[t]['model']+f"|{papers[t]['r2']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['paper']['delta']:+.9f}" for a in ARMS)+'|')
 lines+=['','|任务|'+ '|'.join(a+' 原单位RMSE' for a in ARMS)+'|','|---|'+'---:|'*4]
 for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{details[a]['tasks'][t]['rmse']['mean']:.9f}±{details[a]['tasks'][t]['rmse']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|配对差|宏差|','|---|---:|']
 for k,v in macro_effects.items():lines.append(f'|{k}|{v:+.9f}|')
 lines+=['','|任务|'+ '|'.join(macro_effects)+'|','|---|'+'---:|'*len(macro_effects)]
 for t in TASKS:lines.append('|'+t+'|'+'|'.join(f'{v:+.9f}' for v in effects[t]['mean'].values())+'|')
 lines+=['','实际模型参数量：'+str({a:details[a]['parameters'] for a in ARMS}),
  '','完整逐fold差、task mean/std及宏差在seven_task_results.json；原单位RMSE差负值较好，R²差正值较高。ALIGN归因只依据同路径配对差；跨路径同时改变输入模态、token融合和head容量，不称单独三维因果贡献。组合差分只表示ALIGN效应在两路径的差异。U0_FUSED−历史F_CE为统一FT配方整体效应。',
  '','统一FT配置：'+json.dumps(PROFILE,ensure_ascii=False),'','研究F_CE与部署I不替换；不按task选择候选，不自动启动其他周期。']
 (REVIEW/'formal_results.md').write_text('\n'.join(lines)+'\n')

def capture_diff():
 import difflib,subprocess
 changes=['CURRENT_PLAN.md','src/modules/glt_dual_pretrain.py','scripts/pretrain_glt_dual.py','scripts/finetune_mcl_ph.py','src/dataset/glt_dual.py',
  'scripts/run_glt_align_2d_unit.py','scripts/run_glt_align_2d.py','scripts/select_glt_align_2d.py','tests/test_glt_align_2d.py']
 text=[]
 for name in changes:
  baseline=ORIGINAL_LOG/'source_before'/name
  before=baseline.read_text() if baseline.exists() else ''
  text.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
 dest=REVIEW/'actual_changes.diff'
 if dest.exists():raise FileExistsError('actual diff exists')
 dest.write_text(''.join(text))
 recovery=[]
 for name in changes:
  baseline=LOG/'source_before_recovery'/name
  before=baseline.read_text()
  recovery.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before_recovery/'+name,tofile='after_recovery/'+name))
 recovery_dest=REVIEW/'recovery_changes.diff'
 if recovery_dest.exists():raise FileExistsError('recovery diff exists')
 recovery_dest.write_text(''.join(recovery))
 print('Captured incremental actual diff against preserved pre-execution source; no hashes.',flush=True)

def handoff():
 if int((LOG/'final/exit.txt').read_text())!=0 or read(RECOVERY/'execution.json')['status']!='待审查':raise ValueError('final engineering test not successful')
 report='\n\n## U1 r4 实际执行交接（待审查）\n\n完整授权已执行：独立投影初始化、融合前raw 2D/3D ALIGN；既有双向multi-positive InfoNCE、可微跨rank gather和全局计数复用，无新编码架构。纯二维实际仅O8/norm2/512维head，二维builder/collate无几何字段；FUSED保留原global语义。四条件全部统一uniform_u1，完整冻结后每best一次正式预测。公共初态逐tensor恢复、同路径head共享、所有包strict-load。\n\n实际预算：\n\n~~~json\n'+json.dumps(read(RECOVERY/'consumption.json'),indent=2)+'\n~~~\n\n唯一最终CPU测试（CUDA_VISIBLE_DEVICES为空、timeout300秒、exit0）：\n\n~~~text\n'+(LOG/'final/stdout.log').read_text().strip()+'\n~~~\n\n'+(REVIEW/'formal_results.md').read_text()+'\n\n工程PASS与性能晋级分开报告，自检不代替独立审查。实际增量diff：results/glt_optimization_20261011/r4/U1/final_review/actual_changes.diff；产物根results/glt_optimization_20261011/r4/U1/，日志根logs/glt_optimization_20261011/r4/U1/。窗口Uni-Poly:glt_opt_r4_U1_controller。PT物理0/1/2/3四rank，完成后FT/预测每卡单单元四路、内部cuda:0。PT parent在pretrain/P_ALIGN/deploy_05000.pt，best在finetune/<arm>/<task>/foldN/best.pt，预测formal/<arm>/<task>/foldN/test_predictions.npz，绑定evaluation_before_test.json；完整配置initialization/<arm>.json。每单元保留command/cwd/environment/start/end/stdout/stderr/真实exit。历史与SOTA未修改、全部未实施路线保留，F_CE与部署I不替换，其他周期未启动。\n'
 report=report.replace(str(OUT/'consumption.json'),str(RECOVERY/'consumption.json'))
 report=report.replace('results/glt_optimization_20261011/r4/U1/final_review/actual_changes.diff','results/glt_optimization_20261011/r4/U1/final_review/recovery1/actual_changes.diff').replace('日志根logs/glt_optimization_20261011/r4/U1/','日志根logs/glt_optimization_20261011/r4/U1/recovery1/').replace('窗口Uni-Poly:glt_opt_r4_U1_controller','窗口Uni-Poly:glt_opt_r4_U1_recovery1_controller')
 report+='\n恢复授权：用户“请继续”；只补齐缺失初始化，原P_ALIGN_step0.pt／配置及初始化失败现场均保留，原PT／FT预算未扩大。恢复实耗在recovery1/consumption.json；原consumption.json为中断快照。恢复增量在final_review/recovery1/recovery_changes.diff。\n'
 (REVIEW/'delivery.md').write_text(report)
 plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('**U1，执行中**','**U1，待审查**').replace('（r4，执行中）','（r4，待审查）');plan.write_text(s+report)

if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif sys.argv[1:]==['--capture-diff']:capture_diff()
 elif not sys.argv[1:]:freeze()
 else:raise ValueError('unexpected arguments')
