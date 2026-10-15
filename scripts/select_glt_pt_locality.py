#!/usr/bin/env python3
"""Freeze all FT bests, summarize each saved formal prediction exactly once."""
import csv,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_pt_locality_unit import OUT,BASE,parent,connection_config,ARMS,TASKS,read,write,ft_folder,epochs,validate_config
LOG=ROOT/'logs/glt_optimization_20261011/r3/S3'
REFERENCE=.7855913245367517

def macro_r2(tasks):
 if set(tasks)!=set(TASKS):raise ValueError('complete seven-task matrix required')
 values=[np.asarray(tasks[t],dtype=float) for t in TASKS]
 if any(v.shape!=(5,) or not np.isfinite(v).all() for v in values):raise ValueError('five finite folds required')
 return float(np.mean([v.mean() for v in values]))

def qualifies(macro):
 if not math.isfinite(macro):raise ValueError('nonfinite macro')
 return Decimal(str(macro))-Decimal(str(REFERENCE))>Decimal('.01')

def conditional_effects(b,g,l,a):
 return dict(M_G_minus_F_CE=g-b,M_L_minus_F_CE=l-b,M_A_minus_F_CE=a-b,M_L_minus_M_G=l-g,M_A_minus_M_L=a-l)

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
    if not 1<=best['best_epoch']<=epochs(t) or any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('best epoch/tensor invalid')
    units.append(dict(arm=a,task=t,fold=f,best=str(folder/'best.pt'),parent=str(parent(a)),best_epoch=best['best_epoch'],**connection_config(a),scaler=scaler,test_indices=sorted(entry['test_indices'])))
 if len(units)!=105:raise ValueError('incomplete frozen matrix')
 write(OUT/'evaluation_before_test.json',dict(validation_units=105,units=units,configs=configs,parents={a:str(parent(a)) for a in ARMS},parent_fusion_gate='none',reference_macro=REFERENCE,outer_test='NOT_RUN',selection_by_validation=True))

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
  macro=macro_r2(values);details[a]=dict(parameters=read(OUT/'formal'/a/'eea'/'fold0'/'metrics.json')['parameters'],pt_parameters=next(x['pt_parameters'] for x in read(OUT/'initialization'/'summary.json')['arms'] if x['arm']==a),configuration=frozen['configs'][a],parent=str(parent(a)),tasks=tasks,macro_test_r2=macro,macro_delta=macro-REFERENCE,qualifies=qualifies(macro))
 effects={}
 for t in TASKS:
  folds=[conditional_effects(details['M_G']['tasks'][t]['folds'][f]['reference_r2'],*[details[a]['tasks'][t]['folds'][f]['r2'] for a in ARMS]) for f in range(5)]
  effects[t]=dict(folds=folds,mean={k:float(np.mean([v[k] for v in folds])) for k in folds[0]},std={k:float(np.std([v[k] for v in folds],ddof=0)) for k in folds[0]})
 macro_effects=conditional_effects(REFERENCE,*[details[a]['macro_test_r2'] for a in ARMS])
 passed=[a for a in ARMS if details[a]['qualifies']];proposed=max(passed,key=lambda a:details[a]['macro_test_r2']) if passed else 'F_CE'
 write(OUT/'final_review/seven_task_results.json',dict(arms=details,reference_macro=REFERENCE,proposed_baseline=proposed,conditional_effects=effects,macro_effects=macro_effects,baseline='F_CE unchanged',deployment='I unchanged',protocol='seven task five-fold mean, seed42+fold, std ddof0; shared TEST used for development selection, numeric paper comparison only'))
 with (OUT/'final_review/formal_fold_details.csv').open('x',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# GLT-OPTIMIZATION S3 正式结果（待独立审查）','','F_CE参考宏值0.7855913245367517，唯一门槛为原精度宏差严格>0.01。seed42+fold；每task五折mean/std，std ddof0，再七task等权宏平均。TEST已用于开发选择，非独立盲测。','','|候选|宏R²|相对F_CE|通过|','|---|---:|---:|---|']
 for a in ARMS:lines.append(f"|{a}|{details[a]['macro_test_r2']:.9f}|{details[a]['macro_delta']:+.9f}|{details[a]['qualifies']}|")
 lines+=['','建议基线：'+proposed+'；不自动替换F_CE或部署I。','','|任务|历史F_CE|M_G|M_L|M_A|','|---|---:|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+f"{baseline[t]['r2']['mean']:.9f}±{baseline[t]['r2']['std']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['r2']['mean']:.9f}±{details[a]['tasks'][t]['r2']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|任务|论文最佳模型|论文R²|M_G−论文|M_L−论文|M_A−论文|','|---|---|---:|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+papers[t]['model']+f"|{papers[t]['r2']:.9f}|"+'|'.join(f"{details[a]['tasks'][t]['paper']['delta']:+.9f}" for a in ARMS)+'|')
 lines+=['','|任务|M_G 原单位RMSE|M_L 原单位RMSE|M_A 原单位RMSE|','|---|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+'|'.join(f"{details[a]['tasks'][t]['rmse']['mean']:.9f}±{details[a]['tasks'][t]['rmse']['std']:.9f}" for a in ARMS)+'|')
 lines+=['','|条件差|宏差|','|---|---:|']
 for k,v in macro_effects.items():lines.append(f'|{k}|{v:+.9f}|')
 lines+=['','|任务|M_G−F_CE|M_L−F_CE|M_A−F_CE|M_L−M_G|M_A−M_L|','|---|---:|---:|---:|---:|---:|']
 for t in TASKS:lines.append('|'+t+'|'+'|'.join(f'{v:+.9f}' for v in effects[t]['mean'].values())+'|')
 lines+=['','参数量：'+str({a:dict(ft=details[a]['parameters'],pt=details[a]['pt_parameters']) for a in ARMS})]
 lines+=['','完整逐fold配对差、五个条件差的task/fold/宏值在seven_task_results.json和formal_fold_details.csv。R²差正值较高，RMSE差负值较好。M_G是本轮匹配两rank机制对照；历史F_CE四rank仅为性能数值参考。M_A增加有效可学习端点路径，M_A−M_L不能解释为单个原始特征的独立收益。论文与项目数据规模、Egc/Xc版本、标签处理和精确split等价性未核实，仅作数值对照，不宣称公平超过论文。EAT/Eib/Tg不纳入七任务。']
 (OUT/'final_review/formal_results.md').write_text('\n'.join(lines)+'\n')

def capture_diff():
 import difflib,subprocess
 changes=['src/modules/glt_dual.py','src/modules/glt_dual_opt.py','src/modules/glt_dual_pretrain.py','scripts/pretrain_glt_dual.py','CURRENT_PLAN.md','scripts/run_glt_pt_locality_unit.py',
  'scripts/run_glt_pt_locality.py','scripts/select_glt_pt_locality.py','tests/test_glt_pt_locality.py']
 text=[]
 for name in changes:
  baseline=LOG/'source_before'/name
  before=baseline.read_text() if baseline.exists() else ''
  text.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
 dest=OUT/'final_review/actual_changes.diff'
 if dest.exists():raise FileExistsError('actual diff exists')
 dest.write_text(''.join(text))
 print('Captured incremental actual diff against preserved pre-execution source; no hashes.',flush=True)

def handoff():
 if int((LOG/'final/exit.txt').read_text())!=0 or read(OUT/'execution.json')['status']!='待审查':raise ValueError('final engineering test not successful')
 report='\n\n## S3 r3 实际执行交接（待审查）\n\n授权最小实现及完整实验已执行：只修改共享模型端点路径与必要PT配置传递、本文，新增授权三个入口和一个测试文件；既有训练循环、scaler、optimizer、head初态和评估逻辑复用。M_G使用global，M_L与M_A仅真实入射物理键；所有阶段保持规则一致。三个arm从公共step0重新两rank PT，端点MLP同名同初态，M_G/M_L输入零，M_A使用mask后O8 layers前端点嵌入均值，保留完整mask与物理副本。新增524288个端点参数，无新增buffer／gate／dropout；公共参数正常初始化逐tensor核对。空入射／无效几何保留fallback，非候选不产生output bias更新；新配置缺字段／非法模式或β拒绝恢复。\n\n实际预算：\n\n~~~json\n'+json.dumps(read(OUT/'consumption.json'),indent=2)+'\n~~~\n\n唯一最终CPU测试（空CUDA_VISIBLE_DEVICES，timeout300秒，exit0）：\n\n~~~text\n'+(LOG/'final/stdout.log').read_text().strip()+'\n~~~\n\n'+(OUT/'final_review/formal_results.md').read_text()+'\n\n实际增量diff：results/glt_optimization_20261011/r3/S3/final_review/actual_changes.diff（相对本轮修改前source，含本交接追加）。窗口Uni-Poly:glt_opt_r3_S3_controller；PT依次M_G→M_L→M_A，物理GPU0/1两rank，全部PT结束再FT；FT与正式预测物理GPU0/1/2/3，每卡最多一个单元、总并发≤4，内部cuda:0。日志logs/glt_optimization_20261011/r3/S3/，产物results/glt_optimization_20261011/r3/S3/；checkpoint位于finetune/<arm>/<task>/foldN/best.pt，预测位于formal/<arm>/<task>/foldN/test_predictions.npz，冻结绑定evaluation_before_test.json。每单元保留command/cwd/environment/start/end/stdout/stderr/真实exit。工程测试通过与性能晋级分别报告，自检不代替独立审查；未归档PROJECT_HISTORY，未替换F_CE或部署I，未启动其他周期。\n'
 (OUT/'final_review/delivery.md').write_text(report)
 plan=ROOT/'CURRENT_PLAN.md';s=plan.read_text().replace('**S3，执行中**','**S3，待审查**').replace('r3 当前周期；执行中','r3 当前周期；待审查');plan.write_text(s+report)

if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif sys.argv[1:]==['--capture-diff']:capture_diff()
 elif not sys.argv[1:]:freeze()
 else:raise ValueError('unexpected arguments')
