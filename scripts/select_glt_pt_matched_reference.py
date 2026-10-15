#!/usr/bin/env python3
"""Freeze all S5 units, compare saved predictions once, and write actual handoff."""
import csv,difflib,json,math,sys
from decimal import Decimal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_glt_pt_matched_reference_unit import OUT,LOG,BASE,REFERENCE,ARMS,TASKS,PARAMETERS,RESOURCE,parent,head_path,ft_folder,read,write,validate_config,validate_best,reference_metadata
HISTORICAL=.7855913245367517
U1=ROOT/'results/glt_optimization_20261011/r4/U1'
def qualifies(value):return Decimal(str(value))-Decimal(str(HISTORICAL))>Decimal('.01')
def check_prediction_identity(left,right,where):
 for key in ('indices','sample_keys','y_true','scaler_mean','scaler_scale'):
  if not np.array_equal(left[key],right[key]):raise ValueError('prediction identity differs '+where+' '+key)
 if 'geometry_valid' in right and not np.array_equal(left['geometry_valid'],right['geometry_valid']):
  raise ValueError('geometry coverage differs '+where)
def metric(archive):
 from sklearn.metrics import r2_score
 truth=np.asarray(archive['y_true']);prediction=np.asarray(archive['y_pred'])
 if truth.ndim!=1 or truth.shape!=prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
  raise FloatingPointError('nonfinite or mismatched saved predictions')
 result=dict(r2=float(r2_score(truth,prediction)),rmse=float(np.sqrt(np.mean((truth-prediction)**2))))
 if not all(math.isfinite(v) for v in result.values()):raise FloatingPointError('nonfinite saved metrics')
 return result
def freeze():
 import torch
 if (OUT/'evaluation_before_test.json').exists():raise FileExistsError('S5 already frozen')
 cfg=validate_config(read(OUT/'initialization/F_CE_2R.json'));units=[]
 old=read(REFERENCE/'evaluation_before_test.json')
 if old['validation_units']!=len(old['units']) or len(old['units'])!=35:raise ValueError('T_FCE reference matrix incomplete')
 for task in TASKS:
  split=read(ROOT/'data/splits/mips_paper5_outer'/f'{task}.json')
  for fold in range(5):
   folder=ft_folder('F_CE_2R',task,fold);m=read(folder/'metrics.json');scaler=read(folder/'scaler.json');entry=split['folds'][fold]
   count=70*((len(entry['train_indices'])+23)//24)
   if m['status']!='PASS' or m['outer_test']!='NOT_RUN' or m['epochs']!=70 or m['updates']!={'attempted':count,'completed':count}:raise ValueError('FT matrix incomplete')
   best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);validate_best(best,'F_CE_2R',task,fold,cfg)
   if best['scaler_mean']!=scaler['mean'] or best['scaler_scale']!=scaler['scale'] or not 1<=best['best_epoch']<=70 or any(not torch.isfinite(v).all() for v in best['state_dict'].values()):raise ValueError('best/scaler/epoch invalid')
   if best['best_epoch']!=min(best['history'],key=lambda row:row['validation_rmse'])['epoch']:raise ValueError('best epoch differs from validation minimum')
   refs=[u for u in old['units'] if (u['task'],u['fold'])==(task,fold)]
   if len(refs)!=1 or refs[0]['test_indices']!=sorted(entry['test_indices']) or refs[0]['scaler']!=scaler:raise ValueError('T_FCE reference binding differs')
   units.append(dict(task=task,fold=fold,best=str(folder/'best.pt'),parent=str(parent('F_CE_2R')),
    best_epoch=best['best_epoch'],head_initial=str(head_path('F_CE_2R',task,fold)),scaler=scaler,
    test_indices=sorted(entry['test_indices']),reference_binding=refs[0],**reference_metadata(cfg)))
 if len(units)!=35:raise ValueError('35 complete units required')
 write(OUT/'evaluation_before_test.json',dict(validation_units=35,units=units,configs={'F_CE_2R':cfg},
  reference_configuration=old['configs']['T_FCE'],reference_root=str(REFERENCE),parent=str(parent('F_CE_2R')),outer_test='NOT_RUN',selection_by_validation=True))
def summarize():
 frozen=read(OUT/'evaluation_before_test.json')
 if frozen['validation_units']!=35 or frozen['reference_configuration']!=read(REFERENCE/'initialization/T_FCE.json'):raise ValueError('frozen reference config differs')
 papers={}
 for line in (ROOT/'SOTA_RESULT.md').read_text().splitlines():
  p=[v.strip() for v in line.split('|')]
  if len(p)==5 and p[1].lower() in TASKS and p[3].startswith('**'):papers[p[1].lower()]=dict(model=p[2],r2=float(p[3].strip('*').split('±')[0]))
 if set(papers)!=set(TASKS):raise ValueError('paper matrix incomplete')
 baseline_tasks={};torsion_tasks={};rows=[]
 for task in TASKS:
  records=[]
  for fold in range(5):
   folder=OUT/'formal/F_CE_2R'/task/f'fold{fold}';m=read(folder/'metrics.json')
   binding=next(x for x in frozen['units'] if x['task']==task and x['fold']==fold)
   if m['status']!='PASS' or m['outer_test']!='RUN' or m['parameters']!=PARAMETERS:raise ValueError('formal unit invalid')
   paths=dict(F_CE_2R=folder/'test_predictions.npz',T_FCE=REFERENCE/f'formal/T_FCE/{task}/fold{fold}/test_predictions.npz',
    U0_FUSED=U1/f'formal/U0_FUSED/{task}/fold{fold}/test_predictions.npz',F_CE=BASE/f'formal/F_CE/{task}/fold{fold}/test_predictions.npz')
   result={}
   with np.load(paths['F_CE_2R'],allow_pickle=False) as current:
    if current['indices'].tolist()!=binding['test_indices'] or int(current['best_epoch'])!=binding['best_epoch']:raise ValueError('formal freeze differs')
    result['F_CE_2R']=metric(current)
    for name in ('T_FCE','U0_FUSED','F_CE'):
     with np.load(paths[name],allow_pickle=False) as ref:
      check_prediction_identity(current,ref,name+' '+task+' '+str(fold));result[name]=metric(ref)
    if np.asarray(current['torsion_available']).any():raise ValueError('no-torsion reference claims torsion inputs')
   validation={}
   vp=ft_folder('F_CE_2R',task,fold)/'validation_predictions.npz'
   with np.load(vp,allow_pickle=False) as current:
    validation['F_CE_2R']=metric(current)
    for name,base in [('T_FCE',REFERENCE/'finetune/T_FCE'),('U0_FUSED',U1/'finetune/U0_FUSED'),('F_CE',BASE/'finetune/F_CE')]:
     with np.load(base/task/f'fold{fold}'/'validation_predictions.npz',allow_pickle=False) as ref:
      check_prediction_identity(current,ref,'validation '+name+' '+task+' '+str(fold));validation[name]=metric(ref)
   row=dict(task=task,fold=fold,r2=result['F_CE_2R']['r2'],rmse=result['F_CE_2R']['rmse'],
    T_FCE_r2=result['T_FCE']['r2'],T_FCE_rmse=result['T_FCE']['rmse'],
    T_minus_F_CE_2R_r2=result['T_FCE']['r2']-result['F_CE_2R']['r2'],T_minus_F_CE_2R_rmse=result['T_FCE']['rmse']-result['F_CE_2R']['rmse'],
    F_CE_2R_minus_U0_FUSED_r2=result['F_CE_2R']['r2']-result['U0_FUSED']['r2'],
    F_CE_2R_minus_F_CE_r2=result['F_CE_2R']['r2']-result['F_CE']['r2'],
    validation_T_minus_F_CE_2R_r2=validation['T_FCE']['r2']-validation['F_CE_2R']['r2'],
    validation_T_minus_F_CE_2R_rmse=validation['T_FCE']['rmse']-validation['F_CE_2R']['rmse'],
    validation=validation,best=binding['best'],predictions=str(paths['F_CE_2R']),reference_predictions={k:str(v) for k,v in paths.items() if k!='F_CE_2R'})
   for k in ('r2','rmse'):
    if abs(row[k]-m[k])>1e-12:raise ValueError('saved formal metric differs '+k)
   rows.append(row);records.append(row)
  keys=('r2','rmse','T_FCE_r2','T_FCE_rmse','T_minus_F_CE_2R_r2','T_minus_F_CE_2R_rmse','F_CE_2R_minus_U0_FUSED_r2','F_CE_2R_minus_F_CE_r2','validation_T_minus_F_CE_2R_r2','validation_T_minus_F_CE_2R_rmse')
  stats={k:dict(mean=float(np.mean([r[k] for r in records])),std=float(np.std([r[k] for r in records],ddof=0))) for k in keys}
  stats.update(folds=records,macro_contribution_T_minus_F_CE_2R=stats['T_minus_F_CE_2R_r2']['mean']/7,
   paper=dict(papers[task],delta=stats['r2']['mean']-papers[task]['r2']),
   positive_test_folds=sum(r['T_minus_F_CE_2R_r2']>0 for r in records),
   positive_validation_folds=sum(r['validation_T_minus_F_CE_2R_r2']>0 for r in records))
  baseline_tasks[task]=stats
  torsion_tasks[task]=dict(r2=stats['T_FCE_r2'],rmse=stats['T_FCE_rmse'],paper=dict(papers[task],delta=stats['T_FCE_r2']['mean']-papers[task]['r2']))
 macro=float(np.mean([baseline_tasks[t]['r2']['mean'] for t in TASKS]));t_macro=float(np.mean([torsion_tasks[t]['r2']['mean'] for t in TASKS]))
 if abs(t_macro-read(REFERENCE/'final_review/seven_task_results.json')['arms']['T_FCE']['macro_test_r2'])>1e-12:raise ValueError('T_FCE saved reference macro differs')
 write(OUT/'final_review/seven_task_results.json',dict(arms={'F_CE_2R':dict(tasks=baseline_tasks,macro_test_r2=macro,qualifies=qualifies(macro),parameters=PARAMETERS,configuration=frozen['configs']['F_CE_2R']),
  'T_FCE_reused':dict(tasks=torsion_tasks,macro_test_r2=t_macro,qualifies=qualifies(t_macro),parameters=PARAMETERS+33473,configuration=frozen['reference_configuration'])},
  matched_macro_T_minus_F_CE_2R=t_macro-macro,historical_reference_macro=HISTORICAL,
  consumption=read(OUT/'consumption.json'),consumption_snapshot_stage='before_final_cpu_test',
  protocol='seven tasks, five folds seed42+fold, std ddof0; matched two-rank PT and uniform_u1 FT; added branch capacity remains; shared TEST development results, paper numeric comparison only'))
 with (OUT/'final_review/formal_fold_details.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# S5 两卡匹配参考正式结果（待独立审查）','',
  f'F_CE_2R宏R² {macro:.3f}，复用T_FCE {t_macro:.3f}；T_FCE−F_CE_2R {t_macro-macro:+.3f}。',
  '', '|任务|F_CE_2R R² mean±std|T_FCE R² mean±std|T−参考 R²|参考原单位RMSE|T原单位RMSE|论文最佳模型|论文R²|参考−论文|T−论文|',
  '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
 for task in TASKS:
  x=baseline_tasks[task];t=torsion_tasks[task]
  lines.append(f"|{task}|{x['r2']['mean']:.3f}±{x['r2']['std']:.3f}|{t['r2']['mean']:.3f}±{t['r2']['std']:.3f}|{x['T_minus_F_CE_2R_r2']['mean']:+.3f}|{x['rmse']['mean']:.3f}±{x['rmse']['std']:.3f}|{t['rmse']['mean']:.3f}±{t['rmse']['std']:.3f}|{x['paper']['model']}|{x['paper']['r2']:.3f}|{x['paper']['delta']:+.3f}|{t['paper']['delta']:+.3f}|")
 lines+=['','两条件保持两rank、microbatch84、accumulation6、global1008、原公共step0和PT heads、数据/样本流/seed/目标及统一FT条件。T_FCE复用S4保存产物，参考为本轮新训练；差值描述torsion分支整体效果，仍包含33473参数容量差，不称纯几何信息净收益。',
  '', '所有任务统一train-only StandardScaler及uniform_u1；共享TEST已用于开发，非独立盲测，不按TEST追加实验。论文数据/split/统计协议等价性未完全核实，仅数值对照。R²差越大越好，RMSE差越小越好；统计和门槛使用未舍入值。',
  '', '逐fold/task validation配对差、各任务宏贡献及预测路径见seven_task_results.json／formal_fold_details.csv。F_CE_2R是匹配研究参考，不自动替换历史F_CE或部署I；本周期不实施E0、E1或任何新模型候选。']
 (OUT/'final_review/formal_results.md').write_text('\n'.join(lines)+'\n')
def capture_diff():
 names=['CURRENT_PLAN.md','scripts/pretrain_glt_dual.py','scripts/evaluate_glt_pt_endpoint.py','scripts/run_glt_pt_matched_reference_unit.py','scripts/run_glt_pt_matched_reference.py','scripts/select_glt_pt_matched_reference.py','tests/test_glt_pt_matched_reference.py']
 chunks=[]
 for name in names:
  p=LOG/'source_before'/name;before=p.read_text() if p.exists() else ''
  chunks.extend(difflib.unified_diff(before.splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
 destination=OUT/'final_review/actual_changes.diff';destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('x') as f:f.write(''.join(chunks))
def handoff():
 if int((LOG/'final/exit.txt').read_text())!=0 or read(OUT/'execution.json')['status']!='待审查':raise ValueError('final CPU test not successful')
 result=read(OUT/'final_review/seven_task_results.json');result['consumption']=read(OUT/'consumption.json');result['consumption_snapshot_stage']='final_after_cpu_test'
 write(OUT/'final_review/seven_task_results.json',result)
 report='\n\n## S5 r9 实际执行交接（待审查）\n\n完整两rank无torsion参考PT、35FT、完整冻结、35正式预测及唯一CPU测试完成；T_FCE仅复用保存结果，未重训或重新预测。工程自检PASS，独立审查未执行，不自动替换研究基线或部署。\n\n实耗：\n\n~~~json\n'+json.dumps(read(OUT/'consumption.json'),indent=2)+'\n~~~\n\n唯一CPU测试（空CUDA_VISIBLE_DEVICES，timeout300秒，真实exit0）：\n\n~~~text\n'+(LOG/'final/stdout.log').read_text().strip()+'\n~~~\n\n'+(OUT/'final_review/formal_results.md').read_text()+'\n\n产物results/glt_optimization_20261011/r9/S5/；日志logs/glt_optimization_20261011/r9/S5/；控制器Uni-Poly:glt_opt_r9_S5_gpu23_controller。PT物理GPU2、3两rank；FT及预测单卡两路、内部cuda:0。parent pretrain/F_CE_2R/deploy_05000.pt，best finetune/F_CE_2R/<task>/foldN/best.pt，预测formal/F_CE_2R/<task>/foldN/test_predictions.npz，配置initialization/F_CE_2R.json，冻结evaluation_before_test.json，实际diff final_review/actual_changes.diff。\n'
 (OUT/'final_review/delivery.md').write_text(report)
 p=ROOT/'CURRENT_PLAN.md';s=p.read_text().replace('**S5，执行中**','**S5，待审查**');p.write_text(s+report)
if __name__=='__main__':
 if sys.argv[1:]==['--summarize']:summarize()
 elif sys.argv[1:]==['--handoff']:handoff()
 elif sys.argv[1:]==['--capture-diff']:capture_diff()
 elif not sys.argv[1:]:freeze()
 else:raise ValueError('invalid S5 selection arguments')

