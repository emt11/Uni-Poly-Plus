"""Bounded validation gates, global freeze, zero-update prediction, actual-result self-check."""
import sys,argparse,json,time,math,subprocess,shlex
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.gp_sota_contract import *
from scripts.run_mcl_ph_r4_fast import write_shell,start_window,atomic_json,timestamp
from scripts.run_gp_development_stage import reference,rmse,audit
from src.training.glt_dual_runtime import require_tmux,save_checkpoint,load_sample_index_artifact
from src.training.mcl_ph_direct_contract import assert_actual_equal
from scripts.check_gp_pretrain_package import finite

PRIORITY=('ei','eps','nc','xc')

def gate(task,values):
 a=np.asarray(values,dtype=float);assert a.shape==(2,) and np.isfinite(a).all()
 return dict(passed=bool(a.mean()>=(.02 if task=='xc' else .01) and (a>0).all()),mean_gain=float(a.mean()),positive=int((a>0).sum()),gains=a.tolist())

def run(name,command,gpu=None,*,attempt=1):
 log=LOG/f'{name}_{attempt:02d}';write_shell(log,command,gpu=gpu);start_window('gp_'+name,log)
 while not (log/'exit.code').exists():time.sleep(10)
 code=int((log/'exit.code').read_text());assert code==0,f'{name} failed exit {code}: {log}'
 return log

def package(route):return OUT/f'pretrain/{route}/attempt01/deploy_05000.pt'

def development_scope(name,units):
 return dict(plan=PLAN,revision=REV,name=name,units=units,allowed_units=[f"{u['recipe']}/{u['route']}/{u['task']}/fold{u['fold']}/seed{42+u['fold']}" for u in units],packages={r:str(package(r)) for r in {u['route'] for u in units}})

def stage(name,units,*,resume_startup=False):
 scope=development_scope(name,units);path=OUT/'scopes'/f'{name}.json'
 if resume_startup:
  assert name=='P3a' and not (OUT/'development'/name).exists(), 'startup recovery cannot reuse training outputs'
  assert_actual_equal(read(path),scope,'P3a.existing_scope')
 else:
  path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();atomic_json(path,scope)
 run(name,['python','scripts/run_gp_development_stage.py','--scope',str(path)],attempt=2 if resume_startup else 1)
 return read(OUT/'development'/name/'report.json')['rows']

def units(route,tasks,folds,recipe='F0'):return [dict(route=route,task=t,fold=f,recipe=recipe) for t in tasks for f in folds]

def taskrows(rows,t):return sorted([r for r in rows if r['unit']['task']==t],key=lambda r:r['unit']['fold'])

def initial_gates(rows,tasks):
 return {t:{ref:gate(t,[r[ref+'_relative_gain'] for r in taskrows(rows,t)]) for ref in ('glt','incumbent')} for t in tasks}

def passed(g,t):return all(v['passed'] for v in g[t].values())

def pretrain(route):
 folder=package(route).parent
 run('p2b_'+route,['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_mcl_ph.py','--route',route,'--output',str(folder),'--stop','5000'])
 run('p2b_'+route+'_acceptance',['python','scripts/check_gp_pretrain_package.py','--folder',str(folder)],gpu=0)


def check_p3a_startup_recovery():
 """Read-only check for this pre-launch failure; no model or diagnostic rerun."""
 scope=OUT/'scopes/P3a.json';stageout=OUT/'development/P3a';folder=package('GP').parent
 assert not stageout.exists(), 'P3a outputs exist; partial training recovery requires separate review'
 assert_actual_equal(read(scope),development_scope('P3a',units('GP',PRIORITY,(0,1))),'P3a.existing_scope')
 assert sorted(p.name for p in LOG.glob('P3a_*'))==['P3a_01'], 'P3a child/retry logs already exist'
 failed=LOG/'P3a_01'
 assert int((failed/'exit.code').read_text())==1
 assert shlex.split((failed/'command.txt').read_text())==['python','scripts/run_gp_development_stage.py','--scope',str(scope)]
 error=(failed/'stderr.log').read_text()
 assert 'FileNotFoundError' in error and str(stageout) in error, 'not the known parent-directory failure'
 for p in OUT.glob('**/budget.json'):
  b=read(p)
  assert not (b.get('kind')=='finetune' and b.get('stage')!='smoke'), 'complete FT ledger exists'
 for name,command in [('p2a_GP_01',['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_mcl_ph.py','--route','GP','--output',str(folder),'--stop','5000']),('p2a_GP_acceptance_01',['python','scripts/check_gp_pretrain_package.py','--folder',str(folder)])]:
  log=LOG/name;assert int((log/'exit.code').read_text())==0
  assert shlex.split((log/'command.txt').read_text())==command, name+'.command'
 acceptance=read(folder/'acceptance.json');runtime=read(folder/'runtime.json');budget=read(folder/'budget.json');finite(acceptance)
 for k,v in dict(status='PASS',route='GP',step=5000,evaluated_rows=1024,evaluation_calls_per_package=1,optimizer_updates=0,complete_encoder_exact=True,step0_205_exact=True,resume_finite=True,validation_total_loss_candidate_selection=False).items():
  assert acceptance.get(k)==v, 'GP.acceptance.'+k
 indices=load_sample_index_artifact(INDEX,'fixed_validation')['indices'].tolist()
 assert len(indices)==1024 and acceptance['source_indices']==indices, 'GP fixed validation indices differ'
 assert runtime['status']=='PASS' and runtime['step']==runtime['executed_updates']==5000 and runtime['start_step']==0 and runtime['world']==4
 assert runtime['recipe']==pretrain_recipe('GP')
 assert budget['attempted_updates']==budget['completed_updates']==budget['total_step']==5000 and budget['smoke'] is False
 assert (folder/'validation.writer').is_file() and not (folder/'failure.json').exists()
 saved=torch.load(folder/'resume_05000.pt',map_location='cpu',weights_only=False);deploy=torch.load(package('GP'),map_location='cpu',weights_only=False)
 finite(saved);finite(deploy)
 assert saved['schema']=='gp-sota-resume-v2' and saved['step']==5000 and saved['next_position']==5040000 and len(saved['rng'])==4
 assert saved['contract']['recipe']==pretrain_recipe('GP') and saved['contract']['smoke'] is False
 assert deploy['schema']=='gp-sota-deploy-v2' and deploy['route']=='GP' and deploy['step']==5000
 assert_actual_equal({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},deploy['state_dict'],'GP.resume_deploy')
 return dict(status='PASS_P3A_STARTUP_RECOVERY',scope=str(scope),reused_acceptance=str(folder/'acceptance.json'),reused_package=str(package('GP')),next_stage_log=str(LOG/'P3a_02'),model_calls=0,optimizer_updates=0,new_validation_rows=0,independent_scientific_review=False)


def summarize(selection,formal_paths,reason):
 old=read(ROOT/'results/mcl_ph_r6_20261004/seven_task_results.json');rows=[]
 for oldrow in old['rows']:
  task=oldrow['task'];new=task in formal_paths;paths=formal_paths.get(task,oldrow['prediction_paths']);r2s=[];rmses=[]
  assert len(paths)==5
  for fold,p in enumerate(paths):
   with np.load(p,allow_pickle=False) as z:
    y,zp=z['y_true'].astype(float).reshape(-1),z['y_pred'].astype(float).reshape(-1)
    assert np.isfinite(y).all() and np.isfinite(zp).all();r2s.append(float(1-np.sum((y-zp)**2)/np.sum((y-y.mean())**2)));rmses.append(float(np.sqrt(np.mean((y-zp)**2))))
    if new:
     assert int(z['fold'])==fold and int(z['seed'])==42+fold
     entry=ft_recipe(task,fold,selection[task]['route'],selection[task]['recipe'])['split']['folds'][fold];assert z['test_indices'].tolist()==sorted(entry['test_indices'])
   if not new:assert abs(r2s[-1]-oldrow['selected_fold_r2'][fold])<1e-12
  labels=pd.read_csv(ROOT/f'data/raw/smi_{task}.csv').iloc[:,1].to_numpy(dtype=float)
  for fold,p in enumerate(paths):
   with np.load(p,allow_pickle=False) as z:
    ids=z['test_indices'].astype(int);np.testing.assert_allclose(z['y_true'].reshape(-1),labels[ids],rtol=2e-6,atol=2e-6)
   if new:
    m=read(Path(p).parent/'metrics.json');assert abs(m['test_r2']-r2s[fold])<1e-9 and abs(m['test_rmse']-rmses[fold])<1e-9
    assert m['promotion_optimizer_updates']==0
    dev=Path(m['reused_development_unit']);a=torch.load(dev/'best.pt',map_location='cpu',weights_only=False);b=torch.load(Path(p).parent/'best.pt',map_location='cpu',weights_only=False)
    assert set(a['state_dict'])==set(b['state_dict'])
    for k,v in a['state_dict'].items():assert torch.equal(v,b['state_dict'][k]),k
    assert (OUT/'frozen_selection.json').stat().st_mtime_ns<Path(p).stat().st_mtime_ns
  mean=float(np.mean(r2s));rows.append(dict(task=task,result_kind='new_GP_formal' if new else 'reused_accepted_incumbent',new_gp_status='完成，待审查' if new else '未完成／不可比较：'+reason,selected_route=selection.get(task,{}),prediction_paths=paths,fold_r2=r2s,fold_rmse=rmses,mean_r2=mean,std_r2=float(np.std(r2s)),paper_model=oldrow['paper_model'],paper_best_r2=oldrow['paper_best_r2'],project_minus_paper=mean-oldrow['paper_best_r2']))
 report=dict(status='SELF_CHECK_PASS_PENDING_INDEPENDENT_REVIEW',plan=PLAN,revision=REV,completed_at=timestamp(),rows=rows,fold_seed_protocol='paper5_outer; arithmetic five-fold mean, seed42+fold, std ddof0',paper_protocol_equivalence='未核实，仅数值对照',independent_blind_test=False,project_history_archived=False,stop_reason=reason)
 path=OUT/'seven_task_results_r2.json';assert not path.exists();atomic_json(path,report);return report


def main():
 p=argparse.ArgumentParser();p.add_argument('--from-gp',action='store_true')
 recovery=p.add_mutually_exclusive_group();recovery.add_argument('--resume-p3a-startup',action='store_true',help='reuse completed GP acceptance and unchanged P3a scope; retry only the pre-launch directory failure')
 recovery.add_argument('--check-p3a-startup',action='store_true',help='read-only CPU recovery preflight; launches no training or diagnostic')
 args=p.parse_args();require_tmux();torch.set_num_threads(1)
 if args.resume_p3a_startup or args.check_p3a_startup:
  print(json.dumps(check_p3a_startup_recovery(),ensure_ascii=False),flush=True)
  if args.check_p3a_startup:return
 folder=package('GP').parent;log=LOG/'p2a_GP_01'
 while not (log/'exit.code').exists():time.sleep(10)
 if int((log/'exit.code').read_text())!=0:raise RuntimeError('P2a stopped; preserve and audit outside this continuation')
 if not args.resume_p3a_startup:run('p2a_GP_acceptance',['python','scripts/check_gp_pretrain_package.py','--folder',str(folder)],gpu=0)
 gp=stage('P3a',units('GP',PRIORITY,(0,1)),resume_startup=args.resume_p3a_startup);gg=initial_gates(gp,PRIORITY)
 atomic_json(OUT/'p3a_gates.json',dict(status='PASS' if any(passed(gg,t) for t in PRIORITY) else 'STOP_ALL_TASK_GATES_FAILED',gates=gg,reference='historical GLT; Xc r5 standard',historical_fp_step0_exact='未核实'))
 if not any(passed(gg,t) for t in PRIORITY):summarize({}, {},'P3a四优先任务初筛门槛均失败');return
 pretrain('GU');pretrain('GP_A')
 gu=stage('P3b_GU',units('GU',PRIORITY,(0,1)));ga=stage('P3b_GP_A',units('GP_A',PRIORITY,(0,1)));ag=initial_gates(ga,PRIORITY)
 choices={};dev={}
 for t in PRIORITY:
  eligible=[route for route,g in [('GP',gg),('GP_A',ag)] if passed(g,t)]
  if not eligible:continue
  selected='GP'
  if 'GP' not in eligible:selected='GP_A'
  elif 'GP_A' in eligible:
   gm=np.mean([r['best_validation_rmse'] for r in taskrows(gp,t)]);am=np.mean([r['best_validation_rmse'] for r in taskrows(ga,t)])
   if (gm-am)/gm>.002:selected='GP_A'
  choices[t]=dict(route=selected,recipe='F0');dev[t]=taskrows(gp if selected=='GP' else ga,t)
 macro_gp=np.mean([gg[t]['glt']['mean_gain'] for t in PRIORITY]);macro_a=np.mean([ag[t]['glt']['mean_gain'] for t in PRIORITY]);globalroute='GP_A' if any(passed(ag,t) for t in PRIORITY) and macro_a-macro_gp>=.01 else 'GP'
 atomic_json(OUT/'p3b_choice.json',dict(status='LOCKED',choices=choices,GP_A_gates=ag,GP_macro_gain=macro_gp,GP_A_macro_gain=macro_a,global_route_for_remaining=globalroute,GU_diagnostic_rows=gu,GU_deploy_eligible=False))
 small=('eea','egb','egc');smallrows=stage('P3c',units(globalroute,small,(0,1)));smallgates=initial_gates(smallrows,small)
 for t in small:
  if passed(smallgates,t):choices[t]=dict(route=globalroute,recipe='F0');dev[t]=taskrows(smallrows,t)
 atomic_json(OUT/'p3c_gates.json',smallgates)
 f1units=[u for t in PRIORITY if t in choices for u in units(choices[t]['route'],(t,),(0,1),'F1')]
 if f1units:
  f1=stage('P4',f1units);fg={}
  for t in PRIORITY:
   if t not in choices:continue
   new=taskrows(f1,t);gains=[(old['best_validation_rmse']-cur['best_validation_rmse'])/old['best_validation_rmse'] for old,cur in zip(dev[t],new)];fg[t]=gate(t,gains)
   if fg[t]['passed']:choices[t]['recipe']='F1';dev[t]=new
  atomic_json(OUT/'p4_gates.json',dict(status='LOCKED',gates=fg,choices=choices))
 if not choices:summarize({}, {},'全部候选初筛失败');return
 confirmationunits=[u for t in TASKS if t in choices for u in units(choices[t]['route'],(t,),(2,3,4),choices[t]['recipe'])]
 atomic_json(OUT/'p5_locked_candidates.json',dict(status='LOCKED',choices=choices,locked_at=timestamp(),selection_uses='fold0/1 validation only'))
 confirmation=stage('P5',confirmationunits);final={};cg={}
 for t,c in choices.items():
  extra=taskrows(confirmation,t);allrows=dev[t]+extra;checks={}
  for ref in ('glt','incumbent'):
   q=np.array([r[ref+'_relative_gain'] for r in extra]);five=np.array([r[ref+'_relative_gain'] for r in allrows]);threshold=.02 if t=='xc' else .01
   checks[ref]=dict(passed=bool(q.mean()>=threshold and (q>0).sum()>=2 and q.min()>=-.02 and five.mean()>=threshold),confirmation_gains=q.tolist(),five_fold_gains=five.tolist(),mean_confirmation_gain=float(q.mean()),mean_five_fold_gain=float(five.mean()))
  cg[t]=checks
  if all(v['passed'] for v in checks.values()):final[t]=dict(c,rows=allrows)
 atomic_json(OUT/'p5_gates.json',dict(status='LOCKED',gates=cg,accepted=list(final)))
 # Contract is fixed before any new downstream test reader is invoked.
 freeze=dict(plan=PLAN,revision=REV,status='FROZEN',frozen_at=timestamp(),allowed_units=[],unit_contracts={},seven_task_routes={},historical_fp_step0_exact='未核实')
 for t in TASKS:
  if t not in final:freeze['seven_task_routes'][t]=dict(kind='reused_incumbent');continue
  route,recipe=final[t]['route'],final[t]['recipe'];freeze['seven_task_routes'][t]=dict(kind='new_candidate',route=route,recipe=recipe,package=str(package(route)),five_development_bests=[])
  for row in final[t]['rows']:
   fold=row['unit']['fold'];old=Path(row['folder']);m=read(old/'metrics.json');key=f'{recipe}/{route}/{t}/fold{fold}/seed{42+fold}'
   saved=torch.load(old/'best.pt',map_location='cpu',weights_only=False)
   weights=OUT/'frozen_weights'/t/f'fold{fold}.pt';weights.parent.mkdir(parents=True,exist_ok=True);assert not weights.exists()
   save_checkpoint(weights,dict(state_dict=saved['state_dict'],head_initial=saved['head_initial'],contract=saved['contract']))
   freeze['allowed_units'].append(key);freeze['unit_contracts'][key]=dict(contract=m['contract'],development_unit=str(old),best_epoch=m['best_epoch'],scaler_mean=saved['scaler_mean'],scaler_scale=saved['scaler_scale'],scaler_train_raw=saved['scaler_train_raw'],frozen_weights=str(weights));freeze['seven_task_routes'][t]['five_development_bests'].append(str(old/'best.pt'))
 oldrows=read(ROOT/'results/mcl_ph_r6_20261004/seven_task_results.json')['rows']
 for row in oldrows:
  if row['task'] not in final:freeze['seven_task_routes'][row['task']]['prediction_paths']=row['prediction_paths']
 assert len(freeze['allowed_units'])<=35
 frozen=OUT/'frozen_selection.json';assert not frozen.exists();atomic_json(frozen,freeze)
 formal={}
 for t,c in final.items():
  formal[t]=[]
  for row in c['rows']:
   fold=row['unit']['fold'];folder=OUT/'formal'/c['route']/c['recipe']/t/f'fold{fold}';run(f'P6_{t}_f{fold}',['python','scripts/finetune_glt_mcl_ph.py','--task',t,'--fold',str(fold),'--route',c['route'],'--recipe',c['recipe'],'--stage','formal','--scope',str(frozen),'--checkpoint',str(package(c['route'])),'--reuse',row['folder'],'--output',str(folder)],gpu=fold%4)
   m=read(folder/'metrics.json');assert m['promotion_optimizer_updates']==0 and m['frozen_selection']==str(frozen.resolve());formal[t].append(str(folder/'test_predictions.npz'))
 summarize({t:{k:v for k,v in c.items() if k!='rows'} for t,c in final.items()},formal,'该任务validation确认未通过，保留旧路线')
if __name__=='__main__':main()
