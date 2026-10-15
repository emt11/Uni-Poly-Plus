#!/usr/bin/env python3
"""Fixed GLT-DUAL-OPT r1 matrix. No smoke, hashes, retries or interim tests."""
import argparse,datetime,json,math,os,shlex,subprocess,sys,time,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_dual_opt_20261006/r1';LOG=ROOT/'logs/glt_dual_opt_20261006/r1'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PARENT=ROOT/'results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/deploy_05000.pt'
COMMON=ROOT/'results/mcl_ph_20260921/shared_pretrain/common_init_v1.pt'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
ARMS=('R_INCIDENT','B_FP','A_ALIGN','F_TOKEN','M_TOKEN_CE','G_SEAM_GEO')
PLAN='GLT-DUAL-OPT-20261006-01'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def reference(task,fold,incumbent=False):
 if task=='xc':return ROOT/f'results/mcl_ph_r5_20261004/paper5_outer/xc_standard/glt_ref/xc/fold{fold}'
 route='m_xattn' if incumbent and task=='eea' else 'glt_ref'
 return ROOT/f'results/mcl_ph_r4_20261002/fast_campaign_04/paper5_outer/{route}/{task}/fold{fold}'
def epochs(task):return 60 if task=='egc' else 70
def modes(arm):return dict(fusion_mode='token' if arm in ('F_TOKEN','M_TOKEN_CE') else 'concat',readout='incident' if arm=='R_INCIDENT' else 'internal',third_task='align' if arm=='A_ALIGN' else 'none',geometry_target='seam' if arm=='G_SEAM_GEO' else 'internal',element_mask='all_copies_unknown' if arm in ('F_TOKEN','M_TOKEN_CE') else 'legacy',opt_mode=arm)
def initialize():
 import torch
 from src.utils import set_global_seed
 from src.modules.glt_dual_pretrain import DualPretrainer
 from src.training.glt_dual_runtime import apply_common_initialization,save_checkpoint
 OUT.mkdir(parents=True,exist_ok=False)
 set_global_seed(42);base=DualPretrainer('concat',geometry_head_norm=True)
 info=apply_common_initialization(base,COMMON,direct=True)
 shared={k:v.detach().clone() for k,v in base.state_dict().items()}
 units=[]
 for arm in ARMS:
  for task in TASKS:
   entry=read(SPLIT/f'{task}.json')
   for fold in range(5):
    f=entry['folds'][fold];updates=epochs(task)*math.ceil(len(f['train_indices'])/24)
    units.append(dict(arm=arm,task=task,fold=fold,epochs=epochs(task),updates=updates,status='NOT_RUN'))
 if sum(u['updates'] for u in units)!=308700 or sum(u['epochs'] for u in units)!=14400:raise ValueError('actual split/batch budget differs from plan')
 write(OUT/'units.json',units)
 for arm in ARMS[1:]:
  cfg=modes(arm);set_global_seed(42)
  model=DualPretrainer(cfg['fusion_mode'],third_task=cfg['third_task'],geometry_head_norm=True,opt_mode=arm)
  current=model.state_dict(); extra=[]
  for name,value in current.items():
   if name in shared:
    if value.shape!=shared[name].shape or value.dtype!=shared[name].dtype:raise ValueError('common tensor mismatch '+name)
    current[name]=shared[name].clone()
   else:
    if not name.startswith(('encoder.token_fusion.','align_proj2.','align_proj3.')):raise ValueError('undeclared added tensor '+name)
    extra.append(name)
  missing=set(shared)-set(current)
  if missing:raise ValueError('undeclared removed tensors')
  if cfg['fusion_mode']=='token':
   from src.modules.glt_dual_opt import BondTokenFusion
   set_global_seed(20261006);fusion=BondTokenFusion()
   for name,value in fusion.state_dict().items():current['encoder.token_fusion.'+name]=value.clone()
  model.load_state_dict(current,strict=True)
  for n,v in current.items():
   if not torch.isfinite(v).all():raise FloatingPointError('nonfinite step0 '+n)
  initial=OUT/'initialization'/f'{arm}_step0.pt';initial.parent.mkdir(parents=True,exist_ok=True)
  save_checkpoint(initial,dict(model=current,opt_mode=arm,modes=cfg,step=0,seed=42,shared_from=str(COMMON),new_tensor_names=extra,removed_tensor_names=sorted(missing)))
  config=dict(read(ROOT/'results/mcl_ph_r4_20261002/r3_glt_ref_fast_ra_01/run.json')['identity']['config'])
  config.pop('common_init_artifact',None)
  config.update(cfg,loss_weights=[1.,1.,.1 if cfg['third_task']=='align' else 0.],initial_state=str(initial),save_every=5000,optimization_plan=PLAN,optimization_revision='r1')
  write(OUT/'initialization'/f'{arm}.json',config)
 write(OUT/'initialization/initialization.json',dict(shared=info,fusion_seed=20261006,third_task_removed='fp',initializations='B/A/G共同旧参数；F/M共同token参数，共有张量从B完整step0复制'))
 write(OUT/'execution_status.json',dict(plan=PLAN,revision='r1',status='RUNNING',started=now(),pretraining_units=5,finetune_units=210,max_updates=333700,release_for_training=True))

def parent(arm):return PARENT if arm=='R_INCIDENT' else OUT/'pretrain'/arm/'deploy_05000.pt'
def ft(arm,task,fold,formal=False):
 import numpy as np,torch
 from torch import nn
 from torch.utils.data import DataLoader,Subset
 from sklearn.preprocessing import StandardScaler
 from src.utils import set_global_seed,scale_targets,evaluate
 import scripts.finetune_mcl_ph as old
 from src.dataset.glt_dual import dual_glt_collate
 from src.training.glt_dual_runtime import open_source,save_checkpoint
 from src.modules.glt_dual_pretrain import load_deployment
 from src.modules.glt_adaptation import AdaptationAdapter
 folder=OUT/('formal' if formal else 'development')/arm/task/f'fold{fold}'
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=modes(arm);set_global_seed(42+fold)
  model=old.GLTReferenceArm(fusion_mode=cfg['fusion_mode'],readout=cfg['readout'])
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  # R changes only readout over the exact historical encoder tensors.
  if arm=='R_INCIDENT':package=dict(package,readout='incident')
  else:
   for key in ('fusion_mode','readout','opt_mode','geometry_target','third_task','element_mask'):
    if package.get(key)!=cfg[key]:raise ValueError('deployment mode mismatch '+key)
  load_deployment(model.model,package,5000)
  model.model.opt_mode=arm
  head=torch.load(reference(task,fold)/'head_initial.pt',map_location='cpu',weights_only=False)
  old.periodic_tdl_head(model).load_state_dict(head,strict=True)
  model.to('cuda');old.configure_periodic_tdl_dropout(model,task)
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=OUT/'development'/arm/task/f'fold{fold}'
  if formal:
   selection=read(OUT/'selection_before_test.json')
   if selection['development_units']!=210:raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['modes']!=cfg or best['task']!=task or best['fold']!=fold:raise ValueError('formal model metadata mismatch')
   model.load_state_dict(best['state_dict'],strict=True)
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
  else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,split_manifest=manifest,record_index_path=DOWN/f'{task}_offsets.npy',expected_task_rows=manifest['sample_count'])
  if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows differ')
  ds=old.CleanDualDataset(source,frame['label'].to_numpy(dtype=np.float64))
  set_global_seed(42+fold)
  if formal:
   from src.utils import TargetScaler
   # Reuse the project's scalar interface without fitting on test labels.
   scaler=TargetScaler(task,StandardScaler(),transform_mode='standard');scaler.scaler.mean_=np.asarray(best['scaler_mean']);scaler.scaler.scale_=np.asarray(best['scaler_scale']);scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
   ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
  else:scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
  loader=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
  if formal:
   _,r2,truth,prediction=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),'cuda',scaler=scaler,amp_dtype='fp32')
   epoch=best['best_epoch'];total_epochs=0
  else:
   save_checkpoint(folder/'head_initial.pt',head)
   train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(42+fold))
   old_optimizer=old.periodic_tdl_optimizer
   def counted(*args,**kwargs):
    opt,evidence=old_optimizer(*args,**kwargs);real=opt.step
    def step(self,*a,**kw):
     updates['attempted']+=1
     result=real(*a,**kw);updates['completed']+=1;return result
    opt.step=types.MethodType(step,opt);return opt,evidence
   old.periodic_tdl_optimizer=counted
   def progress(rec):
    history.append(rec);print(json.dumps(rec),flush=True)
    write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
   result=old.run_periodic_tdl_epochs(model,train,loader,nn.MSELoss(),'cuda',task=task,scaler=scaler,progress=progress)
   if updates['attempted']!=updates['completed'] or updates['completed']!=epochs(task)*len(train):raise ValueError('FT updates mismatch')
   truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch'];total_epochs=len(history)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],modes=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

def free_gpus():
 # GPU 3 has desktop/unknown usage: only absence of compute processes permits use.
 result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 all_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 busy={line.split(',')[0].strip() for line in result.splitlines() if line.strip()}
 return [int(line.split(',')[0]) for line in all_gpu.splitlines() if line.split(',')[1].strip() not in busy]
def launch(name,cmd,gpus):
 log=LOG/name;log.mkdir(parents=True,exist_ok=False)
 command=shlex.join(cmd);(log/'command.txt').write_text(command+'\n');(log/'cwd.txt').write_text(str(ROOT)+'\n')
 script=f'''#!/bin/bash
cd {shlex.quote(str(ROOT))} || exit 90
export CUDA_VISIBLE_DEVICES={shlex.quote(','.join(map(str,gpus)))}
export OMP_NUM_THREADS=4
date -u +%FT%TZ > {shlex.quote(str(log/'start.txt'))}
{command} > {shlex.quote(str(log/'stdout.log'))} 2> {shlex.quote(str(log/'stderr.log'))}
code=$?
printf '%s\\n' "$code" > {shlex.quote(str(log/'exit.txt'))}
date -u +%FT%TZ > {shlex.quote(str(log/'end.txt'))}
exit "$code"
'''
 (log/'run.sh').write_text(script)
 subprocess.run(['tmux','new-window','-d','-t','Uni-Poly','-n','glt_dual_opt_r1_'+name,'bash '+shlex.quote(str(log/'run.sh'))],check=True)
 return log

def jobs(arm,formal=False):
 pending=[(t,f) for t in TASKS for f in range(5)];active={};failure=None
 while pending or active:
  available=free_gpus()
  if not failure:
   for gpu in available:
    if pending and gpu not in active:
     task,fold=pending.pop(0);cmd=['python','scripts/run_glt_dual_opt.py','--unit','formal' if formal else 'development','--arm',arm,'--task',task,'--fold',str(fold)]
     name=('formal_' if formal else 'dev_')+arm+'_'+task+'_f'+str(fold)
     active[gpu]=(name,launch(name,cmd,[gpu]))
  for gpu,(name,log) in list(active.items()):
   if (log/'exit.txt').exists():
    code=int((log/'exit.txt').read_text());print(f'{name} exit={code}',flush=True);del active[gpu]
    if code:failure=f'{name} exit={code}'
    if not formal:
     units=read(OUT/'units.json')
     for unit in units:
      if name=='dev_'+unit['arm']+'_'+unit['task']+'_f'+str(unit['fold']):unit['status']='FAILED' if code else 'COMPLETED'
     write(OUT/'units.json',units)
  if failure and not active:raise RuntimeError(failure+'; no retry; remaining units not run')
  time.sleep(5)

def metrics(arm,task,fold,formal=False):
 return read(OUT/('formal' if formal else 'development')/arm/task/f'fold{fold}'/'metrics.json')
def comparison(arm,ref,formal=False):
 import numpy as np
 gains=[]
 for task in TASKS:
  values=[]
  for f in range(5):
   current=metrics(arm,task,f,formal)['rmse']
   if ref in ARMS:baseline=metrics(ref,task,f,formal)['rmse']
   else:
    m=read(reference(task,f,ref=='I')/'metrics.json');baseline=m['test_rmse' if formal else 'best_validation_rmse']
   values.append((baseline-current)/baseline)
  gains.append(values)
 task_means=np.mean(gains,axis=1);macro=float(task_means.mean())
 return dict(reference=ref,fold_gains=gains,task_means=task_means.tolist(),macro=macro,positive_tasks=int((task_means>0).sum()),worst_task=float(task_means.min()),passed=bool(macro>=.01 and (task_means>0).sum()>=5 and task_means.min()>=-.02))
def select():
 candidates=[]
 for arm in ('R_INCIDENT','A_ALIGN','F_TOKEN','M_TOKEN_CE','G_SEAM_GEO'):
  primary=comparison(arm,'H' if arm=='R_INCIDENT' else 'B_FP');inc=comparison(arm,'I')
  import torch
  p=torch.load(parent(arm),map_location='cpu',weights_only=False);params=sum(v.numel() for v in p['state_dict'].values())
  candidates.append(dict(arm=arm,primary=primary,incumbent=inc,parameters=params,eligible=primary['passed'] and inc['passed']))
 eligible=[c for c in candidates if c['eligible']]
 order=list(('R_INCIDENT','A_ALIGN','F_TOKEN','M_TOKEN_CE','G_SEAM_GEO'))
 eligible.sort(key=lambda c:(-c['incumbent']['macro'],-c['incumbent']['worst_task'],c['parameters'],order.index(c['arm'])))
 write(OUT/'selection_before_test.json',dict(plan=PLAN,time=now(),development_units=210,chosen=eligible[0]['arm'] if eligible else 'I',candidates=candidates,best_paths=[str(OUT/'development'/a/t/f'fold{f}'/'best.pt') for a in ARMS for t in TASKS for f in range(5)]))

def finish():
 import numpy as np
 selection=read(OUT/'selection_before_test.json');chosen=selection['chosen'];rows=[]
 paper={'eea':('Periodic-TDL',.934),'egb':('Periodic-TDL (+Chem, +RC)',.923),'egc':('Periodic-TDL (+Chem, +RC)',.911),'ei':('Periodic-TDL',.839),'eps':('Periodic-TDL',.787),'nc':('Periodic-TDL',.856),'xc':('Periodic-TDL',.423)}
 for arm in ARMS:
  for task in TASKS:
   folds=[]
   for f in range(5):
    p=OUT/'formal'/arm/task/f'fold{f}'/'test_predictions.npz'
    with np.load(p,allow_pickle=False) as a:
     y=a['y_true'];pred=a['y_pred'];r2=1-float(np.sum((y-pred)**2)/np.sum((y-y.mean())**2));rmse=float(np.sqrt(np.mean((y-pred)**2)))
    folds.append(dict(fold=f,r2=r2,rmse=rmse,prediction=str(p)))
   r2s=[f['r2'] for f in folds];rmses=[f['rmse'] for f in folds];model,pr=paper[task]
   rows.append(dict(arm=arm,task=task,folds=folds,r2_mean=float(np.mean(r2s)),r2_std=float(np.std(r2s)),rmse_mean=float(np.mean(rmses)),rmse_std=float(np.std(rmses)),paper_model=model,paper_r2=pr,difference=float(np.mean(r2s))-pr))
 comparisons={a:dict(primary=comparison(a,'H' if a=='R_INCIDENT' else 'B_FP',True),incumbent=comparison(a,'I',True)) for a in ARMS if a!='B_FP'}
 if chosen!='I':promoted=comparisons[chosen]['primary']['passed'] and comparisons[chosen]['incumbent']['passed']
 else:promoted=False
 write(OUT/'final_review/seven_task_results.json',dict(rows=rows,selection_before_test=selection,comparisons=comparisons,M_minus_F=comparison('M_TOKEN_CE','F_TOKEN',True),promoted=promoted,retained=chosen if promoted else 'I',protocol='paper5_outer five-fold arithmetic mean; seed42+fold; ddof0; paper equivalence unverified; not blind test'))
 write(OUT/'execution_status.json',dict(plan=PLAN,status='PENDING_INDEPENDENT_REVIEW',logical_training_units=215,ft_epochs=14400,pretraining_updates=25000,ft_updates=308700,total_updates=333700,formal_units=210,engineering_test_calls=sum((p/'start.txt').exists() for p in LOG.glob('final_pytest*')),smoke=0,probes=0,release_for_training=False,finished=now()))
def consumption():
 ft_epochs=ft_updates=ft_attempted=0;pretrain=0;ft_starts=pretrain_starts=0;formal=0
 for log in LOG.iterdir():
  if not log.is_dir() or not (log/'start.txt').exists():continue
  if log.name.startswith('dev_'):ft_starts+=1
  elif log.name.startswith('pretrain_'):pretrain_starts+=1
  elif log.name.startswith('formal_'):formal+=1
 for folder in (OUT/'development').glob('*/*/fold*'):
  if (folder/'metrics.json').exists():
   m=read(folder/'metrics.json');ft_epochs+=m['epochs'];ft_updates+=m['updates']['completed'];ft_attempted+=m['updates']['attempted']
  elif (folder/'failure.json').exists():
   m=read(folder/'failure.json');ft_epochs+=m['completed_epochs'];ft_updates+=m['updates']['completed'];ft_attempted+=m['updates']['attempted']
  elif (folder/'progress.json').exists():
   m=read(folder/'progress.json');ft_epochs+=len(m['history']);ft_updates+=m['updates']['completed'];ft_attempted+=m['updates']['attempted']
 for path in (OUT/'pretrain').glob('*/records_rank0.jsonl'):
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()];pretrain+=max((r['step'] for r in rows),default=0)
 return dict(ft_process_starts=ft_starts,pretraining_process_starts=pretrain_starts,ft_completed_epochs=ft_epochs,ft_completed_updates=ft_updates,ft_attempted_updates=ft_attempted,pretraining_completed_updates_recorded=pretrain,total_completed_updates_recorded=pretrain+ft_updates,formal_process_starts=formal,recorded_counts_note='rank0 per-update records and FT counters; abrupt process loss may leave an unrecorded attempted update')

def campaign():
 initialize()
 try:
  jobs('R_INCIDENT')
  for arm in ARMS[1:]:
   while len(free_gpus())!=4:time.sleep(20)
   cmd=['torchrun','--standalone','--nproc_per_node=4','scripts/pretrain_glt_dual.py','--config',str(OUT/'initialization'/f'{arm}.json'),'--cohort-root',str(PI),'--cache-root',str(PI),'--dual-static-root',str(PI),'--pretrain-target-root',str(PI),'--output',str(OUT/'pretrain'/arm),'--prep-workers','8','--lmdb-readahead']
   log=launch('pretrain_'+arm,cmd,[0,1,2,3])
   while not (log/'exit.txt').exists():time.sleep(5)
   if int((log/'exit.txt').read_text()):raise RuntimeError('pretrain '+arm+' failed; no retry')
   jobs(arm)
  select()
  log=launch('final_pytest',['python','-m','pytest','-q','tests/test_glt_dual_opt.py'],[])
  while not (log/'exit.txt').exists():time.sleep(5)
  if int((log/'exit.txt').read_text()):raise RuntimeError('unique final engineering test failed')
  for arm in ARMS:jobs(arm,True)
  finish()
  write(OUT/'consumption.json',consumption())
 except BaseException as error:
  write(OUT/'consumption.json',consumption())
  write(OUT/'execution_status.json',dict(plan=PLAN,status='BLOCKED',error=repr(error),traceback=traceback.format_exc(),time=now(),release_for_training=False,no_retry=True))
  raise
if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 p=argparse.ArgumentParser();p.add_argument('--unit',choices=('development','formal'));p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
 if a.unit:ft(a.arm,a.task,a.fold,a.unit=='formal')
 else:campaign()
