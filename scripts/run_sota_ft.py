"""Explicit A0-A3 controller only; there is no test or later-stage entry."""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.sota_ft_contract import *
from src.training.sota_ft_runtime import finite,json_write,consume
from scripts.run_mcl_ph_r4_fast import write_shell,start_window,timestamp
from scripts.finetune_mcl_ph import compact_train_validation_indices
from src.training.glt_dual_runtime import require_tmux,open_source

def freeze():
 require_tmux();torch.set_num_threads(1);OUT.mkdir(exist_ok=False)
 pkg=torch.load(PARENT,map_location='cpu',weights_only=False);finite(pkg)
 assert pkg['step']==5000 and pkg['architecture']=='O8-BondPath-GalformerTrimer-Hop2' and pkg['fusion_mode']=='concat' and pkg['use_md200'] is False and pkg['torsion_modules'] is False
 resume=PARENT.with_name('resume_05000.pt');saved=torch.load(resume,map_location='cpu',weights_only=False);finite(saved)
 encoder={k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.') and not k.startswith('encoder.predictor.')}
 assert set(encoder)==set(pkg['state_dict'])
 assert all(torch.equal(encoder[k],v) for k,v in pkg['state_dict'].items())
 del saved,encoder
 protected=[PARENT,resume,DOWN/'direct_source.json',ROOT/'logs/gp_sota_20261004/status.json',ROOT/'logs/gp_sota_20261004/monitor_after_repair_01.py',ROOT/'logs/gp_sota_20261004/nc_r3_independent_review_01/report.json',ROOT/'SOTA_RESULT.md',ROOT/'PROJECT_HISTORY.md',ROOT/'CURRENT_PLAN_GP_SOTA_20261004_r3_ARCHIVE.md']
 scope=dict(plan=PLAN,revision=REV,status='FROZEN_A0_A3',allowed_stages=['smoke','development'],units={},parent_shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in pkg['state_dict'].items()},source_metadata=read(DOWN/'direct_source.json'),historical_fp_step0_exact='未核实',candidates=['F0-E5','SP-single','SP-E5'],gates=dict(mean_gain_min=.02,positive=2,dual_reference=['paired_F0-single','incumbent'],tie_percentage_points=.1,preferences=['single','no_SP']),budget=dict(full_starts=8,full_epochs=560,full_updates=6160,smoke_starts=2,smoke_epochs=6,smoke_updates=66,total_updates=6226),formal_test_authorized=False,created_at=timestamp())
 for task in ('eps','nc'):
  manifest=read(SPLITS/f'{task}.json');protected.extend([SPLITS/f'{task}.json',DOWN/f'{task}_offsets.npy'])
  for fold in (0,1):
   entry=manifest['folds'][fold];selected,train,val=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
   assert not set(selected)&set(entry['test_indices'])
   source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,record_index_path=DOWN/f'{task}_offsets.npy',split_manifest=manifest,expected_task_rows=manifest['sample_count'])
   try:
    assert frame['original_row'].astype(int).tolist()==selected
    labels=frame['label'].to_numpy(dtype=float);finite(labels.tolist());mean=[float(labels[train].mean())];scale=[float(labels[train].std())]
    keys=[source.samples[i][0].hex() for i in val]
   finally:source.close()
   old=reference(task,fold);head=torch.load(old/'head_initial.pt',map_location='cpu',weights_only=False);finite(head)
   assert {k:list(v.shape) for k,v in head.items()}=={'0.weight':[512,1024],'0.bias':[512],'3.weight':[1,512],'3.bias':[1]}
   best=torch.load(old/'best.pt',map_location='cpu',weights_only=False);finite(best)
   np.testing.assert_allclose(best['scaler_mean'],mean,rtol=0,atol=1e-12);np.testing.assert_allclose(best['scaler_scale'],scale,rtol=0,atol=1e-12)
   with np.load(old/'validation_predictions.npz',allow_pickle=False) as z:
    assert z['sample_keys'].tolist()==keys and z['validation_indices'].tolist()==entry['validation_indices']
    np.testing.assert_allclose(z['y_true'].reshape(-1),labels[val],rtol=2e-6,atol=2e-6)
    assert abs(float(np.sqrt(np.mean((z['y_true']-z['y_pred'])**2)))-read(old/'metrics.json')['best_validation_rmse'])<1e-8
   protected.extend([old/x for x in ('head_initial.pt','best.pt','metrics.json','validation_predictions.npz','runtime.json','acceptance.json')])
   for stage in (('smoke','development') if (task,fold)==('eps',0) else ('development',)):
    for arm in ('F0','SP'):
     r=recipe(stage,task,fold,arm);epochs=r['head_epochs']+r['joint_epochs'];k=key(stage,task,fold,arm)
     scope['units'][k]=dict(recipe=r,reserved_epochs=epochs,reserved_updates=epochs*math.ceil(len(train)/24),indices=dict(train=entry['train_indices'],validation=entry['validation_indices'],selected=selected),validation_keys=keys,scaler_mean=mean,scaler_scale=scale)
 scope['protected_inputs']={str(p):sha(p) for p in protected}
 json_write(OUT/'scope.json',scope);json_write(OUT/'a0_report.json',dict(status='PASS_A0_SAVED_CONTRACT',encoder_exact_finite=True,head_initial_finite=True,train_only_scalers_match_history=True,source_keys_match_history=True,unit_count=len(scope['units']),new_model_calls=0,updates=0,historical_fp_step0_exact='未核实'))
 print('PASS_A0',flush=True)

def launch(name,command,gpu=None):
 log=LOG/name;write_shell(log,command,gpu=gpu);start_window(name,log);return log

def wait(log):
 while not (log/'exit.code').exists():time.sleep(5)
 if int((log/'exit.code').read_text())!=0:raise RuntimeError('STOP_FAILED '+str(log))

def stage(stage):
 scope=read(OUT/'scope.json');units=[(k,u) for k,u in scope['units'].items() if u['recipe']['stage']==stage]
 units.sort(key=lambda x:(x[1]['recipe']['arm']!='F0',x[1]['recipe']['task'],x[1]['recipe']['fold']))
 active={}
 while units or active:
  for gpu in (0,1):
   if gpu not in active and units:
    k,u=units.pop(0);r=u['recipe'];log=launch(f"A_{stage}_{r['task']}_f{r['fold']}_{r['arm']}",['python','scripts/finetune_sota_ft.py','--stage',stage,'--task',r['task'],'--fold',str(r['fold']),'--arm',r['arm']],gpu=gpu);active[gpu]=(k,log)
  for gpu,(k,log) in list(active.items()):
   if not (log/'exit.code').exists():continue
   if int((log/'exit.code').read_text())!=0:
    # Stop only other owned windows; no retry or newly scheduled units.
    for other,(ok,ol) in active.items():
     if other!=gpu and not (ol/'exit.code').exists():subprocess.run(['tmux','send-keys','-t',ol.name,'C-c'],check=True)
    raise RuntimeError('unit failed: '+k)
   consume(OUT/k);del active[gpu]
  json_write(LOG/'status.json',dict(status='RUNNING_A0_A3',stage=stage,active=[k for k,l in active.values()],pending=[k for k,u in units],updated_at=timestamp(),formal_test='NOT_AUTHORIZED'))
  time.sleep(5)

def summary():
 rows=[];locks={}
 for task in ('eps','nc'):
  folds=[]
  for fold in (0,1):
   m={arm:consume(OUT/key('development',task,fold,arm)) for arm in ('F0','SP')}
   cells={f'{a}-{s}':m[a]['results'][s]['rmse'] for a in m for s in ('single','E5')}
   inc=read(reference(task,fold)/'metrics.json')['best_validation_rmse'];folds.append(dict(fold=fold,cells=cells,incumbent_rmse=inc))
  candidates=[]
  for c in ('F0-E5','SP-single','SP-E5'):
   gains={ref:[(v['cells']['F0-single'] if ref=='paired' else v['incumbent_rmse']) for v in folds] for ref in ('paired','incumbent')}
   gains={ref:[(r-v['cells'][c])/r for r,v in zip(rs,folds)] for ref,rs in gains.items()}
   eligible=all(np.mean(v)>=.02 and all(x>0 for x in v) for v in gains.values());score=min(float(np.mean(v)) for v in gains.values())
   candidates.append(dict(candidate=c,gains=gains,mean_gains={k:float(np.mean(v)) for k,v in gains.items()},eligible=eligible,score=score))
  valid=[c for c in candidates if c['eligible']];chosen=None
  if valid:
   best=max(c['score'] for c in valid);ties=[c for c in valid if best-c['score']<=.001]
   chosen=min(ties,key=lambda c:(not c['candidate'].endswith('single'),not c['candidate'].startswith('F0'),-c['score']))['candidate']
  locks[task]=dict(status='LOCKED_PENDING_REVIEW' if chosen else 'STOP_NO_CANDIDATE',chosen=chosen,folds=folds,candidates=candidates,A4_A5_authorized=False)
 json_write(OUT/'a3_selection.json',dict(plan=PLAN,revision=REV,status='A3_COMPLETE_PENDING_REVIEW',tasks=locks,formal_test='NOT_RUN',A4_A5_B_C='NOT_AUTHORIZED'))
 print(json.dumps(locks,ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args();require_tmux()
 if args.freeze:freeze();return
 scope=read(OUT/'scope.json');assert scope['status']=='FROZEN_A0_A3'
 wait(launch('A1_cpu',['env','CUDA_VISIBLE_DEVICES=','python','-m','pytest','-q','tests/test_sota_ft.py']))
 stage('smoke');stage('development');summary()
 json_write(LOG/'status.json',dict(status='A0_A3_COMPLETE_PENDING_REVIEW',formal_test='NOT_RUN',updated_at=timestamp()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  json_write(LOG/'controller_failure.json',dict(status='STOPPED',error=str(e),automatic_retry=False,updated_at=timestamp()));raise
