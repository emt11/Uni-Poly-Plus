#!/usr/bin/env python3
"""Matched PT initialization, fixed process validation and downstream units."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/glt_optimization_20261011/r3/S3'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
COMMON=BASE/'initialization/F_CE_step0.pt'
ARMS=('M_G','M_L','M_A');TASKS=('eea','egb','egc','ei','eps','nc','xc')

import torch
from torch import nn
from src.modules.glt_dual_opt import BondTokenFusion
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment

def strict_restore(module,state):
 current=module.state_dict()
 if set(current)!=set(state):raise ValueError('FT tensor names mismatch')
 for name,value in state.items():
  if value.shape!=current[name].shape or value.dtype!=current[name].dtype or not torch.isfinite(value).all():raise ValueError('FT tensor shape/dtype/finite mismatch '+name)
 module.load_state_dict(state,strict=True)

def connection_config(arm):
 if arm not in ARMS:raise ValueError('invalid S3 arm')
 return dict(endpoint_revision='glt_pt_locality_r3',endpoint_module='mlp512_nobias_zero_last',endpoint_init_seed=314159,
  endpoint_input_mode='masked_mean' if arm=='M_A' else 'zero',fusion_attention_mode='global' if arm=='M_G' else 'incident_only',incident_bias=0.0)

def validate_config(cfg,arm):
 from src.modules.glt_dual import locality_options
 locality_options(cfg)
 expected=dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r3',cycle='S3',arm=arm,
  fusion_mode='token',fusion_revision='glt_fusion_r1',geometry_revision='glt_periodic_seam_r1',geometry_mode='finite_incident',
  angle_value_enabled=True,readout='incident',use_md200=False,md200_root=None,atom_ce_input='fused',align_enabled=False,
  local_enabled=False,third_task='none',torsion_modules=False,fusion_gate='none',joint_3d_drop_probability=0.0,
  parent=str(parent(arm)),ft_batch_size=24,ft_amp_dtype='fp32',ft_workers=0,ft_seed_formula='42+fold',split='paper5_outer',target_transform='train_only_standard',
  **connection_config(arm))
 if any(k not in cfg or type(cfg[k]) is not type(v) or cfg[k]!=v for k,v in expected.items()):raise ValueError('invalid S3 configuration')
 return cfg

class LocalityArm(nn.Module):
 def __init__(self,package,arm,fold,cfg):
  super().__init__()
  from src.modules.glt_dual import locality_options
  if arm not in ARMS or fold not in range(5):raise ValueError('invalid FT arm/fold')
  validate_config(cfg,arm)
  modes=locality_options(package)
  if modes!=connection_config(arm):raise ValueError('PT/FT locality mismatch')
  for k in ('fusion_mode','fusion_revision','geometry_revision','geometry_mode','angle_value_enabled','readout','use_md200','atom_ce_input','third_task','align_enabled','local_enabled','torsion_modules'):
   if package.get(k)!=cfg[k]:raise ValueError('parent architecture mismatch '+k)
  self.model=build_dual_glt_model('token',fusion_revision='glt_fusion_r1',use_md200=False,torsion=False,
   **{k:cfg[k] for k in ('geometry_revision','geometry_mode','angle_value_enabled','readout')},**modes)
  load_deployment(self.model,package,5000)
  for name,value in package['state_dict'].items():
   if not torch.isfinite(value).all() or not torch.equal(self.model.state_dict()[name],value):raise ValueError('invalid parent parameter '+name)
 def features(self,data):return self.model.fuse(self.model.encode(data))
 def forward(self,data):return self.model.predictor(self.features(data))
import datetime,math,os,traceback,types
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def epochs(task):return 60 if task=='egc' else 70
def parent(arm):return OUT/'pretrain'/arm/'deploy_05000.pt'
def ft_folder(arm,task,fold):return OUT/'finetune'/arm/task/f'fold{fold}'
def run_ft(arm,task,fold,formal=False):
 ft_candidate=None
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
 folder=OUT/'formal'/arm/task/f'fold{fold}' if formal else ft_folder(arm,task,fold)
 folder.mkdir(parents=True,exist_ok=False)
 updates={'attempted':0,'completed':0};history=[];source=None
 try:
  cfg=read(OUT/'initialization'/f'{arm}.json');set_global_seed(42+fold)
  package=torch.load(parent(arm),map_location='cpu',weights_only=False)
  model=LocalityArm(package,arm,fold,cfg)
  if cfg['parent']!=str(parent(arm)) or cfg['fusion_gate']!=model.model.fusion_gate or cfg['joint_3d_drop_probability']!=0.0:raise ValueError('FT config/model mismatch')
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(42+fold)
   common_head=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1))
  head=torch.load(ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt',map_location='cpu',weights_only=False)
  if set(head)!=set(common_head.state_dict()):raise ValueError('B0 head tensor names mismatch')
  for name,value in head.items():
   target=common_head.state_dict()[name]
   if value.shape!=target.shape or value.dtype!=target.dtype or not torch.isfinite(value).all():raise ValueError('B0 head tensor shape/dtype/value mismatch '+name)
  old.periodic_tdl_head(model).load_state_dict(head,strict=True)
  model.to('cuda:0');old.configure_periodic_tdl_dropout(model,task)
  manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
  best_folder=ft_folder(arm,task,fold)
  if formal:
   selection=read(OUT/'evaluation_before_test.json')
   if selection['validation_units']!=105:raise ValueError('selection not complete')
   best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
   if best['config']!=cfg or best['task']!=task or best['fold']!=fold or best.get('seed')!=42+fold or best.get('parent')!=str(parent(arm)):raise ValueError('formal model metadata mismatch')
   units=[u for u in selection['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
   if len(units)!=1 or units[0]['best']!=str(best_folder/'best.pt') or selection['configs'][arm]!=cfg or read(best_folder/'scaler.json')!=units[0]['scaler'] or best['scaler_mean']!=units[0]['scaler']['mean'] or best['scaler_scale']!=units[0]['scaler']['scale']:
    raise ValueError('formal frozen best/config/scaler mismatch')
   strict_restore(model,best['state_dict'])
   selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
   if selected!=units[0]['test_indices'] or units[0]['parent']!=str(parent(arm)) or best['best_epoch']!=units[0]['best_epoch'] or units[0]['fusion_attention_mode']!=cfg['fusion_attention_mode'] or units[0]['incident_bias']!=cfg['incident_bias'] or any(units[0][k]!=cfg[k] for k in connection_config(arm)):raise ValueError('formal frozen indices/parent/epoch mismatch')
  else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
  md_root=read(Path(cfg['md200_root'])/'metadata.json').get('downstream_sidecar') if cfg['use_md200'] else None
  if cfg['use_md200'] and not md_root:raise ValueError('MD sidecar missing downstream binding')
  source,frame=open_source(DOWN,DOWN,task=task,dual_static_root=DOWN,selected_indices=selected,split_manifest=manifest,record_index_path=DOWN/f'{task}_offsets.npy',expected_task_rows=manifest['sample_count'],verify_key_digest=False,md200_root=md_root)
  if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows differ')
  class NewCleanDataset(old.CleanDualDataset):
   def __getitem__(self,index):
    if index not in self._clean_cache:
     from src.training.glt_dual_runtime import pretrain_source_fields
     from src.dataset.glt_dual import build_dual_sample
     top,tri,text,static,_=pretrain_source_fields(self.source,index)
     self._clean_cache[index]=build_dual_sample(top,tri,text,static=static,geometry_mode=cfg['geometry_mode'])
    data=old.clone_clean_record(self._clean_cache[index])
    data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32)
    if not hasattr(data,'md200'):
     if cfg['use_md200']:raise ValueError('MD-on FT sample missing sidecar')
     data.md200=torch.zeros((1,200));data.md200_valid=torch.zeros(1,dtype=torch.bool)
    return data
  ds=NewCleanDataset(source,frame['label'].to_numpy(dtype=np.float64))
  set_global_seed(42+fold)
  if formal:
   from src.utils import TargetScaler
   # Reuse the project's scalar interface without fitting on test labels.
   scaler=TargetScaler(task,StandardScaler(),transform_mode='standard');scaler.scaler.mean_=np.asarray(best['scaler_mean']);scaler.scaler.scale_=np.asarray(best['scaler_scale']);scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
   ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
  else:
   scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
   references=[u for u in read(BASE/'selection_before_test.json')['units'] if (u['arm'],u['task'],u['fold'])==('F_CE',task,fold)]
   if len(references)!=1 or references[0]['scaler']['mean']!=scaler.scaler.mean_.tolist() or references[0]['scaler']['scale']!=scaler.scaler.scale_.tolist():raise ValueError('F_CE train-only scaler mismatch')
  loader=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
  if formal:
   _,r2,truth,prediction=evaluate(AdaptationAdapter(model),loader,nn.MSELoss(),'cuda:0',scaler=scaler,amp_dtype='fp32')
   epoch=best['best_epoch'];total_epochs=0
  else:
   save_checkpoint(folder/'head_initial.pt',head)
   train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(42+fold))
   old_optimizer=old.periodic_tdl_optimizer
   def counted(*args,**kwargs):
    opt,evidence=old_optimizer(*args,**kwargs);real=opt.step
    def step(self,*a,**kw):
     updates['attempted']+=1
     write(folder/'updates.json',updates)
     result=real(*a,**kw);updates['completed']+=1
     write(folder/'updates.json',updates);return result
    opt.step=types.MethodType(step,opt);return opt,evidence
   old.periodic_tdl_optimizer=counted
   def progress(rec):
    rec=dict(rec,fusion_attention_mode=cfg['fusion_attention_mode'],incident_bias=cfg['incident_bias'])
    history.append(rec);print(json.dumps(rec),flush=True)
    write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
   result=old.run_periodic_tdl_epochs(model,train,loader,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,ft_candidate=ft_candidate)
   if updates['attempted']!=updates['completed'] or updates['completed']!=epochs(task)*len(train):raise ValueError('FT updates mismatch')
   write(folder/'scaler.json',dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard'))
   truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch'];total_epochs=len(history)
   save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(arm)),best_epoch=epoch,scaler_mean=scaler.scaler.mean_.tolist(),scaler_scale=scaler.scaler.scale_.tolist(),history=history,optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],target_transform='standard',scaler_fit='train',outer_test='NOT_RUN'))
  truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
  if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
  ids=selected if formal else entry['validation_indices']
  np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),indices=np.asarray(ids),scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='RUN' if formal else 'NOT_RUN')
  write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,modes=cfg,parameters=sum(p.numel() for p in model.parameters()),epochs=total_epochs,updates=updates,r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
 except BaseException as error:
  write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history),time=now()))
  raise
 finally:
  if source is not None:source.close()

def initialize():
 from src.modules.glt_dual_pretrain import DualPretrainer
 folder=OUT/'initialization';folder.mkdir(parents=True,exist_ok=False)
 common=torch.load(COMMON,map_location='cpu',weights_only=False)
 if common['step']!=0:raise ValueError('common PT step mismatch')
 base_cfg=read(BASE/'initialization/F_CE.json');summaries=[]
 for arm in ARMS:
  cfg=dict(base_cfg,**connection_config(arm),plan='GLT-OPTIMIZATION-20261011-01',revision='r3',cycle='S3',arm=arm,
   cycle_root=str(OUT),expected_world_size=2,initial_state=str(folder/f'{arm}_step0.pt'),fusion_gate='none',quality_root=None,
   joint_3d_drop_probability=0.0,parent=str(parent(arm)),ft_batch_size=24,ft_amp_dtype='fp32',ft_workers=0,ft_seed_formula='42+fold',split='paper5_outer',target_transform='train_only_standard')
  validate_config(cfg,arm);torch.manual_seed(42)
  model=DualPretrainer('token',fusion_revision='glt_fusion_r1',atom_ce_input='fused',geometry_head_norm=True,opt_mode=cfg['opt_mode'],
   **{k:cfg[k] for k in ('geometry_revision','geometry_mode','angle_value_enabled','readout')},**connection_config(arm))
  state=model.state_dict();new=set(state)-set(common['model'])
  if new!={'encoder.endpoint_mlp.0.weight','encoder.endpoint_mlp.2.weight'} or set(common['model'])-set(state):raise ValueError('S3 common tensor names mismatch')
  for k,v in common['model'].items():
   if v.shape!=state[k].shape or v.dtype!=state[k].dtype or not torch.isfinite(v).all():raise ValueError('S3 common tensor mismatch '+k)
   state[k]=v.clone()
  model.load_state_dict(state,strict=True)
  for k,v in common['model'].items():
   if not torch.equal(v,model.state_dict()[k]):raise ValueError('S3 common value changed '+k)
  from src.training.glt_dual_runtime import save_checkpoint
  save_checkpoint(folder/f'{arm}_step0.pt',dict(model=state,step=0,opt_mode=cfg['opt_mode'],arm=arm,config=cfg))
  write(folder/f'{arm}.json',cfg)
  summaries.append(dict(arm=arm,pt_parameters=sum(p.numel() for p in model.parameters()),new_parameters=sum(state[k].numel() for k in new),common=str(COMMON),endpoint_seed=314159))
 write(folder/'summary.json',dict(arms=summaries,common=str(COMMON),state_names='public PT tensors plus identical endpoint MLP'))

def evaluate_process_validation(model, config, step, device, rank, world):
    """Exactly 1024 fixed presentations; leave model/RNG/training stream alone."""
    from scripts.evaluate_glt_pt_endpoint import load_endpoint_subset,component_means,PI
    import torch
    import torch.distributed as dist
    from src.training.glt_dual_runtime import rng_state, restore_rng, open_source, pretrain_source_fields, move_labels
    from src.dataset.glt_dual_pretrain import prepare_pretrain_sample, pretrain_collate
    if world != 2 or step not in (1000,2500,5000):raise ValueError('process validation budget mismatch')
    saved = rng_state(); training = model.training; source = None
    folder = Path(config['cycle_root'])/'pt_validation'/config['arm']/f'step{step:05d}'
    presented = 0
    try:
        if rank == 0:folder.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        subset=load_endpoint_subset(config['sample_index_artifact'])
        source,_=open_source(PI,PI,selected_indices=subset['indices'].tolist(),verify_key_digest=False,readonly_readahead=True)
        from src.training.glt_dual_runtime import apply_geometry_quality
        source=apply_geometry_quality(source,config.get('quality_root'))
        if len(source)!=1024 or any(row['pretrain_role']!='validation' for row in source.entries):
            raise ValueError('process validation source role mismatch')
        model.eval()
        sums=torch.zeros(5,dtype=torch.float64,device=device);counts=torch.zeros_like(sums)
        targets=torch.zeros(7,dtype=torch.float64,device=device)
        with torch.no_grad():
            for start in range(0,1024,168):
                records=[]
                for position in range(start+rank*min(84,(1024-start)//2),start+(rank+1)*min(84,(1024-start)//2)):
                    top,tri,text,static,_=pretrain_source_fields(source,position)
                    records.append(prepare_pretrain_sample(top,tri,text,seed=42,key=source.samples[position][0].hex(),position=position,
                        sigma=config['noise_sigma'],ratio=config['atom_mask_ratio'],static=static,
                        opt_mode=config['opt_mode'],geometry_mode=config['geometry_mode']))
                data,labels=pretrain_collate(records);data=data.to(device);labels=move_labels(labels,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):result=model(data,labels)
                values=torch.cat((result['sums'][:2],result['raw_geometry_sums'],result['extra_sums'][:1]))
                den=torch.stack((result['counts'][0],result['counts'][1],result['counts'][1],result['angle_graphs'],result['extra_counts'][0]))
                if not torch.isfinite(values).all():raise FloatingPointError('nonfinite process validation')
                sums+=values.double();counts+=den.double()
                targets+=torch.stack((result['targets'][0],result['targets'][1],result['targets'][2],
                    data.bond_length_weight.sum(),data.geometry_valid.sum(),
                    torch.tensor(data.periodic_keys.size(0) if config['geometry_mode']=='periodic_template' else 0,device=device),
                    torch.tensor(data.bond_distance.numel(),device=device))).double()
                presented+=len(records)
                write(folder/f'progress_rank{rank}.json',dict(presented=presented,training_updates=0))
        for value in (sums,counts,targets):dist.all_reduce(value)
        total=torch.tensor(presented,device=device);dist.all_reduce(total)
        if int(total)!=1024:raise ValueError('validation presentation mismatch')
        if rank==0:
            write(folder/'metrics.json',dict(status='PASS',step=step,arm=config['arm'],presented=int(total),
                components={name:dict(sum=float(sums[i]),count=int(counts[i]),mean=component_means(sums,counts)[i]) for i,name in enumerate(('atom_ce','geometry','length_mse','angle_mse','align'))},
                targets=dict(zip(('masked_atoms','physical_length_targets','physical_angles','length_weight_denominator','valid_graphs','periodic_bonds','physical_bonds'),targets.tolist())),
                checkpoint_selection=False,padding=0,config=config))
    except BaseException as error:
        if folder.exists():write(folder/f'failure_rank{rank}.json',dict(error=repr(error),traceback=traceback.format_exc(),presented=presented))
        raise
    finally:
        if source is not None:source.close()
        model.train(training);restore_rng(saved)

if __name__=='__main__':
 from src.training.glt_dual_runtime import require_tmux
 require_tmux()
 p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--arm',choices=ARMS)
 p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--formal',action='store_true')
 a=p.parse_args()
 if a.initialize:initialize()
 elif a.arm and a.task and a.fold is not None:run_ft(a.arm,a.task,a.fold,a.formal)
 else:raise ValueError('invalid unit arguments')
