"""r16 single GPU neural FT/formal unit; R0 recipe and fixed cached r10 KRR."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from sklearn.preprocessing import StandardScaler
import scripts.finetune_mcl_ph as old
from scripts.run_fce_tune_t3b_unit import (validate_parent as validate_g05_parent,pt_config_for)
from src.training.fce_param import (TASKS,recipe_for,run_periodic_tdl_epochs,
    configure_periodic_tdl_dropout,finetune_criterion)
from src.training.glt_dual_runtime import require_tmux,open_source,save_checkpoint
from src.modules.glt_dual import strict_module_state
from src.modules.glt_dual_pretrain import restore_deployment
from src.dataset.glt_dual import dual_glt_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import set_global_seed,scale_targets,evaluate,TargetScaler
from src.training.fce_kernel import GlobalBlendGate,bind_branch_arrays,metrics,blend_arrays
from src.training.fce_nonbonded_ft import (ARMS,OUT,LOG,PARENT,POLICY,parent,REVISION,NB_REVISION,INPUT_SPEC,MODULE_INITIAL,input_policy,validate_initial,build_model,NonbondedDataset,stage_groups,NonbondedInput)
MSE_POLICY='MSE';RUNTIME='fce_nonbonded_ft_ft_r16_v1'
R10=ROOT/'results/fce_tune_20261016/global_blend'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def epochs(task):return 10+recipe_for('R0',task)['joint_epochs']
def head_path(task,fold):return ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'
def config_for(arm,task):
    if task not in TASKS:raise ValueError('undefined nonbonded task')
    return dict(plan='FCE-SOTA-20261015-01',revision='r16',arm=arm,task=task,runtime_revision=RUNTIME,
        source_used=str(OUT/'source_used'),parent=str(parent(arm)),parent_step=5000,seed='42+fold',
        prediction_path='fused',input_dim=1024,graph_fusion='none',training_config=pt_config_for('T3B_G05'),input_policy=input_policy(arm),module_initial=str(MODULE_INITIAL),nonbonded_revision=NB_REVISION,
        source=str(DOWN),split=str(SPLIT),recipe=recipe_for('R0',task),loss_policy=MSE_POLICY,adaptation_policy=None,
        target_transform='train_only_standard',feature_cache='head_train_validation_only',geometry_mode='finite_incident',
        geometry_revision='glt_periodic_seam_r1',angle_value_enabled=True,readout='incident',
        fusion_attention_mode='global',fusion_implementation='loop',collect_fusion_attention=False)
def validate_config(cfg,task):
    if cfg!=config_for(cfg.get("arm"),task):raise ValueError('nonbonded config/recipe/loss mismatch')
    return cfg


def source_binding(arm,task,fold):
    return read(OUT/'initialization/bindings'/arm/task/f'fold{fold}.json')

def initialize():
    folder=OUT/'initialization';folder.mkdir(exist_ok=False)
    source=read(R10/'selection_before_test.json');gate=GlobalBlendGate();gate.freeze(source)
    if source['weight']!=.75:raise ValueError('r10 .75 policy required')
    module=NonbondedInput()
    initial=dict(revision=REVISION,seed=314173,parameters=sum(p.numel() for p in module.parameters()),state_dict={n:v.detach().cpu().clone() for n,v in module.state_dict().items()})
    validate_initial(initial);save_checkpoint(MODULE_INITIAL,initial)
    for arm in ARMS:
        package=torch.load(parent(arm),map_location='cpu',weights_only=False);validate_g05_parent(package,'T3B_G05')
        write(folder/arm/'parent_binding.json',dict(parent=str(parent(arm)),step=5000,config=pt_config_for('T3B_G05'),module_initial=str(MODULE_INITIAL),input_policy=input_policy(arm)))
        for u in source['r9_freeze']['units']:
            task,fold=u['task'],u['fold'];ref=u['reference'];candidate=config_for(arm,task)
            if ref['config']['seed']!='42+fold' or ref['seed']!=42+fold or ref['parent']!=str(PARENT) or ref['head_initial']!=str(head_path(task,fold)):
                raise ValueError('reference parent/head/seed mismatch')
            if ref['config']['recipe']!=candidate['recipe'] or u['scaler']!=ref['scaler']:
                raise ValueError('reference R0/parent/scaler mismatch')
            for k in ('prediction_path','input_dim','graph_fusion','target_transform','feature_cache','geometry_mode','geometry_revision','angle_value_enabled','readout','fusion_attention_mode','fusion_implementation','collect_fusion_attention'):
                if ref['config'][k]!=candidate[k]:raise ValueError('reference architecture/data condition mismatch '+k)
            manifest=read(SPLIT/(task+'.json'));entry=manifest['folds'][fold]
            for role in ('train','validation','test'):
                values=entry[role+'_indices'];expected=sorted(values) if role=='test' else values
                if ref[role+'_indices']!=expected:raise ValueError('reference split mismatch')
            if not head_path(task,fold).is_file():raise FileNotFoundError(head_path(task,fold))
            val=Path(u['model']).parent/'validation'/f"alpha_{u['alpha']:g}"/'predictions.npz'
            if not val.is_file():raise FileNotFoundError(val)
            binding=dict(task=task,fold=fold,config=candidate,reference=ref,
                fixed_krr=dict(alpha=u['alpha'],model=u['model'],validation=str(val),scaler=u['scaler'],
                    train_indices=u['train_indices'],validation_indices=u['validation_indices'],test_indices=u['test_indices'],feature_spec=source['r9_freeze']['policy']['feature_spec']),
                identities=source['r9_freeze']['identities'][task],manifest=manifest)
            write(folder/'bindings'/arm/task/f'fold{fold}.json',binding);write(folder/arm/(task+'.json'),candidate)
    print('G05 step5000, common zero-output NonbondedInput and 70 R0 units bound; no initialization training',flush=True)


def train_candidate(model,train,validation,device,*,cfg,task,scaler,progress,step_observer):
    validate_config(cfg,task)
    return run_periodic_tdl_epochs(model,train,validation,finetune_criterion(),device,
        task=task,scaler=scaler,progress=progress,recipe=cfg['recipe'],step_observer=step_observer,adaptation_policy=None)


def validate_checkpoint(meta,binding):
    for k in ('config','task','fold','seed','parent','head_initial','best_epoch','scaler','loss_policy','adaptation_policy','input_policy','module_initial','nonbonded_revision','optimizer_groups'):
        if meta.get(k)!=binding[k]:raise ValueError('nonbonded checkpoint/freeze mismatch '+k)
    if meta.get('optimizer_updates')!=binding['expected_updates']:raise ValueError('nonbonded checkpoint update mismatch')


def validate_frozen(frozen):
    if not frozen.get('frozen') or frozen.get('policy')!=POLICY:raise ValueError('nonbonded TEST before complete freeze')
    units=frozen['units'];expected={(a,t,f) for a in ARMS for t in TASKS for f in range(5)}
    if len(units)!=70 or {(u['arm'],u['task'],u['fold']) for u in units}!=expected:raise ValueError('70 neural bests required before TEST')
    for u in units:
        validate_config(u['config'],u['task'])
        if u['seed']!=42+u['fold'] or u['parent']!=str(parent(u['arm'])) or u['head_initial']!=str(head_path(u['task'],u['fold'])) or u['loss_policy']!=MSE_POLICY or u['adaptation_policy'] is not None or u['input_policy']!=input_policy(u['arm']) or u['module_initial']!=str(MODULE_INITIAL) or u['nonbonded_revision']!=NB_REVISION:
            raise ValueError('nonbonded frozen parent/head/loss/seed mismatch')
        if u['reference']['config']['seed']!='42+fold' or u['reference']['seed']!=u['seed'] or u['config']['recipe']!=u['reference']['config']['recipe'] or u['scaler']!=u['fixed_krr']['scaler']:
            raise ValueError('nonbonded frozen R0/reference/scaler mismatch')
    return frozen


def selected_history(history):
    if not history or any(not math.isfinite(r['validation_rmse']) for r in history):raise ValueError('invalid nonbonded validation history')
    return min(history,key=lambda r:r['validation_rmse'])


def run_ft(arm,task,fold,formal=False):
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('physical GPU2/3 only')
    cfg=validate_config(read(OUT/'initialization'/arm/(task+'.json')),task);source_meta=source_binding(arm,task,fold)
    if cfg!=source_meta['config']:raise ValueError('initialization config changed')
    ref=source_meta['reference'];entry=source_meta['manifest']['folds'][fold];binding=None
    if formal:
        frozen=validate_frozen(read(OUT/'selection_before_test.json'))
        binding=next(u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold))
    folder=OUT/('formal' if formal else 'ft')/arm/task/f'fold{fold}';folder.mkdir(parents=True,exist_ok=False)
    updates=dict(attempted=0,completed=0);history=[];source=None
    try:
        set_global_seed(42+fold);package=torch.load(parent(arm),map_location='cpu',weights_only=False)
        head=torch.load(ref['head_initial'],map_location='cpu',weights_only=False)
        initial=validate_initial(torch.load(MODULE_INITIAL,map_location='cpu',weights_only=False))
        model=build_model(package,cfg,head,initial);del package
        model.to('cuda:0');configure_periodic_tdl_dropout(model,task,recipe=cfg['recipe'])
        count=epochs(task)*math.ceil(len(entry['train_indices'])/24)
        best_folder=OUT/'ft'/arm/task/f'fold{fold}'
        if formal:
            best=torch.load(binding['best'],map_location='cpu',weights_only=False);validate_checkpoint(best,binding);strict_module_state(model,best['state_dict'])
            selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
            if selected!=binding['test_indices']:raise ValueError('frozen TEST indices changed')
            scaler_record=binding['scaler']
        else:selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        source,frame=open_source(DOWN,DOWN,task=task,selected_indices=selected,split_manifest=source_meta['manifest'],verify_key_digest=False)
        if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected source rows changed')
        for i,idx in enumerate(selected):
            if source.samples[i][0].hex()!=source_meta['identities'][str(idx)]['sample_key'] or frame.iloc[i]['normalized_smiles']!=source_meta['identities'][str(idx)]['normalized_smiles']:
                raise ValueError('source sample identity mismatch')
        roles=['test']*len(selected) if formal else ['train' if i in set(entry['train_indices']) else 'validation' for i in selected]
        published_rows=[0]
        def publish_input(stats):
            rows=stats['rows']
            if len(rows)>published_rows[0]:
                with (folder/'input_rows.jsonl').open('a') as stream:
                    for row in rows[published_rows[0]:]:stream.write(json.dumps(row)+'\n')
                published_rows[0]=len(rows)
            write(folder/'input_consumption.json',{k:v for k,v in stats.items() if k!='rows'})
        ds=NonbondedDataset(source,frame['label'].to_numpy(dtype=np.float64),cfg,indices=selected,roles=roles,on_stats=publish_input);set_global_seed(42+fold)
        if formal:
            scaler=TargetScaler(task,StandardScaler(),transform_mode='standard');scaler.scaler.mean_=np.asarray(scaler_record['mean']);scaler.scaler.scale_=np.asarray(scaler_record['scale'])
            scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
            ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
        else:
            scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
            scaler_record=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
            if scaler_record!=ref['scaler']:raise ValueError('nonbonded original train-only scaler mismatch')
        validation=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=dual_glt_collate)
        if formal:
            _,r2,truth,prediction=evaluate(AdaptationAdapter(model),validation,finetune_criterion(),'cuda:0',scaler=scaler,amp_dtype='fp32');epoch=best['best_epoch']
        else:
            model.feature_cache_context=dict(parent=str(parent(arm)),configuration=dict(cfg,sample_roles=roles),input_view='clean_train_validation',dtype='fp32',sample_indices=selected)
            save_checkpoint(folder/'head_initial.pt',head)
            train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=dual_glt_collate,generator=torch.Generator().manual_seed(42+fold))
            def observe(kind):
                if updates[kind]>=count:raise RuntimeError('nonbonded unit update budget exhausted')
                updates[kind]+=1;write(folder/'updates.json',updates)
            parameter_groups={}
            def progress(record):
                if record['stage'] not in parameter_groups:
                    parameter_groups[record['stage']]=stage_groups(model,record['stage'],record['optimizer_groups'])
                    write(folder/'parameter_groups.json',parameter_groups)
                if record['stage']=='head' and any(not torch.equal(v.detach().cpu(),initial['state_dict'][n]) for n,v in model.model.glt.nonbonded_input.state_dict().items()):raise ValueError('new module changed during head phase')
                row=copy.deepcopy(record);row['loss_policy']=MSE_POLICY;row['adaptation_policy']=None
                history.append(row);write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'));print(json.dumps(row),flush=True)
            result=train_candidate(model,train,validation,'cuda:0',cfg=cfg,task=task,scaler=scaler,progress=progress,step_observer=observe)
            if len(history)!=epochs(task) or updates!=dict(attempted=count,completed=count):raise ValueError('nonbonded FT budget mismatch')
            write(folder/'scaler.json',scaler_record);truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch']
            save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,loss_policy=MSE_POLICY,adaptation_policy=None,input_policy=input_policy(arm),module_initial=str(MODULE_INITIAL),nonbonded_revision=NB_REVISION,task=task,fold=fold,seed=42+fold,
                parent=str(parent(arm)),head_initial=ref['head_initial'],best_epoch=epoch,scaler=scaler_record,optimizer_groups=parameter_groups,optimizer_updates=count,outer_test='NOT_RUN'))
        if len(ds.rows)!=len(selected) or len(ds.attempted)!=len(selected):raise ValueError('input materialization coverage mismatch')
        truth=np.asarray(truth,dtype=np.float64).reshape(-1);prediction=np.asarray(prediction,dtype=np.float64).reshape(-1)
        values=metrics(truth,prediction);ids=selected if formal else entry['validation_indices']
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,
            indices=np.asarray(ids),sample_keys=[source.samples[i][0].hex() for i in local_val],scaler_mean=scaler_record['mean'],scaler_scale=scaler_record['scale'],best_epoch=epoch)
        if formal:
            # This is the first cached TEST read, after the full 70-best freeze.
            r10=read(R10/'final_review/seven_task_results.json');prior=next(r for r in r10['folds'] if (r['task'],r['fold'])==(task,fold))
            if prior['alpha']!=binding['fixed_krr']['alpha'] or prior['weight']!=.75:raise ValueError('fixed r10 KRR alpha/weight mismatch')
            with np.load(prior['paths']['krr'],allow_pickle=False) as saved:krr={k:saved[k].copy() for k in saved.files}
            krunit=dict(binding,alpha=binding['fixed_krr']['alpha']);aligned=bind_branch_arrays(krr,krunit,'test',source_meta['identities'],('y_pred',))
            if not np.array_equal(aligned['y_true'],truth):raise ValueError('nonbonded/KRR TEST labels mismatch')
            fusion=blend_arrays(prediction,aligned['y_pred'],.75)
            np.savez(folder/'fused_predictions.npz',y_true=truth,y_pred=fusion,y_neural=prediction,indices=ids,sample_keys=[source.samples[i][0].hex() for i in local_val],
                scaler_mean=scaler_record['mean'],scaler_scale=scaler_record['scale'],best_epoch=epoch,weight=.75,krr_alpha=binding['fixed_krr']['alpha'])
            extra=dict(fused=metrics(truth,fusion),krr_path=prior['paths']['krr'],neural_test_forward_calls=1)
        else:extra={}
        write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,config=cfg,loss_policy=MSE_POLICY,adaptation_policy=None,input_policy=input_policy(arm),module_initial=str(MODULE_INITIAL),nonbonded_revision=NB_REVISION,parameters=sum(p.numel() for p in model.parameters()),
            epochs=len(history),updates=updates,best_epoch=epoch,**values,outer_test='RUN' if formal else 'NOT_RUN',**extra))
    except BaseException as error:
        write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history)));raise
    finally:
        if source is not None:source.close()


if __name__=='__main__':
    require_tmux();p=argparse.ArgumentParser();p.add_argument('--initialize',action='store_true');p.add_argument('--arm',choices=ARMS);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--formal',action='store_true');a=p.parse_args()
    if a.initialize:initialize()
    elif a.arm and a.task and a.fold is not None:run_ft(a.arm,a.task,a.fold,a.formal)
    else:p.error('explicit initialization or unit required')
