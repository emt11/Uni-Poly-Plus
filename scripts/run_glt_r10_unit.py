"""Fixed r10 initialization and FT/prediction units; reuse the original epoch loop."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from src.modules.glt_r10 import (OUT,COMMON,RUNTIME,REVISION,TASKS,PT_ARMS,FT_ARMS,
                                GEOMETRY,pt_config,validate_pt_config,validate_package,
                                configure_runtime,GraphFiLM,GraphAttention)
from scripts.run_glt_align_2d_unit import PROFILE
from scripts.run_glt_r10 import write
import scripts.finetune_mcl_ph as old
from src.training.glt_dual_runtime import (require_tmux,open_source,pretrain_source_fields,
                                          save_checkpoint)
from src.modules.glt_dual import strict_module_state,restore_pure_2d
from src.modules.glt_dual_pretrain import DualPretrainer,restore_deployment
from src.dataset.glt_dual import (build_dual_sample,dual_glt_collate,build_glt_2d_sample,
                                  glt_2d_collate)
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import set_global_seed,scale_targets,evaluate,TargetScaler

DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
PI=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'


def read(path):return json.loads(Path(path).read_text())


def parent(cycle,arm):
    if arm not in FT_ARMS[cycle]:raise ValueError('undefined FT arm')
    if cycle in ('F2','F3'):return BASE/'pretrain/F_CE/deploy_05000.pt'
    return OUT/cycle/'pretrain'/arm/'deploy_05000.pt'


def head_path(cycle,task,fold):
    if cycle in ('A1','A2'):
        return ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'/task/f'fold{fold}.pt'
    return ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'


def ft_config(cycle,arm):
    if arm not in FT_ARMS[cycle]:raise ValueError('undefined FT arm')
    pure=cycle in ('A1','A2')
    return dict(plan='GLT-OPTIMIZATION-20261011-01',revision='r10',cycle=cycle,arm=arm,
        r10_revision=REVISION,runtime_revision=RUNTIME,parent=str(parent(cycle,arm)),parent_step=5000,
        prediction_path='pure_2d' if pure else 'fused',input_dim=512 if pure else 1024,
        graph_fusion='film' if arm=='F2_FILM' else 'attention' if cycle=='F3' else 'none',
        profile=PROFILE,ft_profile='uniform_u1',seed='42+fold',source=str(DOWN),
        split=str(SPLIT),target_transform='train_only_standard',workers=0,
        common_initialization=str(COMMON),training_config=pt_config(cycle,arm) if cycle not in ('F2','F3') else None)


def validate_ft(cfg,cycle,arm):
    if cfg!=ft_config(cycle,arm):raise ValueError('r10 FT configuration mismatch')
    return cfg


def trainer_for(cfg):
    validate_pt_config(cfg)
    model=DualPretrainer('token',fusion_revision='glt_fusion_r1',geometry_head_norm=True,
        opt_mode='GLT_FUSION_R1',atom_ce_input=cfg['atom_ce_input'],
        align_enabled=cfg['align_enabled'],align_revision=cfg['align_revision'],
        align_init_seed=cfg['align_init_seed'],align_direction=cfg['align_direction'],
        torsion=cfg['torsion_modules'],torsion_revision=cfg.get('torsion_revision'),
        nonbonded_revision=cfg['nonbonded_revision'],**GEOMETRY)
    configure_runtime(model.encoder)
    return model


def initialize(cycle):
    folder=OUT/cycle/'initialization';folder.mkdir(parents=True,exist_ok=False)
    if cycle=='Q':
        from src.dataset.glt_r10 import build_qc_sidecar
        build_qc_sidecar(OUT/'Q/qc',ROOT/'results/glt_quality_fusion_20261010/r1/quality_completed/pi1m_quality.jsonl',PI/'records.jsonl')
    common=torch.load(COMMON,map_location='cpu',weights_only=False)
    if common.get('step')!=0:raise ValueError('common initialization is not step0')
    records=[]
    for arm in PT_ARMS[cycle]:
        cfg=pt_config(cycle,arm);set_global_seed(cfg['seed']);model=trainer_for(cfg)
        state=model.state_dict();extra=set(state)-set(common['model'])
        if set(common['model'])-set(state):raise ValueError('missing public step0 tensors')
        expected_extra=328448 if cfg['align_enabled'] else 33473 if cfg['torsion_modules'] else 34368 if cfg['nonbonded_revision'] else 0
        if sum(state[k].numel() for k in extra)!=expected_extra:raise ValueError('new PT parameter count mismatch')
        for name,value in common['model'].items():
            if state[name].shape!=value.shape or state[name].dtype!=value.dtype or not torch.isfinite(value).all():raise ValueError('common PT tensor mismatch '+name)
            state[name]=value.clone()
        strict_module_state(model,state)
        save_checkpoint(folder/(arm+'_step0.pt'),dict(model=model.state_dict(),step=0,opt_mode=cfg['opt_mode'],arm=arm,config=cfg))
        write(folder/(arm+'_PT.json'),cfg)
        records.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),extra_parameters=expected_extra,common=str(COMMON)))
        del model,state
    for arm in FT_ARMS[cycle]:write(folder/(arm+'.json'),ft_config(cycle,arm))
    write(folder/'summary.json',dict(cycle=cycle,status='PASS',pt_arms=records,ft_arms=list(FT_ARMS[cycle]),runtime_revision=RUNTIME))


class Fused(nn.Module):
    def __init__(self,model,kind):
        super().__init__();self.model=model
        self.graph_fusion=GraphFiLM() if kind=='film' else GraphAttention() if kind=='attention' else None
    def features(self,data):
        encoded=self.model.encode(data);features=self.model.fuse(encoded)
        return self.graph_fusion(features,encoded['geometry_valid']) if self.graph_fusion is not None else features
    def forward(self,data):return self.model.predictor(self.features(data))


def build_model(package,cfg,head):
    if cfg['training_config'] is not None:
        if validate_package(package)!=cfg['training_config']:raise ValueError('FT parent recipe mismatch')
    elif package.get('r10_revision') or package.get('align_enabled') or package.get('torsion_modules'):
        raise ValueError('F2/F3 requires original F_CE parent')
    if cfg['prediction_path']=='pure_2d':
        return restore_pure_2d(package,head)
    encoder=restore_deployment(package);configure_runtime(encoder)
    model=Fused(encoder,cfg['graph_fusion']);strict_module_state(old.periodic_tdl_head(model),head)
    return model


class CleanDataset(old.CleanDualDataset):
    def __init__(self,source,targets,cfg):super().__init__(source,targets);self.cfg=cfg
    def __getitem__(self,index):
        if index not in self._clean_cache:
            top,tri,text,static,_=pretrain_source_fields(self.source,index)
            if self.cfg['prediction_path']=='pure_2d':data=build_glt_2d_sample(top,text,static=static)
            else:
                pt=self.cfg['training_config'] or {}
                data=build_dual_sample(top,tri,text,static=static,geometry_mode='finite_incident',
                    torsion_mode='on' if pt.get('torsion_modules') else None,torsion_revision=pt.get('torsion_revision'))
                if pt.get('nonbonded_revision'):
                    from src.dataset.glt_r10 import attach_nonbonded
                    attach_nonbonded(data,static,tri,tri.trimer_pos)
                data.md200=torch.zeros((1,200));data.md200_valid=torch.zeros(1,dtype=torch.bool)
            self._clean_cache[index]=data
        data=old.clone_clean_record(self._clean_cache[index]);data.y=torch.tensor([float(self.targets[index])],dtype=torch.float32)
        return data


def run_ft(cycle,arm,task,fold,formal=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('single physical GPU2/3 required')
    cfg=validate_ft(read(OUT/cycle/'initialization'/(arm+'.json')),cycle,arm)
    folder=OUT/cycle/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
    folder.mkdir(parents=True,exist_ok=False)
    updates={'attempted':0,'completed':0};history=[];source=None
    try:
        set_global_seed(42+fold)
        package=torch.load(parent(cycle,arm),map_location='cpu',weights_only=False)
        head=torch.load(head_path(cycle,task,fold),map_location='cpu',weights_only=False)
        model=build_model(package,cfg,head);del package
        model.to('cuda:0');old.configure_periodic_tdl_dropout(model,task,profile='uniform_u1')
        manifest=read(SPLIT/(task+'.json'));entry=manifest['folds'][fold]
        best_folder=OUT/cycle/'finetune'/arm/task/f'fold{fold}'
        if formal:
            frozen=read(OUT/cycle/'evaluation_before_test.json')
            if frozen['validation_units']!=35*len(FT_ARMS[cycle]):raise ValueError('incomplete frozen FT matrix')
            matches=[u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
            if len(matches)!=1:raise ValueError('missing frozen formal binding')
            binding=matches[0];best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
            for key,value in dict(config=cfg,task=task,fold=fold,seed=42+fold,parent=str(parent(cycle,arm)),head_initial=str(head_path(cycle,task,fold))).items():
                if best.get(key)!=value or binding.get(key)!=value:raise ValueError('formal metadata mismatch '+key)
            scaler_record=read(best_folder/'scaler.json')
            if scaler_record!=binding['scaler'] or best['scaler']!=scaler_record or best['best_epoch']!=binding['best_epoch']:raise ValueError('frozen scaler/epoch mismatch')
            strict_module_state(model,best['state_dict'])
            selected=sorted(entry['test_indices']);local_val=list(range(len(selected)))
            if selected!=binding['test_indices']:raise ValueError('formal test indices changed')
        else:
            selected,local_train,local_val=old.compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        source,frame=open_source(DOWN,DOWN,task=task,selected_indices=selected,split_manifest=manifest,verify_key_digest=False)
        if frame['original_row'].astype(int).tolist()!=selected:raise ValueError('selected rows changed')
        ds=CleanDataset(source,frame['label'].to_numpy(dtype=np.float64),cfg)
        set_global_seed(42+fold)
        if formal:
            scaler=TargetScaler(task,StandardScaler(),transform_mode='standard')
            scaler.scaler.mean_=np.asarray(scaler_record['mean']);scaler.scaler.scale_=np.asarray(scaler_record['scale'])
            scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
            ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
        else:
            scaler=scale_targets(ds,task,train_indices=local_train,transform_mode='standard')
            scaler_record=dict(mean=scaler.scaler.mean_.tolist(),scale=scaler.scaler.scale_.tolist(),fit='train',transform='standard')
            old_bind=[u for u in read(BASE/'selection_before_test.json')['units'] if (u['arm'],u['task'],u['fold'])==('F_CE',task,fold)]
            if len(old_bind)!=1 or any(old_bind[0]['scaler'][k]!=scaler_record[k] for k in ('mean','scale')):raise ValueError('original train-only scaler mismatch')
        collate=glt_2d_collate if cfg['prediction_path']=='pure_2d' else dual_glt_collate
        validation=DataLoader(Subset(ds,local_val),batch_size=24,shuffle=False,num_workers=0,collate_fn=collate)
        if formal:
            _,r2,truth,prediction=evaluate(AdaptationAdapter(model),validation,nn.MSELoss(),'cuda:0',scaler=scaler,amp_dtype='fp32')
            epoch=best['best_epoch']
        else:
            model.feature_cache_context=dict(parent=str(parent(cycle,arm)),configuration=cfg,input_view='clean_train_validation',dtype='fp32',sample_indices=selected)
            save_checkpoint(folder/'head_initial.pt',head)
            train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(42+fold))
            def observe(kind):updates[kind]+=1;write(folder/'updates.json',updates)
            def progress(record):
                history.append(record);print(json.dumps(record),flush=True)
                write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
            result=old.run_periodic_tdl_epochs(model,train,validation,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,profile='uniform_u1',step_observer=observe)
            if len(history)!=70 or updates!={'attempted':70*len(train),'completed':70*len(train)}:raise ValueError('FT budget mismatch')
            write(folder/'scaler.json',scaler_record)
            truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch']
            save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,
                parent=str(parent(cycle,arm)),head_initial=str(head_path(cycle,task,fold)),best_epoch=epoch,scaler=scaler_record,
                optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],outer_test='NOT_RUN'))
        truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
        ids=selected if formal else entry['validation_indices']
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,
            indices=np.asarray(ids),sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),
            scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,
            geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='RUN' if formal else 'NOT_RUN')
        write(folder/'metrics.json',dict(status='PASS',cycle=cycle,arm=arm,task=task,fold=fold,config=cfg,
            parameters=sum(p.numel() for p in model.parameters()),epochs=len(history),updates=updates,best_epoch=epoch,
            r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
    except BaseException as error:
        write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history)))
        raise
    finally:
        if source is not None:source.close()


if __name__=='__main__':
    require_tmux()
    parser=argparse.ArgumentParser();parser.add_argument('--cycle',choices=tuple(FT_ARMS),required=True)
    parser.add_argument('--initialize',action='store_true');parser.add_argument('--arm');parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--fold',type=int,choices=range(5));parser.add_argument('--formal',action='store_true')
    args=parser.parse_args()
    if args.initialize:initialize(args.cycle)
    elif args.arm in FT_ARMS[args.cycle] and args.task and args.fold is not None:run_ft(args.cycle,args.arm,args.task,args.fold,args.formal)
    else:parser.error('invalid fixed unit')
