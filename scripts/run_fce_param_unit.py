"""Fixed FCE r2 training and selected formal units, single physical GPU2/3."""
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
import scripts.finetune_mcl_ph as old
from src.training.fce_param import (ARMS,TASKS,RUNTIME,recipe_for,run_periodic_tdl_epochs,configure_periodic_tdl_dropout)
from src.training.glt_dual_runtime import require_tmux,open_source,pretrain_source_fields,save_checkpoint
from src.modules.glt_dual import strict_module_state
from src.modules.glt_dual_pretrain import restore_deployment
from src.dataset.glt_dual import build_dual_sample,dual_glt_collate,build_glt_2d_sample,glt_2d_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import set_global_seed,scale_targets,evaluate,TargetScaler
OUT=ROOT/'results/fce_param_20261014/r2'
LOG=ROOT/'logs/fce_param_20261014/r2'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
BASE=ROOT/'results/glt_periodic_seam_20261009/r1'
PARENT=BASE/'pretrain/F_CE/deploy_05000.pt'

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def epochs(task):return 60 if task=='egc' else 70

def head_path(task,fold):
    return ROOT/'results/glt_fusion_20261007/r1/finetune/B0'/task/f'fold{fold}'/'head_initial.pt'

def config_for(arm,task):
    return dict(plan='FCE-PARAM-20261014-01',revision='r2',arm=arm,task=task,
        runtime_revision=RUNTIME,parent=str(PARENT),parent_step=5000,
        prediction_path='fused',input_dim=1024,graph_fusion='none',training_config=None,
        seed='42+fold',source=str(DOWN),split=str(SPLIT),recipe=recipe_for(arm,task),
        target_transform='train_only_standard',feature_cache='head_train_validation_only',
        geometry_mode='finite_incident',geometry_revision='glt_periodic_seam_r1',
        angle_value_enabled=True,readout='incident',fusion_attention_mode='global',
        fusion_implementation='loop',collect_fusion_attention=False)

def validate_config(cfg,arm,task):
    if cfg!=config_for(arm,task):raise ValueError('FCE configuration mismatch')
    return cfg

def build_model(package,cfg,head):
    reference=read(BASE/'initialization/F_CE.json')
    for k in ('fusion_mode','fusion_revision','use_md200','atom_ce_input','align_enabled','local_enabled','third_task','geometry_revision','geometry_mode','angle_value_enabled','readout'):
        if package.get(k)!=reference[k]:raise ValueError('parent metadata mismatch '+k)
    encoder=restore_deployment(package,expected_step=5000)
    encoder.token_fusion.set_implementation('loop');encoder.collect_fusion_attention=False
    model=Fused(encoder,'none');strict_module_state(old.periodic_tdl_head(model),head)
    return model

class Fused(nn.Module):
    def __init__(self,model,kind):
        super().__init__();self.model=model
        if kind!='none':raise ValueError('graph architecture changes forbidden')
        self.graph_fusion=None
    def features(self,data):
        encoded=self.model.encode(data);features=self.model.fuse(encoded)
        return self.graph_fusion(features,encoded['geometry_valid']) if self.graph_fusion is not None else features
    def forward(self,data):return self.model.predictor(self.features(data))


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


def run_ft(arm,task,fold,formal=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3'):raise ValueError('single physical GPU2/3 required')
    cfg=validate_config(read(OUT/'initialization'/arm/(task+'.json')),arm,task)
    folder=OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
    folder.mkdir(parents=True,exist_ok=False)
    updates={'attempted':0,'completed':0};history=[];source=None
    try:
        set_global_seed(42+fold)
        package=torch.load(PARENT,map_location='cpu',weights_only=False)
        head=torch.load(head_path(task,fold),map_location='cpu',weights_only=False)
        model=build_model(package,cfg,head);del package
        model.to('cuda:0');configure_periodic_tdl_dropout(model,task)
        manifest=read(SPLIT/(task+'.json'));entry=manifest['folds'][fold]
        best_folder=OUT/'finetune'/arm/task/f'fold{fold}'
        if formal:
            frozen=read(OUT/'selection_before_test.json')
            if arm not in ('R0',frozen['candidate']):raise ValueError('unselected arm cannot read TEST')
            if frozen['validation_units']!=140:raise ValueError('incomplete frozen FT matrix')
            matches=[u for u in frozen['units'] if (u['arm'],u['task'],u['fold'])==(arm,task,fold)]
            if len(matches)!=1:raise ValueError('missing frozen formal binding')
            binding=matches[0];best=torch.load(best_folder/'best.pt',map_location='cpu',weights_only=False)
            for key,value in dict(config=cfg,task=task,fold=fold,seed=42+fold,parent=str(PARENT),head_initial=str(head_path(task,fold))).items():
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
            model.feature_cache_context=dict(parent=str(PARENT),configuration=cfg,input_view='clean_train_validation',dtype='fp32',sample_indices=selected)
            save_checkpoint(folder/'head_initial.pt',head)
            train=DataLoader(Subset(ds,local_train),batch_size=24,shuffle=True,num_workers=0,collate_fn=collate,generator=torch.Generator().manual_seed(42+fold))
            def observe(kind):
                if kind=='attempted' and updates[kind]>=epochs(task)*len(train):raise RuntimeError('unit update budget exhausted')
                updates[kind]+=1;write(folder/'updates.json',updates)
            def progress(record):
                history.append(record);print(json.dumps(record),flush=True)
                write(folder/'progress.json',dict(history=history,updates=updates,outer_test='NOT_RUN'))
            result=run_periodic_tdl_epochs(model,train,validation,nn.MSELoss(),'cuda:0',task=task,scaler=scaler,progress=progress,recipe=cfg['recipe'],step_observer=observe)
            if len(history)!=epochs(task) or updates!={'attempted':epochs(task)*len(train),'completed':epochs(task)*len(train)}:raise ValueError('FT budget mismatch')
            write(folder/'scaler.json',scaler_record)
            truth,prediction=result['predictions'];r2=result['best_r2'];epoch=result['best_epoch']
            save_checkpoint(folder/'best.pt',dict(state_dict=result['state'],config=cfg,task=task,fold=fold,seed=42+fold,
                parent=str(PARENT),head_initial=str(head_path(task,fold)),best_epoch=epoch,scaler=scaler_record,
                optimizer_groups=result['optimizer_groups_by_stage'],optimizer_updates=updates['completed'],outer_test='NOT_RUN'))
        truth=np.asarray(truth).reshape(-1);prediction=np.asarray(prediction).reshape(-1)
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not math.isfinite(float(r2)):raise FloatingPointError('nonfinite predictions')
        ids=selected if formal else entry['validation_indices']
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),y_true=truth,y_pred=prediction,
            indices=np.asarray(ids),sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),
            scaler_mean=scaler.scaler.mean_,scaler_scale=scaler.scaler.scale_,best_epoch=epoch,
            geometry_valid=np.asarray([source.entries[i]['geometry_status']=='PASS' for i in local_val]),outer_test='RUN' if formal else 'NOT_RUN')
        write(folder/'metrics.json',dict(status='PASS',arm=arm,task=task,fold=fold,config=cfg,
            parameters=sum(p.numel() for p in model.parameters()),epochs=len(history),updates=updates,best_epoch=epoch,
            r2=float(r2),rmse=float(np.sqrt(np.mean((truth-prediction)**2))),outer_test='RUN' if formal else 'NOT_RUN'))
    except BaseException as error:
        write(folder/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),updates=updates,completed_epochs=len(history)))
        raise
    finally:
        if source is not None:source.close()


if __name__=='__main__':
    require_tmux()
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--formal',action='store_true');a=p.parse_args()
    run_ft(a.arm,a.task,a.fold,a.formal)
