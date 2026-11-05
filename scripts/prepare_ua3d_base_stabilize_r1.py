#!/usr/bin/env python3
"""Only normal prepare: roles/scalers and one new random factory; no forwards."""
import os,sys,math,importlib.metadata,platform,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import ua3d_base_stabilize_r1 as t

def main():
    t.require_tmux();t.install_stop_handler();t.check_clock(True);t.check_space()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('prepare must hide GPU')
    setup=t.m.configure('DET');t.seed_all(42);t.update_plan('RUNNING','prepare','新周期recipe/source_factory、原数据角色和唯一随机初始化。')
    state=dict(status='RUNNING',prepare_starts=1,initialization=0,heads=0,historical_tensor_reads=0,model_forwards=0)
    t.write(t.OUT/'prepare_consumption.json',state)
    try:
        records=[];splits={};count=dict(train=0,validation=0,test=0,updates=0);reference={}
        for task in t.TASKS:
            split=t.read(t.SPLIT/f'{task}.json');splits[task]=split;reference[task]=[]
            if split.get('validation_is_test') is not False or len(split['folds'])!=5:raise ValueError('split role identity')
            for fold,e in enumerate(split['folds']):
                t.check_clock(True);cfg=t.read(t.base_path(task,fold)/'config.json')
                if cfg['task']!=task or cfg['fold']!=fold or cfg['recipe']['pt']!=t.m.PT or cfg['recipe']['ft']!=t.m.FT or cfg['model']!=t.m.base.CONFIG or cfg['workers']!=0 or cfg['batch']!=24 or cfg['epochs']!=70:raise ValueError('reference recipe roles differ from registered recovery')
                sets=[set(e[k]) for k in ('train_indices','validation_indices','test_indices')]
                if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('split overlap')
                ds=t.d.Data(task,fold,False,t.DOWN,t.SPLIT,cfg['input_cache'])
                try:t.scaler_validate(ds.raw_targets,ds.train,t.read(cfg['scaler_reference']))
                finally:ds.close()
                for name,k in [('train','train_indices'),('validation','validation_indices'),('test','test_indices')]:count[name]+=len(e[k])
                count['updates']+=70*math.ceil(len(e['train_indices'])/24)
                reference[task].append(t.read(t.B/'formal/F_PD_N1_F0'/task/f'fold{fold}'/'metrics.json')['r2'])
                records.append(dict(task=task,fold=fold,source_config=str(t.base_path(task,fold)/'config.json'),config=cfg,train_rows=len(e['train_indices']),validation_rows=len(e['validation_indices']),test_rows=len(e['test_indices']),new_head_seed=20261025+100*t.TASKS.index(task)+fold,old_model_tensor_reads=0,test_targets_read=False))
        if count!=dict(train=18786,validation=4714,test=5875,updates=56000):raise ValueError('35-fold input accounting mismatch')
        pt_meta=t.read(t.PT_CACHE/'metadata.json');mapping=t.read(t.MAP_ROOT/'metadata.json')
        if pt_meta['schema']!='ua2d_prepared_templates_r4' or pt_meta['status']!='COMPLETE' or pt_meta['train_rows']!=911391 or mapping['records']!=910756 or mapping['status']!='COMPLETE' or mapping['parent_cache']!=str(t.PT_CACHE):raise ValueError('PT raw/logical roles')
        t.exclusive(t.OUT/'recipe_audit.json',dict(plan=t.PLAN,reference=t.REFERENCE,units=records,reference_tasks=t.summary(reference),source_native_factory='src/modules/ua2d_galformer_pretrain.py:UA2DGalformerPretrainer',PT_data_config=dataclasses.asdict(t.d.raw.arm_config('D_A020')),PT_data_config_role='mask/corruption only; alignment weight comes exclusively from model.PT=.08 per direction',model=t.m.CONFIG,training_workers=0,PT_workers=4,PT_prefetch=1,new_statistics=0,changes=['restored random factory; no modal re-reset','DET numerical execution','fixed same-trajectory SWA versus BEST'],historical_tensor_reads=0))
        factory=dict(plan=t.PLAN,native_seed=42,native_constructor="UA2DGalformerPretrainer(PretrainConfig(strategy='mask_align'))",native_extraction_order=list(t.m.NATIVE),mask_extraction=list(t.m.MASK),unused_rng_consumers=['proj2','proj3'],line_head_isolated_seed=20261019,modal_constructor="modal_modules('TMOL3M')",modal_seed=20261025,modal_second_reset=False,head_seed='20261025+100*task_index+fold',head_constructor_dropout=.1,actual_FT_head_dropout=.5,gate_seed=20261025,historical_tensor_reads=0)
        t.exclusive(t.OUT/'initialization/source_factory.json',factory)
        state['initialization']=1;t.write(t.OUT/'prepare_consumption.json',state)
        value,fusion,rng=t.m.fresh_factory();stream=dict(policy='disjoint_modal_v1',B=86400010,counters=[0,0,0],forward_domain=[0,60000],terminal_counter=60000)
        package=dict(schema=t.m.INITIAL_SCHEMA,plan=t.PLAN,revision='r1',step=0,seed=42,config=t.m.CONFIG,fp_prediction=False,historical_tensor_reads=0,stream=stream,state_dict=value,ft_fusion_state=fusion,factory_rng=rng)
        t.m.validate_pt(package,0);t.save_checkpoint(t.INITIAL,package)
        factory.update(parameter_keys=list(value),shapes={k:list(v.shape) for k,v in value.items()},dtypes={k:str(v.dtype) for k,v in value.items()},strict_new_object_extraction=True);t.write(t.OUT/'initialization/source_factory.json',factory)
        import torch
        for index,task in enumerate(t.TASKS):
            for fold in range(5):
                t.check_clock(True);seed=20261025+100*index+fold
                with torch.random.fork_rng(devices=[]):torch.default_generator.manual_seed(seed);head=t.m.native.property_head(.1)
                tensors=t.m.clone_tree(head.state_dict())
                if set(tensors)!={'0.weight','0.bias','3.weight','3.bias'} or [list(tensors[k].shape) for k in ('0.weight','0.bias','3.weight','3.bias')]!=[[256,256],[256],[1,256],[1]]:raise ValueError('property head shape/rule contradiction')
                state['heads']+=1;t.write(t.OUT/'prepare_consumption.json',state);t.save_checkpoint(t.head_path(task,fold),dict(schema=t.m.HEAD_SCHEMA,plan=t.PLAN,revision='r1',task=task,fold=fold,seed=seed,state_dict=tensors))
        t.exclusive(t.OUT/'identity.json',dict(plan=t.PLAN,revision='r1',splits=splits,counts=count,initial=str(t.INITIAL),head_units=35,new_statistics=0,historical_tensor_reads=0,test_targets_read=False,PT_raw=str(t.PT_CACHE),logical_map=str(t.MAP_ROOT)))
        t.exclusive(t.OUT/'registry/canonical.json',dict(plan=t.PLAN,groups=list(t.GROUPS),canonical=list(t.CANONICAL),new_ft=[t.config(g,task,fold) for g in t.GROUPS for task in t.TASKS for fold in range(5)],formal_estimators=['BEST','SWA'],formal_units=210,PRIMARY_RUN='S42_A',FINAL_ESTIMATOR='BASE_SWA_LAST4_CYCLES'))
        t.write(t.OUT/'environment.json',dict(setup=setup,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),python=platform.python_version(),pyg=importlib.metadata.version('torch-geometric'),numpy=importlib.metadata.version('numpy'),affinity=sorted(os.sched_getaffinity(0)),CPU_metadata_limit=min(96,len(os.sched_getaffinity(0))),disk_free=t.check_space(),PT_metadata=pt_meta,logical_mapping=mapping))
        state.update(status='COMPLETE',ended=t.now());t.write(t.OUT/'prepare_consumption.json',state);t.check_clock(True)
        print('ONE NEW RANDOM FACTORY /35 HEADS COMPLETE; HISTORICAL TENSORS0; FORWARDS0',flush=True)
    except BaseException:
        import traceback
        state.update(status='FAILED',error=traceback.format_exc());t.write(t.OUT/'prepare_consumption.json',state);raise
if __name__=='__main__':main()
