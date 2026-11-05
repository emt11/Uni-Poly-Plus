"""One isolated anchored cycle. Original checkpoints, scalers and histories are read-only."""
import datetime
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from src.dataset import ua3d_anchor_r2 as d
from src.modules import ua3d_anchor_r2 as m
from src.modules import ua3d_stability as numerical
from src.dataset import ua3m_paperfuse as original_data
from src.training.glt_dual_runtime import require_tmux,save_checkpoint
from src.training.ua2d_p10k_nofp_align import metrics,process_identity,process_alive,SOTA
from src.training.ua2d_a020_fp_fusion import transform,scaler_validate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ua3d_anchor_opt_20261102/r2'; LOG=ROOT/'logs/ua3d_anchor_opt_20261102/r2'
BASE=ROOT/'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
INPUT=ROOT/'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1/inputs'
SIDE=ROOT/'results/ua3d_ph_20261028/r1/repairs/gpu2_r2/inputs'
PACK=ROOT/'results/ua3d_ph_opt_20261031/r1/packing'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
PLAN='UA3D-BASE-ANCHOR-20261102-01'; REVISION='r2'; RESOURCE='gpu2-pt0-fttest4-anchor-r2'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
SPEC=json.loads((ROOT/'configs/ua3d_anchor_r2_gpu2.json').read_text())
ACTIVATION=json.loads((OUT/'authorization/activation.json').read_text())
T0=ACTIVATION['tN_unix']; DEADLINE=ACTIVATION['deadline_unix']; REFERENCE=.804675462276949


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.tmp.{os.getpid()}')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); os.replace(q,p)


def exclusive(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)


def contract():
    return dict(plan=PLAN,revision=REVISION,resource_revision=RESOURCE,physical_gpus=[2],logical_device='cuda:0',world_size=1,phase_exclusive=True)


def guard(reserve=900,preparation=False):
    if time.time()>=DEADLINE-reserve:
        raise TimeoutError('hard deadline and closing reserves')
    if preparation and time.time()>=T0+SPEC['preparation_seconds']:
        raise TimeoutError('8h preparation limit')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def base_folder(task,fold):
    return BASE/'finetune/F_PD_N1_F0'/task/f'fold{fold}'


def job_path(kind,arm=None,task=None,fold=None,repeat=None):
    if kind in ('cache_trainval','cache_test'):
        return OUT/'feature_cache'/('test' if kind=='cache_test' else 'trainval')/task/f'fold{fold}'
    if kind=='diagnostic':
        label,r=repeat.split('_'); return OUT/'diagnostics'/label/arm/r
    if kind in ('ft','test'):
        return OUT/('finetune' if kind=='ft' else 'formal')/arm/task/f'fold{fold}'
    return OUT/('preflight_cpu' if kind=='preflight' else 'final_cpu')


def config(arm,task,fold):
    if arm not in m.ARMS or task not in TASKS or fold not in range(5):
        raise ValueError('fixed identity matrix')
    return dict(**contract(),arm=arm,task=task,fold=fold,seed=42+fold,base_best=str(base_folder(task,fold)/'best.pt'),
                batch=24,epochs=70,workers=0,precision='FP32',optimizer=SPEC['optimizer'],scheduler=SPEC['scheduler'],
                loss='MSE(y_hat,y)+0.01*mean((y_hat-y_BASE)^2)',same_batch_difference=True,residual_gain=1,
                frozen_base=True,base_eval=True,adapter_seed=2026102801,reference=REFERENCE)


def register():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU registration requires hidden GPU')
    prior=read(BASE/'final_review/results.json')
    if prior['formal']['F_PD_N1_F0']['macro']!=REFERENCE or prior['FINAL']!='E_P_SYM_SHARED':
        raise ValueError('original BASE identity/result source')
    source=read(BASE/'registry/recipes.json')['recipes']['F_PD_N1_F0']
    if source['ft']!=m.base.DEFAULT_FT:
        raise ValueError('original FT recipe')
    anchors={}; splits={}; rows=read(INPUT/'rows.json'); sums=dict(train=0,validation=0,test=0,train_batches=0,validation_batches=0,test_batches=0)
    for task in TASKS:
        splits[task]=read(SPLIT/f'{task}.json')
        for fold,e in enumerate(splits[task]['folds']):
            for role in ('train','validation','test'):
                n=len(e[role+'_indices']); sums[role]+=n; sums[role+'_batches']+=math.ceil(n/24)
            folder=base_folder(task,fold); cfg=read(folder/'config.json'); met=read(folder/'metrics.json'); scaler=read(folder/'scaler.json')
            if cfg['recipe']!=source or met['config']!=cfg or cfg['seed']!=42+fold or cfg['task']!=task or cfg['fold']!=fold or met['status']!='COMPLETE' or met['scaler']!=scaler:
                raise ValueError('BASE35 source config/scaler/task/fold')
            anchors[f'{task}/{fold}']=dict(task=task,fold=fold,best=str(folder/'best.pt'),source_config=cfg,
                schema=m.base.FT_SCHEMA,best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=scaler,
                validation_predictions=str(folder/'validation_predictions.npz'))
    if sums!=dict(train=18786,validation=4714,test=5875,train_batches=800,validation_batches=210,test_batches=270):
        raise ValueError('split budget mismatch')
    exclusive(OUT/'registry/anchors.json',dict(anchors=anchors,splits=splits,rows=rows,split_counts=sums,reference_source=str(BASE/'final_review/results.json')))
    exclusive(OUT/'registry/matrix.json',dict(units=[config(a,t,f) for t in TASKS for f in range(5) for a in m.ARMS],new_ft=[config(a,t,f) for t in TASKS for f in range(5) for a in m.TRAIN_ARMS]))
    inventory=read(PACK/'inventory.json')
    exclusive(OUT/'registry/workload.json',inventory)
    exclusive(OUT/'environment.json',dict(versions=numerical.versions(),time=now(),contract=contract()))


def anchor(task,fold):
    return read(OUT/'registry/anchors.json')['anchors'][f'{task}/{fold}']


def load_base(task,fold):
    source=anchor(task,fold); saved=torch.load(source['best'],map_location='cpu',weights_only=False)
    if saved['schema']!=source['schema'] or saved['architecture']!=m.base.CONFIG or saved['config']!=source['source_config'] or any(saved[k]!=source[k] for k in ('best_epoch','best_stage','scaler')):
        raise ValueError('BASE checkpoint schema/full identity mismatch')
    if any(x.is_floating_point() and (x.dtype!=torch.float32 or not torch.isfinite(x).all()) for x in saved['state_dict'].values()):
        raise ValueError('BASE state finite FP32')
    model=m.base.BaseOptFuse(False,ft=saved['config']['recipe']['ft']); m.base.strict_module_state(model,saved['state_dict']); model.stage=saved['best_stage']
    return m.freeze_base(model),saved['state_dict']


def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('WORLD_SIZE')!='1' or torch.cuda.device_count()!=1:
        raise ValueError('physicalGPU2/world1/one visible card')
    actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual!=os.environ.get('UA3M_PHYSICAL_GPU_UUID'):
        raise ValueError('frozen actual GPU UUID mismatch')
    numerical.check_det()
    return dict(**contract(),actual_uuid=actual,physical_gpu=2)


class Consumption:
    def __init__(self,folder,kind):
        self.folder=folder; self.start=time.time()
        self.value=dict(status='RUNNING',kind=kind,started_unix=self.start,process=process_identity(os.getpid()),attempted={},completed={},**contract()); self.record()
    def add(self,k,n=1,completed=False):
        d_=self.value['completed' if completed else 'attempted']; d_[k]=d_.get(k,0)+int(n)
    def record(self):
        self.value['elapsed_seconds']=time.time()-self.start; write(self.folder/'consumption.json',self.value)
    def finish(self):
        self.value.update(status='COMPLETE',ended_unix=time.time()); self.record()
    def fail(self):
        self.value.update(status='FAILED',ended_unix=time.time(),error=traceback.format_exc()); self.record()


def cache_data(arm,task,fold,formal=False):
    return d.CacheData(OUT/'feature_cache',arm,task,fold,formal,SIDE,PACK,SPLIT)


def source_lock():
    for n in SOURCES:
        if (ROOT/n).read_bytes()!=(OUT/'authorization/source_at_launch'/n).read_bytes():
            raise ValueError('locked implementation changed: '+n)


SOURCES=('src/dataset/ua3d_anchor_r2.py','src/modules/ua3d_anchor_r2.py','src/training/ua3d_anchor_r2.py',
         'scripts/prepare_ua3d_anchor_r2.py','scripts/run_ua3d_anchor_r2.py','configs/ua3d_anchor_r2_gpu2.json','tests/test_ua3d_anchor_r2.py')


def build_model(arm,task,fold,device='cuda:0',saved_adapter=None):
    seed_all(42+fold); base,source_state=load_base(task,fold); model=m.AnchorModel(base,arm)
    if saved_adapter is not None:
        if saved_adapter['schema']!=m.SCHEMA or saved_adapter['config']!=config(arm,task,fold) or saved_adapter['base_anchor']!=anchor(task,fold):
            raise ValueError('adapter checkpoint/source identity mismatch')
        m.base.strict_module_state(model.adapter,saved_adapter['adapter_state'])
    model.to(device); model.assert_frozen()
    # Actual accidental encoder calls fail inside the registered operation, never trigger probes.
    def no_encoder(module,args):
        raise RuntimeError('encoder call forbidden after teacher caching')
    model.base.o8.register_forward_pre_hook(no_encoder); model.base.encoder3d.register_forward_pre_hook(no_encoder)
    return model,source_state


def accounted(model,batch,usage,training=False):
    before=dict(model.cost)
    try:
        return model.training_loss(batch) if training else model.predict(batch)
    finally:
        for k,v in model.cost.items():
            usage.add(k,v-before[k])
            # These counters record completed frozen layers/PH calls only after the operation;
            # the full job reservation remains charged on any failure.
            usage.add(k,v-before[k],True)


def write_environment(folder,physical,setup):
    write(folder/'environment.json',dict(versions=numerical.versions(),physical=physical,setup=setup,numerical=numerical.check_det()))


def diagnose(arm,repeat,setup):
    label,r=repeat.split('_'); task,fold=('eps',0) if label=='S' else ('egc',1)
    require_tmux(); guard(preparation=True); physical=gpu_contract(); folder=job_path('diagnostic',arm,repeat=repeat)
    folder.mkdir(parents=True,exist_ok=False); u=Consumption(folder,'diagnostic'); u.add('diagnostic_starts'); u.record(); ds=None
    try:
        ds=cache_data(arm,task,fold); ids=ds.train[:24]
        expected=read(SPLIT/f'{task}.json')['folds'][fold]['train_indices'][:24]
        if [ds.indices[i] for i in ids]!=expected:
            raise ValueError('fixed cached train diagnostic identity')
        batch=d.collate([ds[i] for i in ids]).to('cuda:0')
        model,source_state=build_model(arm,task,fold); initial=numerical.clone_tree(model.adapter.state_dict())
        write_environment(folder,physical,setup); write(folder/'input_identity.json',dict(rows=expected,keys=batch.keys,task=task,fold=fold))
        model.eval(); eval_times=[]
        for k in range(2):
            u.add('diagnostic_eval_forwards'); u.add('diagnostic_eval_rows',24); u.record()
            torch.cuda.synchronize(); tick=time.perf_counter()
            with torch.no_grad():
                prediction=accounted(model,batch,u)
            torch.cuda.synchronize(); eval_times.append(time.perf_counter()-tick)
            if not torch.equal(prediction,batch.y_base):
                raise ValueError('zero residual must exactly restore cached BASE prediction')
            save_checkpoint(folder/f'eval{k}.pt',dict(prediction=numerical.clone_tree(prediction),anchor=numerical.clone_tree(batch.y_base),
                intermediate=numerical.clone_tree(model.last),rng=numerical.rng_state()))
            u.add('diagnostic_eval_forwards',completed=True); u.add('diagnostic_eval_rows',24,True); u.record()
        save_checkpoint(folder/'initial.pt',dict(adapter=initial,rng=numerical.rng_state()))
        model.train(); opt,sch=m.optimizer(model); seed_all(42+fold); timings=[]; active_grads=set()
        for step in range(1,21):
            guard(preparation=True); model.train(); opt.zero_grad(set_to_none=True)
            u.add('diagnostic_train_forwards'); u.add('diagnostic_train_rows',24); u.record()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick=time.perf_counter()
            loss,parts=accounted(model,batch,u,True)
            if not torch.isfinite(loss):
                raise FloatingPointError('diagnostic loss')
            u.add('diagnostic_train_forwards',completed=True); u.add('diagnostic_train_rows',24,True)
            u.add('diagnostic_backwards'); loss.backward(); u.add('diagnostic_backwards',completed=True)
            model.assert_frozen()
            for name,p in model.adapter.named_parameters():
                inactive=(arm=='P_PE' and name.startswith('filter.'))
                if not inactive and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise ValueError('unexpected disconnected/nonfinite gradient: '+name)
                if p.grad is not None and bool(p.grad.count_nonzero()):
                    active_grads.add(name)
            norm=torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),5.,error_if_nonfinite=True)
            if step in (1,11):
                save_checkpoint(folder/f'grad{step}.pt',{n:numerical.clone_tree(p.grad) for n,p in model.adapter.named_parameters() if p.grad is not None})
            u.add('diagnostic_updates'); u.record(); opt.step(); u.add('diagnostic_updates',completed=True); sch.step()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-tick
            timings.append(dict(step=step,seconds=elapsed,peak_mib=torch.cuda.max_memory_reserved()/1024**2))
            save_checkpoint(folder/f'step{step:02d}.pt',dict(loss=float(loss.detach()),parts={k:float(v) for k,v in parts.items()},
                clip_norm=float(norm),adapter=numerical.clone_tree(model.adapter.state_dict()),rng=numerical.rng_state(),learning_rates=[g['lr'] for g in opt.param_groups],
                intermediates=numerical.clone_tree(model.last)))
            u.record()
        m.verify_unchanged(model,source_state)
        changed=[n for n,p in model.adapter.state_dict().items() if not torch.equal(p.detach().cpu(),initial[n])]
        if not changed or not any(n.startswith('out.' if arm!='G_GEOM' else 'up.') for n in changed):
            raise ValueError('adapter failed to learn within20steps')
        if arm=='P_LEARN' and not any(n.startswith('filter.') for n in changed):
            raise ValueError('learned filtration did not update within20steps')
        write(folder/'gradient_contract.json',dict(base_frozen=True,base_eval=True,base_unchanged=True,active_grad_parameters=sorted(active_grads),changed_parameters=changed,
             expected_inactive='filter' if arm=='P_PE' else 'router PH input columns' if arm=='G_GEOM' else None))
        write(folder/'timing.json',dict(train=timings,eval=eval_times))
        u.add('diagnostic_starts',completed=True); u.finish()
    except BaseException:
        u.fail(); raise
    finally:
        if ds is not None:
            ds.close()


def compare_diagnostics():
    checks={}; result=dict(pass_=False,checks=checks)
    for label in ('S','L'):
        for arm in m.TRAIN_ARMS:
            paths=[job_path('diagnostic',arm,repeat=label+'_'+r) for r in ('R0','R1')]
            for name in ['eval0.pt','eval1.pt','initial.pt','grad1.pt','grad11.pt']+[f'step{i:02d}.pt' for i in range(1,21)]:
                a,b=[torch.load(p/name,map_location='cpu',weights_only=False) for p in paths]
                value=numerical.exact_compare(a,b); checks[label+'/'+arm+'/'+name]=value
                if not value['equal']:
                    result['pass']=result.pop('pass_'); write(OUT/'diagnostics/comparison.json',result)
                    raise ValueError('diagnostic exact reproducibility failed: '+label+'/'+arm+'/'+name)
            for path in paths:
                a=torch.load(path/'eval0.pt',map_location='cpu',weights_only=False); b=torch.load(path/'eval1.pt',map_location='cpu',weights_only=False)
                value=numerical.exact_compare(a,b); checks[str(path)+'/intra']=value
                if not value['equal']:
                    result['pass']=result.pop('pass_'); write(OUT/'diagnostics/comparison.json',result); raise ValueError('intra-eval reproducibility')
    result['pass_']=True; result['pass']=result.pop('pass_'); exclusive(OUT/'diagnostics/comparison.json',result)


def evaluate(model,ds,usage,formal=False):
    model.eval(); y=[]; predictions=[]; base=[]; key='test' if formal else 'validation'
    for batch in d.loader(ds,ds.validation):
        guard(); count=len(batch.y); usage.add('adapter_test_forwards' if formal else 'validation_forwards')
        usage.add('test_rows' if formal else 'validation_rows',count); usage.record()
        batch=batch.to('cuda:0')
        with torch.no_grad():
            p=accounted(model,batch,usage)
        y.extend(batch.y.cpu().numpy().tolist()); predictions.extend(p.cpu().numpy().tolist()); base.extend(batch.y_base.cpu().numpy().tolist())
        usage.add('adapter_test_forwards' if formal else 'validation_forwards',completed=True)
        usage.add('test_rows' if formal else 'validation_rows',count,True)
        if formal:
            usage.add('test_batches'); usage.add('test_batches',completed=True)
    truth=transform(np.asarray(y),ds.scaler,True); prediction=transform(np.asarray(predictions),ds.scaler,True)
    return truth,prediction,metrics(truth,prediction)


def save_predictions(p,ds,y,prediction,**extra):
    np.savez(p,indices=np.asarray(ds.indices)[ds.validation],sample_keys=ds.keys[ds.validation],y_true=y,y_pred=prediction,**extra)


def unit(arm,task,fold,formal,setup):
    require_tmux(); guard()
    if arm=='C_ANCHOR':
        return anchor_output(task,fold)
    physical=gpu_contract()
    folder=job_path('test' if formal else 'ft',arm,task,fold); folder.mkdir(parents=True,exist_ok=False)
    usage=Consumption(folder,'test' if formal else 'ft'); key='test_starts' if formal else 'ft_starts'; usage.add(key); usage.record(); ds=None
    try:
        cfg=config(arm,task,fold); write(folder/'config.json',cfg); saved=None
        if formal:
            frozen=read(OUT/'selection/final_frozen.json'); selected=frozen['units'][f'{arm}/{task}/{fold}']
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if selected['config']!=cfg or saved['best_epoch']!=selected['best_epoch'] or saved['scaler']!=selected['scaler']:
                raise ValueError('frozen adapter identity')
        model,source_state=build_model(arm,task,fold,saved_adapter=saved)
        write_environment(folder,physical,setup); ds=cache_data(arm,task,fold,formal)
        if formal:
            truth,prediction,result=evaluate(model,ds,usage,True); best_epoch=saved['best_epoch']
            with np.load(job_path('test','C_ANCHOR',task,fold)/'test_predictions.npz',allow_pickle=False) as ref:
                if not np.array_equal(ref['indices'],np.asarray(ds.indices)) or not np.array_equal(ref['sample_keys'],ds.keys) or not np.array_equal(ref['y_true'],truth):
                    raise ValueError('formal adapter identity differs from C_ANCHOR')
            save_predictions(folder/'test_predictions.npz',ds,truth,prediction)
        else:
            train=d.loader(ds,ds.train,True,42+fold); seed_all(42+fold); opt,sch=m.optimizer(model); history=[]; best_rmse=float('inf')
            (folder/'validation').mkdir()
            for epoch in range(1,71):
                guard(); tick=time.time(); model.train(); usage.add('ft_epochs'); usage.record(); losses=[]; rates=[g['lr'] for g in opt.param_groups]
                for batch in train:
                    guard(); usage.add('ft_train_forwards'); usage.add('ft_train_rows',len(batch.y)); usage.record()
                    opt.zero_grad(set_to_none=True); batch=batch.to('cuda:0'); loss,parts=accounted(model,batch,usage,True)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('adapter FT loss')
                    loss.backward(); model.assert_frozen(); torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),5.,error_if_nonfinite=True)
                    usage.add('ft_updates'); usage.record(); opt.step()
                    usage.add('ft_updates',completed=True); usage.add('ft_train_forwards',completed=True); usage.add('ft_train_rows',len(batch.y),True); losses.append(float(loss.detach()))
                sch.step(); usage.add('validation_count'); truth,prediction,met=evaluate(model,ds,usage)
                save_predictions(folder/'validation'/f'epoch{epoch:03d}.npz',ds,truth,prediction,epoch=epoch)
                if met['rmse']<best_rmse:
                    best_rmse=met['rmse']; best_epoch=epoch; result=met
                    save_checkpoint(folder/'best.pt',dict(schema=m.SCHEMA,config=cfg,adapter_state=numerical.clone_tree(model.adapter.state_dict()),
                        base_anchor=anchor(task,fold),scaler=ds.scaler,best_epoch=epoch,outer_test='NOT_RUN'))
                    save_predictions(folder/'validation_predictions.npz',ds,truth,prediction,best_epoch=epoch)
                history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=met,learning_rates=rates,seconds=time.time()-tick,
                    original_parameters_frozen=True,original_modules_eval=True))
                write(folder/'history.json',history); usage.add('ft_epochs',completed=True); usage.add('validation_count',completed=True); usage.record()
                print(json.dumps(dict(arm=arm,task=task,fold=fold,epoch=epoch,validation=met)),flush=True)
            m.verify_unchanged(model,source_state)
            if usage.value['completed'].get('ft_updates')!=70*len(train):
                raise ValueError('adapter FT update count')
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,scaler=ds.scaler,**result))
        usage.add(key,completed=True); usage.finish()
    except BaseException:
        usage.fail(); raise
    finally:
        if ds is not None:
            ds.close()


def anchor_output(task,fold):
    require_tmux(); guard()
    if not read(OUT/'selection/final_frozen.json')['frozen']:
        raise ValueError('anchor output before freeze')
    folder=job_path('test','C_ANCHOR',task,fold); folder.mkdir(parents=True,exist_ok=False)
    u=Consumption(folder,'test'); u.add('test_starts'); u.record()
    try:
        cache=job_path('cache_test',task=task,fold=fold)
        meta=read(cache/'metadata.json')
        if meta['status']!='COMPLETE' or meta['scaler']!=anchor(task,fold)['scaler']:
            raise ValueError('C_ANCHOR teacher source/scaler')
        with np.load(cache/'test_predictions.npz',allow_pickle=False) as p:
            values={k:p[k].copy() for k in p.files}
        count=len(values['indices']); batches=math.ceil(count/24)
        u.add('test_rows',count); u.add('test_batches',batches); u.add('anchor_output_batches',batches); u.record()
        with np.load(BASE/'formal/F_PD_N1_F0'/task/f'fold{fold}/test_predictions.npz',allow_pickle=False) as ref:
            for k in ('indices','sample_keys','y_true'):
                if not np.array_equal(values[k],ref[k]):
                    raise ValueError('C_ANCHOR original TEST identity: '+k)
            if not np.allclose(values['y_pred'],ref['y_pred'],atol=SPEC['prediction_atol'],rtol=SPEC['prediction_rtol']):
                raise ValueError('C_ANCHOR original TEST prediction tolerance')
        np.savez(folder/'test_predictions.npz',**values)
        write(folder/'metrics.json',dict(status='COMPLETE',config=config('C_ANCHOR',task,fold),scaler=meta['scaler'],best_epoch=anchor(task,fold)['best_epoch'],**metrics(values['y_true'],values['y_pred'])))
        for k,v in [('test_starts',1),('test_rows',count),('test_batches',batches),('anchor_output_batches',batches)]:
            u.add(k,v,True)
        u.finish()
    except BaseException:
        u.fail(); raise


def aggregate(arm,formal=False):
    tasks={}; units={}
    for task in TASKS:
        scores=[]
        for fold in range(5):
            if not formal and arm=='C_ANCHOR':
                path=job_path('cache_trainval',task=task,fold=fold)
                gate=read(path/'validation_gate.json'); met=gate['new_metrics']; cfg=config(arm,task,fold)
                best_epoch=anchor(task,fold)['best_epoch']; scaler=anchor(task,fold)['scaler']
            else:
                path=job_path('test' if formal else 'ft',arm,task,fold); met=read(path/'metrics.json'); u=read(path/'consumption.json')
                if met['status']!='COMPLETE' or u['status']!='COMPLETE' or met['config']!=config(arm,task,fold):
                    raise ValueError('incomplete estimator')
                split=read(SPLIT/f'{task}.json')['folds'][fold]
                counts=dict(test_starts=1,test_rows=len(split['test_indices']),test_batches=math.ceil(len(split['test_indices'])/24)) if formal else dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(split['train_indices'])/24),validation_count=70)
                if any(u['completed'].get(k)!=v for k,v in counts.items()):
                    raise ValueError('estimator consumption mismatch')
                cfg=met['config']; best_epoch=met['best_epoch']; scaler=met['scaler']
            scores.append(met['r2']); units[f'{arm}/{task}/{fold}']=dict(config=cfg,best=str(base_folder(task,fold)/'best.pt') if arm=='C_ANCHOR' else str(job_path('ft',arm,task,fold)/'best.pt'),
                best_epoch=best_epoch,scaler=scaler,base_anchor=anchor(task,fold),kind='original_BASE' if arm=='C_ANCHOR' else 'adapter')
        tasks[task]=dict(mean=float(np.mean(scores)),std=float(np.std(scores,ddof=1)),folds=scores)
    return dict(macro=float(np.mean([v['mean'] for v in tasks.values()])),tasks=tasks,units=units)


def select_final(validation):
    baseline=validation['C_ANCHOR']; eligible=[]
    for arm in m.CANDIDATES:
        score=validation[arm]
        if score['macro']<=baseline['macro'] or any(score['tasks'][task]['mean']<baseline['tasks'][task]['mean']-.005 for task in TASKS):
            continue
        if arm=='P_LEARN' and score['macro']<=validation['P_PE']['macro']:
            continue
        eligible.append(arm)
    return max(eligible,key=lambda a:(validation[a]['macro'],-m.CANDIDATES.index(a))) if eligible else 'BASE_KEEP'


def freeze():
    if (OUT/'feature_cache/test').exists():
        raise ValueError('TEST feature cache existed before freeze')
    if not read(OUT/'diagnostics/comparison.json')['pass'] or not read(OUT/'gates/pilot.json')['pass']:
        raise ValueError('unpassed gates')
    validation={arm:aggregate(arm) for arm in m.ARMS}; final=select_final(validation)
    units={key:u for v in validation.values() for key,u in v['units'].items()}
    if len(units)!=140:
        raise ValueError('140 frozen estimator identities required')
    rows=read(INPUT/'rows.json')
    for key,u in units.items():
        cfg=u['config']; task=cfg['task']; fold=cfg['fold']; e=read(SPLIT/f'{task}.json')['folds'][fold]
        u['test_indices']=sorted(e['test_indices']); mapping={r['row']:r['key'] for r in rows[task]}; u['test_keys']=[mapping[i] for i in u['test_indices']]
        if cfg['arm']!='C_ANCHOR':
            saved=torch.load(u['best'],map_location='cpu',weights_only=False)
            if saved['schema']!=m.SCHEMA or saved['config']!=cfg or saved['base_anchor']!=u['base_anchor'] or saved['scaler']!=u['scaler'] or saved['best_epoch']!=u['best_epoch']:
                raise ValueError('freeze adapter BEST identity')
    exclusive(OUT/'selection/final_frozen.json',dict(**contract(),schema='ua3d-anchor-freeze-r2',frozen=True,time=now(),time_unix=time.time(),
        FINAL=final,validation=validation,units=units,reference=REFERENCE,selection_rule='A7 global validation only',test_accessed=False,
        teacher_cache_schema=d.SCHEMA,source_result=str(BASE/'final_review/results.json')))


def anchor_test_gate():
    anchor_metrics=aggregate('C_ANCHOR',True)
    delta=anchor_metrics['macro']-REFERENCE
    value=dict(pass_=abs(delta)<=SPEC['r2_tolerance'],macro=anchor_metrics['macro'],reference=REFERENCE,delta=delta,
        identity_and_prediction_tolerance='all35 C_ANCHOR output units passed',reference_replaced=False)
    value['pass']=value.pop('pass_'); exclusive(OUT/'formal/anchor_gate.json',value)
    if not value['pass']:
        raise ValueError('C_ANCHOR TEST macro differs from fixed original reference')


def profiles():
    result={}
    for label in ('S','L'):
        result[label]={}
        for arm in m.TRAIN_ARMS:
            timing=[read(job_path('diagnostic',arm,repeat=label+'_'+r)/'timing.json') for r in ('R0','R1')]
            # Exclude only the separately recorded first step; all later steps remain measured.
            result[label][arm]=dict(train=max(float(np.median([x['seconds'] for x in v['train'] if x['step']>1])) for v in timing),
                eval=max(v['eval'][1] for v in timing),cold_eval=max(v['eval'][0] for v in timing),peak_mib=max(x['peak_mib'] for v in timing for x in v['train']))
    return result


def cost_gate(jobs,after_pilot=False):
    inventory=read(OUT/'registry/workload.json'); measured=profiles(); estimates=[]; unknown=[]; completed={(j['arm'],j['task'],j['fold']) for j in jobs if j['kind']=='ft' and j['exit']==0}
    pilot_factor=1.
    def work_ratio(arm,task,ids,label):
        reference=inventory['diagnostic_inputs'][label]['work']; total=d.packed.sum_work([inventory['rows'][task][str(i)] for i in ids])
        columns=['rows','bonds','spatial_edges'] if arm=='G_GEOM' else ['rows','nodes']+(['learned_work','learned_membership_bound'] if arm=='P_LEARN' else [])
        values=[]
        for k in columns:
            if reference[k]==0 and total[k]>0:
                unknown.append(dict(arm=arm,task=task,column=k)); return None
            values.append(total[k]/max(reference[k],1))
        return max(values)
    cache_jobs={(j['task'],j['fold']):j for j in jobs if j['kind']=='cache_trainval' and j['exit']==0}
    def ft_estimate(arm,task,fold):
        e=read(SPLIT/f'{task}.json')['folds'][fold]; candidates=[]
        for label in ('S','L'):
            train=work_ratio(arm,task,e['train_indices'],label); val=work_ratio(arm,task,e['validation_indices'],label)
            if train is None or val is None:
                return None
            candidates.append(70*(measured[label][arm]['train']*train+measured[label][arm]['eval']*val))
        cache=read(cache_jobs[(task,fold)]['consumption']); rows=len(e['train_indices'])+len(e['validation_indices'])
        io_allowance=70*cache['elapsed_seconds']
        return max(candidates)+io_allowance+60.
    if after_pilot:
        pilot=next(j for j in jobs if j['kind']=='ft' and j['arm']=='P_LEARN' and j['task']=='egc' and j['fold']==1)
        pilot_factor=max(1.,(pilot['ended_unix']-pilot['started_unix'])/ft_estimate('P_LEARN','egc',1))
    for task in TASKS:
        for fold in range(5):
            e=read(SPLIT/f'{task}.json')['folds'][fold]; cache=read(cache_jobs[(task,fold)]['consumption'])
            trainval_batches=cache['attempted']['cache_trainval_batches']; test_batches=math.ceil(len(e['test_indices'])/24)
            estimates.append(dict(kind='cache_test',arm=None,task=task,fold=fold,seconds=cache['elapsed_seconds']*test_batches/trainval_batches+30.))
            for arm in m.TRAIN_ARMS:
                if (arm,task,fold) not in completed:
                    value=ft_estimate(arm,task,fold)
                    estimates.append(dict(kind='ft',arm=arm,task=task,fold=fold,seconds=None if value is None else value*(pilot_factor if arm=='P_LEARN' else 1.)))
            for arm in m.ARMS:
                if arm=='C_ANCHOR':
                    value=5.
                else:
                    ratios=[work_ratio(arm,task,e['test_indices'],lab) for lab in ('S','L')]
                    value=None if any(r is None for r in ratios) else max(measured[lab][arm]['eval']*r for lab,r in zip(('S','L'),ratios))+30.+cache['elapsed_seconds']*test_batches/trainval_batches
                estimates.append(dict(kind='test',arm=arm,task=task,fold=fold,seconds=value))
    memory=read(OUT/'registry/gpu.json'); peak=max(p['peak_mib'] for x in measured.values() for p in x.values())
    # Strict12GiB route floors and whole-card2GiB reserve also constrain ETA concurrency.
    per_route=max(12288.,peak+2048.); routes=min(2,max(0,int((memory['total_mib']-2048)//per_route)))
    raw=sum(v['seconds'] or 0 for v in estimates if v['kind']=='cache_test')+sum(v['seconds'] or 0 for v in estimates if v['kind']!='cache_test')/max(routes,1)+1800.+900.
    remaining=DEADLINE-time.time(); estimate=raw*1.5
    value=dict(pass_=not unknown and routes>0 and estimate<remaining,estimated_seconds_with_margin=estimate,remaining_seconds=remaining,
        effective_routes=routes,eta_margin=1.5,per_unit=estimates,profiles=measured,unknown_workloads=unknown,pilot_factor=pilot_factor,
        cache_time_allowance='per-fold measured trainval cache time per epoch conservatively includes IO/encoder/loading; TEST cache extrapolated separately',
        memory_route_mib=per_route,after_pilot=after_pilot,uncertainty='structure-scaled measured costs, no4-route promise; not a completion guarantee')
    value['pass']=value.pop('pass_'); exclusive(OUT/'gates'/('cost_after_pilot.json' if after_pilot else 'cost.json'),value)
    if not value['pass']:
        raise RuntimeError('conservative full-matrix ETA/coverage gate failed')
    return value


def pilot_gate(jobs):
    pilot=next(j for j in jobs if j['kind']=='ft' and j['arm']=='P_LEARN' and j['task']=='egc' and j['fold']==1)
    elapsed=pilot['ended_unix']-pilot['started_unix']; usage=read(pilot['consumption'])
    value=dict(pass_=False,wrapper_elapsed_seconds=elapsed,limit_seconds=3600,exit=pilot['exit'],epochs=usage['completed'].get('ft_epochs'))
    if pilot['exit']!=0 or value['epochs']!=70 or elapsed>3600:
        value['pass']=value.pop('pass_'); exclusive(OUT/'gates/pilot.json',value); raise RuntimeError('pilot70epochs/exit0/3600s gate failed')
    try:
        value['pass_']=cost_gate(jobs,True)['pass']
    finally:
        value['pass']=value.pop('pass_'); exclusive(OUT/'gates/pilot.json',value)


def report(jobs,error):
    validation={}; formal={}
    for arm in m.ARMS:
        for dest,flag in ((validation,False),(formal,True)):
            count=sum(j['kind']==('test' if flag else 'cache_trainval' if arm=='C_ANCHOR' else 'ft') and (not flag and arm=='C_ANCHOR' or j['arm']==arm) and j['exit']==0 for j in jobs)
            if count==35:
                dest[arm]=aggregate(arm,flag)
    frozen=read(OUT/'selection/final_frozen.json') if (OUT/'selection/final_frozen.json').exists() else None
    final=frozen['FINAL'] if frozen else None; complete=len(formal)==4
    selected=REFERENCE if complete and final=='BASE_KEEP' else formal[final]['macro'] if complete and final in formal else None
    delta=selected-REFERENCE if selected is not None else None
    cpu=next((j['exit'] for j in jobs if j['kind']=='cpu'),None)
    preflight=next((j['exit'] for j in jobs if j['kind']=='preflight'),None)
    result=dict(**contract(),status='STOPPED' if error else 'COMPLETE_PENDING_REVIEW',error=error,independent_review='PENDING',FINAL=final,
        reference=REFERENCE,selected_macro=selected,selected_delta=delta,performance_pass=(final!='BASE_KEEP' and delta>=.010) if delta is not None else None,
        validation=validation,formal=formal,formal_test_complete=complete,ft_complete=len(validation)==4,execution_complete=complete and cpu==0 and error is None,
        preflight_cpu=dict(attempts=sum(j['kind']=='preflight' for j in jobs),exit=preflight),final_cpu=dict(attempts=sum(j['kind']=='cpu' for j in jobs),exit=cpu),
        stage_facts=read(OUT/'execution_manifest.json')['stages'],evaluation='shared development; previous exposed TEST labels/results retained; one training seed42+fold',
        metrics_status='完整正式TEST' if complete else '未完成/不可比较',sota={},deltas={})
    if complete:
        for arm in m.ARMS:
            result['deltas'][arm+'-original_BASE']=formal[arm]['macro']-REFERENCE
            result['deltas'][arm+'-C_ANCHOR']=formal[arm]['macro']-formal['C_ANCHOR']['macro']
        result['deltas']['P_LEARN-P_PE']=formal['P_LEARN']['macro']-formal['P_PE']['macro']
        std=dict(eea=.013,egb=.009,egc=.008,ei=.018,eps=.033,nc=.032,xc=.102)
        for arm in m.ARMS:
            result['sota'][arm]={task:dict(project=formal[arm]['tasks'][task],paper_mean=SOTA[task][2],paper_std=std[task],paper_model=SOTA[task][1],delta=formal[arm]['tasks'][task]['mean']-SOTA[task][2]) for task in TASKS}
    write(OUT/'final_review/results.json',result)
    lines=['# BASE ANCHOR r2 交付待审查','',f'状态：{result["status"]}；{result["metrics_status"]}；independent_review=PENDING。',
        f'前置CPU次数/exit：{result["preflight_cpu"]!r}；最终CPU次数/exit：{result["final_cpu"]!r}。',
        f'FINAL={final!r}；selected_macro={selected!r}；selected_delta={delta!r}；reference={REFERENCE!r}。',
        '原BASE35套下游BEST只读冻结，原模块eval；同批U/H差值锚定；旧周期/BASE/旧FINAL不改写。',
        '共享开发评价、旧TEST标签暴露和已看TEST路线开发限制持续披露，不称独立盲测。','',
        '| Arm | TEST macro | Δ原BASE |','|---|---:|---:|']
    for arm in m.ARMS:
        v=formal.get(arm); lines.append(f'|{arm}|'+(f'{v["macro"]:.3f}|{v["macro"]-REFERENCE:+.3f}|' if v else '未完成/不可比较|—|'))
    for arm,values in result['sota'].items():
        lines+=['',f'## {arm}','', '|任务|项目mean±std|论文mean±std|差值|论文模型|','|---|---:|---:|---:|---|']
        for task,v in values.items():
            lines.append(f'|{task}|{v["project"]["mean"]:.3f}±{v["project"]["std"]:.3f}|{v["paper_mean"]:.3f}±{v["paper_std"]:.3f}|{v["delta"]:+.3f}|{v["paper_model"]}|')
    lines+=['','论文为逐任务最佳包络，不是单一模型；数据/split/统计未证实等价，仅数值对照。Egc3380/论文4124，Xc432/论文430差异保留。',
        '阻断：'+(error or '无')+'。','不自动修复重试/补单元，不扩时，不替换BASE，不自动开启下一轮。']
    (OUT/'delivery_summary.md').write_text('\n'.join(lines)+'\n'); (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    return result
