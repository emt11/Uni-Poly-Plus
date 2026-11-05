"""Independent PH r2 jobs; no redirection of any historical trainer/controller."""
import datetime
import json
import math
import os
import random
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from src.dataset import ua3d_ph_r2 as d
from src.modules import ua3d_ph_r2 as m
from src.modules import ua3d_stability as numerical
from src.training.glt_dual_runtime import save_checkpoint, require_tmux
from src.training.ua2d_p10k_nofp_align import metrics, process_identity, process_alive, SOTA
from src.training.ua2d_a020_fp_fusion import scaler_validate, transform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ua3d_ph_20261028/r1/repairs/gpu2_r2'
LOG = ROOT / 'logs/ua3d_ph_20261028/r1/repairs/gpu2_r2'
REF = ROOT / 'results/ua3d_base_opt_20261026/r6/repairs/concurrency_r1'
PARENT = ROOT / 'results/ua3m_paperfuse_20261025/r1/repairs/gpu0123_restart1'
DOWN = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT / 'data/splits/mips_paper5_outer'
SCALERS = ROOT / 'results/ua2d_p10k_nofp_align_20261024/r1/finetune/D_A020'
PLAN = 'UA3D-PH-20261028-01'; REVISION = 'r2'; RESOURCE = 'gpu2-pt2-fttest4-ph-r2'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
T0 = 1793199774.008285; DEADLINE = 1793372574.008285; REFERENCE = .804675462276949
UUID = 'GPU-31456dd5-a421-142b-644b-7a80f482dfb9'
FIXED_ROWS = [353,267,250,91,219,73,112,10,220,37,29,326,70,345,136,159,297,251,257,203,315,102,82,135]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); os.replace(tmp, p)


def exclusive(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def contract():
    return dict(plan=PLAN, revision=REVISION, resource_revision=RESOURCE, physical_gpus=[2], world_size=1,
                logical_device='cuda:0', caps=dict(pt=2, diagnostic=1, ft=4, test=4), phase_exclusive=True,
                pt_starts=0, pt_updates=0)


def guard_deadline(reserve=900):
    if time.time() >= DEADLINE - reserve:
        raise TimeoutError('inherited hard deadline CPU/delivery reserve reached')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def gpu_for(task, fold):
    if task not in TASKS or type(fold) is not int or fold not in range(5):
        raise ValueError('task/fold matrix')
    return 2


def job_path(kind, arm=None, task=None, fold=None, repeat=None):
    if kind == 'diagnostic':
        return OUT / 'diagnostics' / arm / 'gpu2' / repeat
    if kind in ('ft', 'test'):
        return OUT / ('formal' if kind == 'test' else 'finetune') / arm / task / f'fold{fold}'
    return OUT / ('preflight_cpu' if kind == 'preflight' else 'final_cpu' if kind == 'cpu' else 'inputs')


def recipe():
    r = read(REF / 'registry/recipes.json')['recipes']['F_PD_N1_F0']
    if r['ft'] != m.base.DEFAULT_FT or r['pt'] != m.base.pt_config('P_CURRENT_SHARED', .1, 0.) or r['pt_id'] != 'F_PD_N1_F0':
        raise ValueError('original F recipe changed')
    return r


def config(arm, task, fold):
    if arm not in m.ARMS:
        raise ValueError('arm matrix')
    return dict(**contract(), arm=arm, task=task, fold=fold, seed=42+fold, physical_gpu=gpu_for(task, fold),
                batch=24, epochs=70, head_epochs=10, precision='FP32', workers=0,
                recipe=recipe(), parent=str(REF / 'pretrain/F_PD_N1_F0/deploy.pt'),
                head=str(PARENT / 'initialization/heads' / task / f'fold{fold}.pt'),
                scaler_source=str(SCALERS / task / f'fold{fold}/scaler.json'), input_cache=str(PARENT / 'inputs'),
                sidecar=str(OUT / 'inputs'), numerical_policy='DET', selection='strict original-unit validation RMSE; earliest exact tie',
                inactive_parameters='filter' if arm in ('P_PE','P_FIXED') else 'router' if arm=='M_EQ' else None)


def load_base(arm, task, fold):
    cfg = config(arm, task, fold)
    package = torch.load(cfg['parent'], map_location='cpu', weights_only=False)
    head = torch.load(cfg['head'], map_location='cpu', weights_only=False)
    model = m.base.initialize_ft(package, head, cfg['recipe'])
    return m.attach_adapter(model, arm)


def check_prepared():
    meta = read(OUT / 'inputs/metadata.json')
    if meta.get('schema') != d.SCHEMA or meta.get('status') != 'COMPLETE' or tuple(meta[k] for k in ('records','geometry_available','geometry_missing')) != (5875,5860,15):
        raise ValueError('PH sidecar incomplete')
    registered = read(OUT / 'registry/recipes.json')
    if any(read(SPLIT / f'{task}.json') != registered['splits'][task] for task in TASKS):
        raise ValueError('split changed after registration')


def stability_gate():
    s = ROOT / 'results/ua3d_stability_20261028/r1'
    if read(s / 'controller_exit.json')['exit'] != 0 or read(s / 'cpu_exit.json')['exit'] != 0:
        raise ValueError('STABILITY real exit gate failed')
    parent_jobs=read(s/'budget.json')['jobs']
    pairs = []
    for task in TASKS:
        for fold in range(5):
            p = s / 'reproducibility' / task / f'fold{fold}.json'; value = read(p)
            if value.get('equal') is not True or len(value.get('independent_processes', [])) != 2 or value['independent_processes'][0] == value['independent_processes'][1]:
                raise ValueError('STABILITY independent same-seed pair')
            for run in ('S42_A','S42_B'):
                js=[j for j in parent_jobs if j['kind']=='ft' and j.get('arm')==run and j.get('task')==task and j.get('fold')==fold]
                if len(js)!=1 or js[0]['exit']!=0:
                    raise ValueError('STABILITY FT true exit missing/nonzero')
                u = read(s / 'finetune/DET_BASE' / run / task / f'fold{fold}/consumption.json')
                if u['status'] != 'COMPLETE' or u['completed_epochs'] != 70 or u['completed_updates'] != u['attempted_updates']:
                    raise ValueError('STABILITY incomplete training evidence')
                if not (s / 'finetune/DET_BASE' / run / task / f'fold{fold}/last.pt').is_file():
                    raise ValueError('STABILITY last state missing')
            pairs.append(str(p))
    return dict(equal=True, pairs=pairs, count=35, controller_exit=0, cpu_exit=0, new_test_scores_read=False)


def register():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only registration')
    if read(OUT / 'execution_manifest.json')['status'] != 'STOPPED':
        raise ValueError('prior r2 failure must remain STOPPED')
    source = recipe(); splits = {task:read(SPLIT / f'{task}.json') for task in TASKS}
    original=read(REF/'final_review/results.json')
    if original['formal']['F_PD_N1_F0']['macro']!=REFERENCE or original['FINAL']!='E_P_SYM_SHARED' or original['performance_pass'] is not False:
        raise ValueError('original fixed reference/parent outcome differs')
    sums = dict(train=0, validation=0, test=0, train_batches=0, validation_batches=0, test_batches=0)
    for split in splits.values():
        for e in split['folds']:
            for role in ('train','validation','test'):
                n = len(e[role+'_indices']); sums[role] += n; sums[role+'_batches'] += math.ceil(n/24)
    if sums != dict(train=18786,validation=4714,test=5875,train_batches=800,validation_batches=210,test_batches=270):
        raise ValueError('registered split/batch budget mismatch')
    if splits['eps']['folds'][0]['train_indices'][:24] != FIXED_ROWS:
        raise ValueError('diagnostic fixed train rows changed')
    gate = stability_gate(); exclusive(OUT / 'registry/stability_gate.json', gate)
    rows = read(PARENT / 'inputs/rows.json')
    if sum(map(len, rows.values())) != 5875:
        raise ValueError('source row matrix')
    exclusive(OUT / 'registry/recipes.json', dict(**contract(), recipe=source, splits=splits, rows=rows, split_counts=sums,
              source_pt=str(REF / 'pretrain/F_PD_N1_F0/deploy.pt'), reference_macro=REFERENCE,
              performance_threshold=.814675462276949, diagnostic_rows=FIXED_ROWS,
              parent_accounting=str(OUT/'parent_accounting.json'), evaluation='shared development; parent TEST label exposure'))
    exclusive(OUT / 'environment.continuation01.json', dict(versions=numerical.versions(), cwd=str(ROOT), time=now()))


def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('WORLD_SIZE') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('single physical GPU2 / WORLD_SIZE=1 required')
    actual = 'GPU-' + str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if actual != UUID or os.environ.get('UA3M_PHYSICAL_GPU_UUID') != actual:
        raise ValueError('GPU2 UUID changed')
    numerical.check_det()
    return dict(**contract(), actual_uuid=actual, physical_gpu=2)


def verify_runtime(model, cfg, evaluation=False):
    numerical.check_det()
    if cfg['recipe']['ft'] != m.base.DEFAULT_FT:
        raise ValueError('FT science changed')
    for name, expected in (('o8',0.),('encoder3d',0.),('fp_encoder',.2),('head',.5)):
        modules = [x for x in getattr(model, name).modules() if isinstance(x, torch.nn.Dropout)]
        if not modules or any(x.p != expected for x in modules):
            raise ValueError('actual dropout changed: '+name)
    if evaluation and any(x.training for x in model.modules()):
        raise ValueError('evaluation mode')
    if any(x.is_floating_point() and x.dtype != torch.float32 for x in model.state_dict().values()):
        raise ValueError('non-FP32 model')


def schedule(model, stage):
    opt, groups = m.base.optimizer_ft(model, stage)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=3e-5) if stage=='head'
           else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-6))
    return opt, sch, groups


def save_predictions(path, ds, truth, prediction, **metadata):
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError('saved predictions')
    np.savez(path, y_true=truth, y_pred=prediction, indices=np.asarray(ds.indices)[ds.validation], sample_keys=ds.keys[ds.validation],
             availability=np.asarray([[True,ds.cache.mapping[ds.task][ds.indices[i]]['available'],True] for i in ds.validation]), **metadata)


class Consumption:
    def __init__(self, folder, kind):
        self.folder=folder; self.started=time.time()
        self.value=dict(status='RUNNING', kind=kind, process=process_identity(os.getpid()), started_unix=self.started,
                        attempted={}, completed={}, **contract())
        self.record()
    def add(self, key, amount=1, completed=False):
        dest=self.value['completed' if completed else 'attempted']; dest[key]=dest.get(key,0)+amount
    def record(self):
        self.value['elapsed_seconds']=time.time()-self.started; write(self.folder/'consumption.json',self.value)
    def finish(self):
        self.value.update(status='COMPLETE', ended_unix=time.time()); self.record()
    def fail(self):
        self.value.update(status='FAILED',ended_unix=time.time(),error=traceback.format_exc()); self.record()


def reductions(model):
    return model.graph_ph.reductions if hasattr(model,'graph_ph') else 0


def evaluate(model, ds, loader, scaler, usage, formal):
    model.eval(); verify_runtime(model,config(model.ph_arm,ds.task,ds.fold),True)
    y=[]; predictions=[]; key='test' if formal else 'validation'
    with torch.inference_mode():
        for batch in loader:
            guard_deadline(); usage.add(key+'_forwards'); usage.add(key+'_rows',len(batch.y)); usage.record()
            before=reductions(model); batch=batch.to('cuda:0')
            try:
                p,_,_=model.predict(batch)
            finally:
                usage.add('p_learn_reductions',reductions(model)-before)
            if not torch.isfinite(p).all():
                raise FloatingPointError('evaluation predictions')
            usage.add(key+'_forwards',completed=True); usage.add(key+'_rows',len(p),True)
            y.append(batch.y.cpu().numpy()); predictions.append(p.cpu().numpy())
    truth=transform(np.concatenate(y),scaler,True); prediction=transform(np.concatenate(predictions),scaler,True)
    return truth,prediction,metrics(truth,prediction)


def unit(arm,task,fold,formal=False,setup=None):
    require_tmux(); guard_deadline(); check_prepared(); physical=gpu_contract(); cfg=config(arm,task,fold)
    folder=job_path('test' if formal else 'ft',arm,task,fold); folder.mkdir(parents=True,exist_ok=False)
    usage=Consumption(folder,'test' if formal else 'ft'); usage.add('test_starts' if formal else 'ft_starts'); usage.record()
    write(folder/'config.json',cfg); ds=None; model=None
    try:
        scaler=read(SCALERS/task/f'fold{fold}/scaler.json')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json'); selected=frozen['units'][f'{arm}/{task}/{fold}']
            if not frozen['frozen'] or selected['config']!=cfg:
                raise ValueError('formal unit is not frozen')
            saved=torch.load(selected['best'],map_location='cpu',weights_only=False)
            if saved['schema']!=m.SCHEMA or saved['config']!=cfg or saved['scaler']!=scaler or any(saved[k]!=selected[k] for k in ('best_epoch','best_stage')):
                raise ValueError('frozen BEST/checkpoint mismatch')
            model=m.PHFuse(arm,cfg['recipe']['ft']); m.base.strict_module_state(model,saved['state_dict']); model.stage=saved['best_stage']
            best_epoch=saved['best_epoch']; best_stage=saved['best_stage']; del saved
        else:
            seed_all(cfg['seed']); model=load_base(arm,task,fold)
        verify_runtime(model,cfg)
        ds=d.Data(task,fold,formal,DOWN,SPLIT,PARENT/'inputs',OUT/'inputs')
        if formal:
            if ds.indices!=selected['test_indices'] or ds.keys.tolist()!=selected['test_keys']:
                raise ValueError('frozen TEST row/key mismatch')
        else:
            scaler_validate(ds.raw_targets,ds.train,scaler)
        ds.targets=transform(ds.raw_targets,scaler); write(folder/'scaler.json',scaler)
        write(folder/'environment.json',dict(versions=numerical.versions(),setup=setup,numerical=numerical.check_det(),physical=physical))
        model.to('cuda:0'); validation=d.loader(ds,ds.validation)
        if formal:
            truth,prediction,result=evaluate(model,ds,validation,scaler,usage,True)
            ref=REF/'formal/F_PD_N1_F0'/task/f'fold{fold}'
            with np.load(ref/'test_predictions.npz',allow_pickle=False) as prior:
                for key,value in (('indices',np.asarray(ds.indices)),('sample_keys',ds.keys),('y_true',truth)):
                    if not np.array_equal(prior[key],value):
                        raise ValueError('reference TEST identity differs: '+key)
            save_predictions(folder/'test_predictions.npz',ds,truth,prediction,best_epoch=best_epoch,best_stage=best_stage)
            write(folder/'identity_alignment.json',dict(indices=ds.indices,sample_keys=ds.keys.tolist(),raw_targets=ds.raw_targets.tolist(),scaler=scaler,matched=True))
        else:
            train=d.loader(ds,ds.train,True,cfg['seed']); seed_all(cfg['seed']); best_rmse=math.inf; history=[]
            (folder/'validation').mkdir()
            for stage,epochs in (('head',10),('joint',60)):
                opt,sch,groups=schedule(model,stage)
                for local_epoch in range(epochs):
                    guard_deadline(); tick=time.time(); usage.add('ft_epochs'); usage.record(); model.train(); verify_runtime(model,cfg)
                    losses=[]; parts_sum={}; rates=[g['lr'] for g in opt.param_groups]
                    for batch in train:
                        guard_deadline(); usage.add('ft_train_forwards'); usage.add('ft_train_rows',len(batch.y)); usage.record()
                        batch=batch.to('cuda:0'); opt.zero_grad(set_to_none=True); before=reductions(model)
                        try:
                            loss,parts=model.training_loss(batch)
                        finally:
                            usage.add('p_learn_reductions',reductions(model)-before)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('training loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                        usage.add('ft_updates'); usage.record(); opt.step()
                        usage.add('ft_updates',completed=True); usage.add('ft_train_forwards',completed=True); usage.add('ft_train_rows',len(batch.y),True)
                        losses.append(float(loss.detach()))
                        for k,v in parts.items():
                            parts_sum[k]=parts_sum.get(k,0.)+float(v)*len(batch.y)
                    sch.step(); usage.add('validation_count'); usage.record()
                    truth,prediction,met=evaluate(model,ds,validation,scaler,usage,False)
                    epoch=len(history)+1
                    save_predictions(folder/'validation'/f'epoch{epoch:03d}.npz',ds,truth,prediction,epoch=epoch)
                    if met['rmse']<best_rmse:
                        best_rmse=met['rmse']; best_epoch=epoch; best_stage=stage; result=met
                        save_checkpoint(folder/'best.pt',dict(schema=m.SCHEMA,config=cfg,state_dict=numerical.clone_tree(model.state_dict()),
                            best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,outer_test='NOT_RUN'))
                        save_predictions(folder/'validation_predictions.npz',ds,truth,prediction,best_epoch=best_epoch,best_stage=best_stage)
                    history.append(dict(epoch=epoch,stage=stage,stage_epoch=local_epoch+1,validation=met,train_loss=float(np.mean(losses)),
                        training_components={k:v/len(ds.train) for k,v in parts_sum.items()},learning_rates=rates,optimizer_groups=groups,
                        runtime=model.runtime_contract(),seconds=time.time()-tick))
                    write(folder/'history.json',history); usage.add('ft_epochs',completed=True); usage.add('validation_count',completed=True); usage.record()
                    print(json.dumps(dict(arm=arm,task=task,fold=fold,epoch=epoch,validation=met)),flush=True)
            if usage.value['completed'].get('ft_epochs')!=70 or usage.value['completed'].get('ft_updates')!=70*len(train):
                raise ValueError('incomplete FT')
        write(folder/'metrics.json',dict(status='COMPLETE',config=cfg,best_epoch=best_epoch,best_stage=best_stage,scaler=scaler,**result))
        usage.add('test_starts' if formal else 'ft_starts',completed=True); usage.finish()
    except BaseException:
        usage.fail(); raise
    finally:
        if ds is not None:
            ds.close()


def diagnostic(arm,repeat,setup):
    require_tmux(); guard_deadline(); check_prepared(); physical=gpu_contract()
    folder=job_path('diagnostic',arm,repeat=repeat); folder.mkdir(parents=True,exist_ok=False)
    usage=Consumption(folder,'diagnostic'); usage.add('diagnostic_starts'); usage.record(); ds=None; hooks=None
    try:
        ds=d.Data('eps',0,False,DOWN,SPLIT,PARENT/'inputs',OUT/'inputs')
        positions={row:i for i,row in enumerate(ds.indices)}; ids=[positions[row] for row in FIXED_ROWS]
        if any(i not in ds.train for i in ids):
            raise ValueError('diagnostic rows must be original training rows')
        scaler=read(SCALERS/'eps/fold0/scaler.json'); scaler_validate(ds.raw_targets,ds.train,scaler); ds.targets=transform(ds.raw_targets,scaler)
        write(folder/'input_identity.json',dict(indices=FIXED_ROWS,keys=ds.keys[ids].tolist(),scaler=scaler))
        write(folder/'environment.json',dict(versions=numerical.versions(),setup=setup,numerical=numerical.check_det(),physical=physical))
        batch=d.collate([ds[i] for i in ids]).to('cuda:0')
        seed_all(42); model=load_base(arm,'eps',0).to('cuda:0'); model.eval(); hooks=numerical.Hooks(model)
        for i in range(2):
            usage.add('diagnostic_eval_forwards'); usage.add('diagnostic_eval_rows',24); usage.record()
            hooks.begin(); before=reductions(model)
            try:
                with torch.inference_mode():
                    p,w,z=model.predict(batch)
            finally:
                usage.add('p_learn_reductions',reductions(model)-before)
            save_checkpoint(folder/f'eval{i}.pt',dict(prediction=numerical.clone_tree(p),weights=numerical.clone_tree(w),modalities=numerical.clone_tree(z),ph=numerical.clone_tree(model.last_ph),hooks=hooks.take(),rng=numerical.rng_state()))
            usage.add('diagnostic_eval_forwards',completed=True); usage.add('diagnostic_eval_rows',24,True); usage.record()
        hooks.close(); hooks=None; del model
        usage.add('diagnostic_train_starts'); usage.record()
        seed_all(42); model=load_base(arm,'eps',0)
        save_checkpoint(folder/'initial.pt',dict(state_dict=numerical.clone_tree(model.state_dict()),rng=numerical.rng_state()))
        model.to('cuda:0'); hooks=numerical.Hooks(model); seed_all(42); records=[]
        for stage in ('head','joint'):
            opt,sch,groups=schedule(model,stage)
            for k in range(10):
                guard_deadline(); step=len(records)+1; numerical.check_det(); model.train(); hooks.begin(); opt.zero_grad(set_to_none=True)
                rates=[g['lr'] for g in opt.param_groups]; before_rng=numerical.rng_state()
                usage.add('diagnostic_train_forwards'); usage.add('diagnostic_train_rows',24); usage.record(); before=reductions(model)
                try:
                    loss,parts=model.training_loss(batch)
                finally:
                    usage.add('p_learn_reductions',reductions(model)-before)
                if not torch.isfinite(loss):
                    raise FloatingPointError('diagnostic loss')
                usage.add('diagnostic_train_forwards',completed=True); usage.add('diagnostic_train_rows',24,True)
                usage.add('diagnostic_backwards'); usage.record(); loss.backward(); usage.add('diagnostic_backwards',completed=True)
                if arm=='P_LEARN':
                    grads=[p.grad for p in model.graph_ph.filter.parameters()]
                    if any(x is None or not torch.isfinite(x).all() for x in grads):
                        raise ValueError('learned filtration gradient disconnected/nonfinite')
                if step in (1,11):
                    save_checkpoint(folder/f'grad_pre_step{step}.pt',{n:numerical.clone_tree(p.grad) for n,p in model.named_parameters() if p.grad is not None})
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                if step in (1,11):
                    save_checkpoint(folder/f'grad_post_step{step}.pt',{n:numerical.clone_tree(p.grad) for n,p in model.named_parameters() if p.grad is not None})
                usage.add('diagnostic_updates'); usage.record(); opt.step(); usage.add('diagnostic_updates',completed=True); sch.step()
                record=dict(step=step,stage=stage,loss=float(loss.detach()),components={k:float(v) for k,v in parts.items()},learning_rates=rates,
                            optimizer_groups=groups,clip_norm=float(norm),hooks=hooks.take(),ph=numerical.clone_tree(model.last_ph),runtime=model.runtime_contract(),rng_before=before_rng,rng_after=numerical.rng_state())
                save_checkpoint(folder/f'train_step{step:02d}.pt',record); records.append(dict(step=step,stage=stage,loss=record['loss']))
                if step in (1,11,20):
                    save_checkpoint(folder/f'updated_step{step}.pt',dict(state_dict=numerical.clone_tree(model.state_dict()),optimizer=numerical.clone_tree(opt.state_dict()),scheduler=numerical.clone_tree(sch.state_dict()),rng=numerical.rng_state()))
                write(folder/'history.json',records); usage.record()
        usage.add('diagnostic_starts',completed=True); usage.add('diagnostic_train_starts',completed=True); usage.finish()
    except BaseException:
        usage.fail(); raise
    finally:
        if hooks is not None:
            hooks.close()
        if ds is not None:
            ds.close()


def compare_diagnostics():
    files=['initial.pt','eval0.pt','eval1.pt']+[f'train_step{i:02d}.pt' for i in range(1,21)]+[f'grad_{s}_step{i}.pt' for i in (1,11) for s in ('pre','post')]+[f'updated_step{i}.pt' for i in (1,11,20)]
    checks={}
    for arm in m.ARMS:
        paths=[job_path('diagnostic',arm,repeat=r) for r in ('R0','R1')]
        for name in files:
            a,b=[torch.load(p/name,map_location='cpu',weights_only=False) for p in paths]
            check=numerical.exact_compare(a,b); checks[arm+'/'+name]=check
            if not check['equal']:
                exclusive(OUT/'diagnostics/comparison.json',dict(numerical_diagnostic_pass=False,checks=checks)); raise ValueError('same-card diagnostic exact comparison failed')
        for repeat in ('R0','R1'):
            path=job_path('diagnostic',arm,repeat=repeat)
            a,b=[torch.load(path/f'eval{i}.pt',map_location='cpu',weights_only=False) for i in range(2)]
            check=numerical.exact_compare(a,b); checks[arm+'/intra/'+repeat]=check
            c0=torch.load(job_path('diagnostic','C0_BASE',repeat=repeat)/'eval0.pt',map_location='cpu',weights_only=False)
            shared=numerical.exact_compare({k:a[k] for k in ('prediction','weights','modalities')},{k:c0[k] for k in ('prediction','weights','modalities')})
            checks[arm+'/zero_residual/'+repeat]=shared
            if not check['equal'] or not shared['equal']:
                exclusive(OUT/'diagnostics/comparison.json',dict(numerical_diagnostic_pass=False,checks=checks)); raise ValueError('zero residual/intraprocess eval gate')
    exclusive(OUT/'diagnostics/comparison.json',dict(numerical_diagnostic_pass=True,checks=checks,time=now()))


def aggregate(arm, formal=False):
    tasks={}; units={}
    for task in TASKS:
        values=[]
        for fold in range(5):
            folder=job_path('test' if formal else 'ft',arm,task,fold); met=read(folder/'metrics.json'); usage=read(folder/'consumption.json')
            if usage['status']!='COMPLETE' or met['status']!='COMPLETE' or met['config']!=config(arm,task,fold):
                raise ValueError('incomplete estimator')
            split=read(SPLIT/f'{task}.json')['folds'][fold]
            wanted=dict(test_starts=1,test_rows=len(split['test_indices']),test_forwards=math.ceil(len(split['test_indices'])/24)) if formal else dict(ft_starts=1,ft_epochs=70,ft_updates=70*math.ceil(len(split['train_indices'])/24),ft_train_rows=70*len(split['train_indices']),validation_rows=70*len(split['validation_indices']),validation_count=70)
            if any(usage['completed'].get(k)!=v for k,v in wanted.items()):
                raise ValueError('unit consumption mismatch')
            values.append(met['r2']); units[f'{arm}/{task}/{fold}']=dict(config=met['config'],best=str(job_path('ft',arm,task,fold)/'best.pt'),best_epoch=met['best_epoch'],best_stage=met['best_stage'],scaler=met['scaler'],test_indices=sorted(split['test_indices']))
        tasks[task]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),folds=values)
    return dict(tasks=tasks,macro=float(np.mean([x['mean'] for x in tasks.values()])),units=units)


def select_final(validation):
    c0=validation['C0_BASE']; eligible=[]
    for arm in m.CANDIDATES:
        x=validation[arm]
        matched=validation['P_PE']['macro'] if arm in m.P_ARMS else max(validation[a]['macro'] for a in ('M_EQ','M_GEOM'))
        if x['macro']>c0['macro'] and x['macro']>matched and all(x['tasks'][task]['mean']>=c0['tasks'][task]['mean']-.005 for task in TASKS):
            eligible.append(arm)
    return max(eligible,key=lambda a:(validation[a]['macro'],-m.CANDIDATES.index(a))) if eligible else 'BASE_KEEP'


def freeze():
    check_prepared(); stability_gate()
    if read(OUT/'diagnostics/comparison.json')['numerical_diagnostic_pass'] is not True:
        raise ValueError('diagnostic gate')
    validation={a:aggregate(a) for a in m.ARMS}; final=select_final(validation)
    units={}; rows=read(OUT/'inputs/rows.json')
    for arm,value in validation.items():
        for key,u in value['units'].items():
            cfg=u['config']; saved=torch.load(u['best'],map_location='cpu',weights_only=False)
            if saved['schema']!=m.SCHEMA or saved['config']!=cfg or any(saved[k]!=u[k] for k in ('best_epoch','best_stage','scaler')):
                raise ValueError('freeze checkpoint metadata')
            mapping={e['row']:e['key'] for e in rows[cfg['task']]}; u['test_keys']=[mapping[i] for i in u['test_indices']]
            u.update(inference_allowed=True,sidecar_schema=d.SCHEMA,environment=read(job_path('ft',arm,cfg['task'],cfg['fold'])/'environment.json'))
            units[key]=u
    if len(units)!=245:
        raise ValueError('freeze matrix')
    aliases=[dict(task=task,fold=fold,best=str(REF/'finetune/F_PD_N1_F0'/task/f'fold{fold}/best.pt'),inference_allowed=False) for task in TASKS for fold in range(5)]
    selection=dict(FINAL=final,validation=validation,rule='CURRENT_PLAN r2 section7 / PH section11',test_scores_read=False)
    exclusive(OUT/'selection/validation_selection.json',selection)
    exclusive(OUT/'selection/final_frozen.json',dict(**contract(),frozen=True,FINAL=final,time=now(),time_unix=time.time(),units=units,reference_aliases=aliases,reference_macro=REFERENCE,selection=selection,test_labels_read=False))


def report(jobs, error):
    validation={}; formal={}
    for arm in m.ARMS:
        for dest,flag in ((validation,False),(formal,True)):
            if sum(j['kind']==('test' if flag else 'ft') and j.get('arm')==arm and j['exit']==0 for j in jobs)==35:
                dest[arm]=aggregate(arm,flag)
    fp=OUT/'selection/final_frozen.json'; frozen=read(fp) if fp.exists() else None
    final=frozen['FINAL'] if frozen else None; delta=formal[final]['macro']-REFERENCE if final in formal else None
    cpu=next((j['exit'] for j in jobs if j['kind']=='cpu'),None); preflight=next((j['exit'] for j in jobs if j['kind']=='preflight'),None)
    result=dict(**contract(),status='STOPPED' if error else 'COMPLETE_PENDING_REVIEW',error=error,FINAL=final,
        execution_complete=len(formal)==7 and cpu==0 and error is None,ft_complete=len(validation)==7,formal_test_complete=len(formal)==7,
        preflight_cpu_pass=preflight==0 if preflight is not None else None,final_cpu_pass=cpu==0 if cpu is not None else None,
        performance_pass=delta>=.010 if delta is not None else False if final=='BASE_KEEP' and len(formal)==7 else None,
        reference_macro=REFERENCE,delta_macro=delta,validation=validation,formal=formal,independent_review='PENDING',
        evaluation='shared development; one parent eea/fold4 TEST label exposed; single training seed rule 42+fold',
        metrics_status='完整固定矩阵' if len(formal)==7 else '未完成/不可比较',sota={},ph_increments={})
    if len(formal)==7:
        for a,b in (('P_FIXED','P_PE'),('P_LEARN','P_FIXED'),('P_LEARN','P_PE'),('M_PH','M_GEOM'),('M_GEOM','M_EQ'),('P_PE','C0_BASE'),('M_EQ','C0_BASE'),('M_PH','C0_BASE')):
            result['ph_increments'][a+'-'+b]=formal[a]['macro']-formal[b]['macro']
        result['delta_all_vs_reference']={a:x['macro']-REFERENCE for a,x in formal.items()}
        result['FINAL_minus_C0']=formal[final]['macro']-formal['C0_BASE']['macro'] if final in formal else None
        paper_std=dict(eea=.013,egb=.009,egc=.008,ei=.018,eps=.033,nc=.032,xc=.102)
        for arm in m.ARMS:
            result['sota'][arm]={task:dict(project=formal[arm]['tasks'][task],paper_mean=SOTA[task][2],paper_std=paper_std[task],paper_model=SOTA[task][1],delta=formal[arm]['tasks'][task]['mean']-SOTA[task][2],source='SOTA_RESULT.md',comparison='numeric only; protocols not proven equivalent; per-task model envelope') for task in TASKS}
    write(OUT/'final_review/results.continuation01.json',result)
    lines=['# PH r2 continuation01 交付待审查','',f'执行状态：{result["status"]}；{result["metrics_status"]}；independent_review=PENDING。',
        f'前置CPU={preflight!r}；最终CPU={cpu!r}；FINAL={final!r}；Δmacro完整精度={delta!r}。',
        '原t0及北京时间2026-10-30 23:02:54硬截止不变；父STOPPED/启动失败和违规CPU记录只读保留。',
        '共享开发评价，父期Eea/fold4一条TEST标签暴露；不称独立盲测或跨seed稳定提升。','',
        '| Arm | TEST macro | Δ原BASE |','|---|---:|---:|']
    for arm in m.ARMS:
        x=formal.get(arm); lines.append(f'| {arm} | '+(f'{x["macro"]:.3f} | {x["macro"]-REFERENCE:+.3f} |' if x else '未完成/不可比较 | — |'))
    for arm,entries in result['sota'].items():
        lines+=['',f'## {arm}','', '|任务|项目mean±std|论文mean±std|差值|论文模型|','|---|---:|---:|---:|---|']
        for task,e in entries.items():
            lines.append(f'|{task}|{e["project"]["mean"]:.3f}±{e["project"]["std"]:.3f}|{e["paper_mean"]:.3f}±{e["paper_std"]:.3f}|{e["delta"]:+.3f}|{e["paper_model"]}|')
    lines+=['','论文为逐任务最佳配置包络；数据/split/统计未证明等价，仅数值对照。Egc3380/论文4124、Xc432/论文430的差异保留。',
            '阻断：'+(error or '无')+'。','本次不自动修复/重试/补单元，不替换BASE。']
    (OUT/'report.continuation01.md').write_text('\n'.join(lines)+'\n')
    (OUT/'closeout.continuation01.md').write_text('\n'.join(lines[:8])+'\n')
    return result
