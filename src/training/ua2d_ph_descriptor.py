"""Isolated PH descriptor fusion campaign: explicit paths, fixed PTSCALE/P10K reference, no PT."""
import os,json,math,copy,time,traceback,datetime,sqlite3
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.dataset.ua2d_ph_descriptor import Data,loader,SEMANTICS,INPUT_SCHEMA,PHCache,ph_job,fit_feature_scaler
from src.modules.ua2d_ph_descriptor import (ARMS,SCHEMA as FT_SCHEMA,ARCHITECTURE,PHModel,
    initialize as initialize_model,optimizer,restore as restore_best,new_branches,validate_parent)
from src.training.ua2d_fpdecay_sg import (finite,metrics,seed_all,begin_epoch_reservation,close_epoch_reservation,
    conservative_ft_counts,process_identity)
from src.training.glt_dual_runtime import require_tmux,save_checkpoint,open_source
from src.training.ua2d_ptscale import PROFILE as ORIGINAL_PROFILE
ROOT=Path(__file__).resolve().parents[2]
PLAN='UA2D-PH-DESC-20261023-01'
REVISION='r1.2-P10K-GPU01'
OUT=ROOT/'results/ua2d_ph_descriptor_20261023/r1_p10k_gpu01'
LOG=ROOT/'logs/ua2d_ph_descriptor_20261023/r1_p10k_gpu01'
INPUTS=OUT/'inputs'
REF=ROOT/'results/ua2d_ptscale_20261022/r1'
PARENT=REF/'pretrain/checkpoints/step10000/deploy.pt'
DOWN=ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT=ROOT/'data/splits/mips_paper5_outer'
GRAPH=ROOT/'results/ua2d_transfer_20261022/r1/inputs'
HEADS=ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS=('eea','egb','egc','ei','eps','nc','xc')
PROFILE=copy.deepcopy(ORIGINAL_PROFILE)
PROFILE['checkpoint_selection']='strict original-unit validation RMSE decrease, ties earlier'
FINAL='GATE_PH'
LIMITS=dict(prepare_starts=1,initialization_starts=1,pt_starts=0,pt_updates=0,pt_presentations=0,
 fp_generation=0,ft_starts=70,epochs_started=4900,validation_started=4900,attempted_updates=112000,
 test_starts=70,test_graphs=11750,cpu_suites=1)
PER_ARM_LIMIT=4
TOTAL_LIMIT=4
SOTA={'eea':('Eea','Periodic-TDL',.934),'egb':('Egb','Periodic-TDL (+Chem, +RC)',.923),
 'egc':('Egc','Periodic-TDL (+Chem, +RC)',.911),'ei':('Ei','Periodic-TDL',.839),
 'eps':('EPS','Periodic-TDL',.787),'nc':('Nc','Periodic-TDL',.856),'xc':('Xc','Periodic-TDL',.423)}


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def unit_path(arm,task,fold,formal=False):return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'
def parent_path(arm):return PARENT
STEPS={a:10000 for a in ARMS}
GPU_LIMITS={2:4}
FT_GPU={a:2 for a in ARMS}

def gpu_for(task,fold):return 2

def feature_scaler_path(task,fold):return OUT/'feature_scalers'/task/f'fold{fold}.json'



def unit_config(arm,task,fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):raise ValueError('outside fixed two-arm35 matrix')
    return dict(plan=PLAN,revision=REVISION,arm=arm,task=task,fold=fold,seed=42+fold,
        parent=str(PARENT),parent_step=10000,parent_plan='UA2D-PTSCALE-20261022-01',parent_arm='P10K',
        parent_revision='r1',parent_schema='ua2d-ptscale-pretrain-r1',parent_world_size=2,head_initial=str(HEADS/task/f'fold{fold}.pt'),
        source=str(DOWN),split=str(SPLIT/f'{task}.json'),graph_cache=str(GRAPH),ph_cache=str(INPUTS),feature_scaler_path=str(feature_scaler_path(task,fold)),
        ph_semantics=SEMANTICS,ph_schema=INPUT_SCHEMA,input_semantics='finite Trimer PH412 + unchanged2hop O8',
        architecture=ARCHITECTURE,profile=PROFILE,physical_gpu=gpu_for(task,fold),logical_gpu_slot=(TASKS.index(task)*5+fold)%2,world_size=1,
        attention=dict(revision='ua2d_attention_scale_r1',scale_denominator=512,scale=512**-.5,bias_scaled=False))


def initialize(parent,head,arm):
    common=torch.load(INPUTS/'fusion_initialization.pt',map_location='cpu',weights_only=False)
    if common.get('seed')!=20261023 or common.get('plan')!=PLAN:raise ValueError('fusion initial source mismatch')
    return initialize_model(parent,head,common,arm)


def frozen_path(arm):return OUT/'selection/final_frozen.json'


def reference_results():
    frozen=read(REF/'selection/final_frozen.json');budget=read(REF/'budget.json')
    for kind in ('ft','test'):
        jobs=[j for j in budget['jobs'] if j['kind']==kind and j['arm']=='P10K']
        if len(jobs)!=35 or any(j['exit']!=0 for j in jobs):raise ValueError('P10K_REF complete real exits missing')
    if len([k for k in frozen['units'] if k.startswith('P10K/')])!=35 or 'P10K' not in frozen['models']:raise ValueError('P10K_REF freeze invalid')
    tasks={};units={}
    for task in TASKS:
        values=[]
        for fold in range(5):
            path=REF/'formal/P10K'/task/f'fold{fold}'
            m=read(path/'metrics.json');c=read(path/'consumption.json');key=f'P10K/{task}/{fold}'
            if (m['status']!='COMPLETE' or c['status']!='COMPLETE' or c['test_started']!=1
                or m['config']!=frozen['units'][key]['config'] or not (path/'test_predictions.npz').is_file()):
                raise ValueError('P10K_REF original formal identity mismatch')
            values.append(float(m['r2']));units[key]=dict(metrics=str(path/'metrics.json'),predictions=str(path/'test_predictions.npz'))
        finite(values);tasks[task]=dict(folds=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
    return dict(macro=float(np.mean([tasks[t]['mean'] for t in TASKS])),tasks=tasks,units=units,
        frozen=str(REF/'selection/final_frozen.json'),reference_only=True)


def prepare():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from collections import deque
    from importlib.metadata import version
    from src.dataset.mcl_ph_direct import DirectMCLSource
    from src.training.ua2d_fpdecay_sg import cache_metadata,validate_rows
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('PH prepare must be CPU-only')
    INPUTS.mkdir(exist_ok=False);(INPUTS/'barcodes').mkdir();start=time.monotonic()
    state=dict(plan=PLAN,status='RUNNING',process=process_identity(os.getpid()),workers=[],records=0,computed_unique=0,initialization_starts=0)
    write(INPUTS/'consumption.json',state);pool=None;worker_processes=[]
    try:
        validate_parent(torch.load(PARENT,map_location='cpu',weights_only=False))
        parent_meta=read(PARENT.parent/'metadata.json')
        if parent_meta.get('status')!='COMPLETE' or parent_meta.get('step')!=10000 or parent_meta.get('schema')!='ua2d-ptscale-pretrain-r1':raise ValueError('P10K snapshot not complete')
        write(OUT/'authorization/P10K_REF.json',reference_results())
        cache_metadata(GRAPH);graph_rows=read(GRAPH/'rows.json');mapping=validate_rows(graph_rows,5875,TASKS)
        pool=ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'))
        pending=deque();seen={};geometries={};rows={};records=[];raw=[];valid=[]
        def receive():
            future=pending.popleft();i,value,barcode=future.result()
            if raw[i] is not None:raise ValueError('duplicate PH computation result')
            raw[i]=value;write(INPUTS/'barcodes'/f'{i:05d}.json',dict(index=i,key=records[i]['key'],**barcode))
            state['computed_unique']+=1;write(INPUTS/'consumption.json',state)
        for task in TASKS:
            entries=sorted(graph_rows[task],key=lambda e:e['row']);indices=[e['row'] for e in entries]
            source=DirectMCLSource(DOWN,task=task,selected_indices=indices,split_manifest=read(SPLIT/f'{task}.json'),verify_key_digest=False)
            rows[task]=[]
            try:
                for local,(entry,row) in enumerate(zip(entries,source.entries)):
                    if int(row['original_row'])!=entry['row'] or row['sample_key']!=entry['key']:raise ValueError('PH original graph/source identity mismatch')
                    record=source.record(local);tri=record.trimer;key=row['sample_key'];sm=row['normalized_smiles']
                    z=torch.as_tensor(tri.trimer_atomic_number).cpu().numpy().astype(np.int64,copy=True)
                    pos=torch.as_tensor(tri.trimer_pos).cpu().numpy().astype(np.float64,copy=True)
                    ru=torch.as_tensor(tri.trimer_ru_offset).cpu().numpy().copy()
                    base=torch.as_tensor(tri.trimer_base_ru_atom_id).cpu().numpy().copy()
                    good=bool(tri.trimer_geometry_valid)
                    if key in seen:
                        i=seen[key];prior=geometries[key]
                        if (records[i]['smiles']!=sm or valid[i]!=good or not np.array_equal(prior[0],z)
                            or not np.array_equal(prior[1],pos,equal_nan=True) or not np.array_equal(prior[2],ru)
                            or not np.array_equal(prior[3],base)):raise ValueError('same identity has conflicting atoms/coordinates/RU anchors')
                    else:
                        i=len(records);seen[key]=i;geometries[key]=(z,pos,ru,base)
                        records.append(dict(key=key,smiles=sm,geometry_status=row['geometry_status'],atomic_numbers=z.tolist(),ru_offsets=ru.tolist(),base_ids=base.tolist()))
                        raw.append(None);valid.append(good);pending.append(pool.submit(ph_job,i,z,pos,good))
                        worker_processes=list(pool._processes.values());state['workers']=[process_identity(p.pid) for p in worker_processes]
                        if len(pending)>=8:receive()
                    rows[task].append(dict(entry,smiles=sm,ph_index=i));state['records']+=1
                    if state['records']%25==0:write(INPUTS/'consumption.json',state);print(json.dumps(state),flush=True)
            finally:source.close()
        while pending:receive()
        pool.shutdown(wait=True);pool=None
        state['worker_exits']=[dict(pid=p.pid,exit=p.exitcode) for p in worker_processes]
        if len(records)!=3654 or state['records']!=5875 or sum(valid)!=3646 or len(valid)-sum(valid)!=8:
            raise ValueError('fixed PH cohort/validity differs from existing3654/3646/8 coverage')
        array=np.stack(raw).astype(np.float32)
        if array.shape!=(3654,412) or not np.isfinite(array).all() or (array<0).any():raise ValueError('PH raw arrays invalid')
        np.save(INPUTS/'descriptors.npy',array,allow_pickle=False);np.save(INPUTS/'ph_valid.npy',np.array(valid,dtype=bool),allow_pickle=False)
        write(INPUTS/'rows.json',rows);write(INPUTS/'structures.json',records)
        state['initialization_starts']=1;write(INPUTS/'consumption.json',state)
        ph,projection,gate=new_branches();save_checkpoint(INPUTS/'fusion_initialization.pt',dict(plan=PLAN,seed=20261023,ph_encoder=ph.state_dict(),ph_projection=projection.state_dict(),gate=gate.state_dict()))
        metadata=dict(plan=PLAN,revision=REVISION,schema=INPUT_SCHEMA,status='COMPLETE',records=5875,unique_structures=3654,
            valid_unique=3646,invalid_unique=8,descriptor_dtype='float32',geometry_generation_calls=0,online_fallback=False,
            semantics=SEMANTICS,graph_cache=str(GRAPH),source=str(DOWN),split=str(SPLIT),gudhi_version=version('gudhi'),
            numpy_version=np.__version__,torch_version=str(torch.__version__),feature_scaler_fitted_during_prepare=False,seconds=time.monotonic()-start)
        write(INPUTS/'metadata.json',metadata)
        gpu_map={f'{task}/{fold}':dict(gpu=gpu_for(task,fold),logical_slot=(TASKS.index(task)*5+fold)%2) for task in TASKS for fold in range(5)}
        write(OUT/'selection/gpu_map.json',dict(frozen=True,map=gpu_map,only_gpus=[2],gpu_limits={'2':4},source='latest user GPU2-only override; original parity retained as logical slot'))
        state.update(status='COMPLETE',seconds=time.monotonic()-start);write(INPUTS/'consumption.json',state)
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());write(INPUTS/'consumption.json',state)
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        raise


def ensure_feature_scalers():
    # Normal FT initialization, AFTER deterministic prepare; never fit on validation/TEST.
    cache=PHCache(INPUTS,GRAPH)
    for task in TASKS:
        rows=cache.rows[task];by_row={e['row']:e for e in rows}
        if sorted(by_row)!=list(range(len(rows))):raise ValueError('non-contiguous original task rows')
        ids=[by_row[i]['ph_index'] for i in range(len(rows))];raw=cache.raw[ids];valid=cache.valid[ids]
        for fold in range(5):
            path=feature_scaler_path(task,fold)
            if path.exists():raise FileExistsError('feature scaler already fitted')
            entry=read(SPLIT/f'{task}.json')['folds'][fold]
            value=fit_feature_scaler(raw,valid,entry['train_indices'])
            value.update(task=task,fold=fold,source=str(INPUTS),schema=INPUT_SCHEMA,
                valid_training_keys=[by_row[i]['key'] for i in value['valid_training_rows']])
            write(path,value)


def load_reference_scaler(ds,task,fold,formal):
    from src.utils import TargetScaler
    from sklearn.preprocessing import StandardScaler
    value=read(REF/'finetune/P10K'/task/f'fold{fold}'/'scaler.json')
    if value.get('fit')!='train' or value.get('transform')!='standard':raise ValueError('reference scaler semantics mismatch')
    entry=read(SPLIT/f'{task}.json')['folds'][fold]
    ref_cfg=read(REF/'finetune/P10K'/task/f'fold{fold}'/'config.json')
    if ref_cfg['seed']!=42+fold or ref_cfg['split']!=str(SPLIT/f'{task}.json') or ref_cfg['head_initial']!=str(HEADS/task/f'fold{fold}.pt'):
        raise ValueError('reference scaler fold/head/split identity mismatch')
    if not formal:
        actual=[ds.indices[i] for i in ds.train]
        if sorted(actual)!=sorted(entry['train_indices']):raise ValueError('scaler training rows differ')
        y=ds.raw_targets[ds.train];mean=float(y.mean());scale=float(y.std()) or 1.
        if not np.allclose([mean,scale],[value['mean'][0],value['scale'][0]],rtol=1e-10,atol=1e-12):raise ValueError('train-only scaler numerical mismatch')
    scaler=TargetScaler(task,StandardScaler(),transform_mode='standard')
    scaler.scaler.mean_=np.array(value['mean']);scaler.scaler.scale_=np.array(value['scale']);scaler.scaler.var_=scaler.scaler.scale_**2;scaler.scaler.n_features_in_=1
    return scaler,value

def unit(arm, task, fold, formal=False):
    from src.utils import TargetScaler
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(FT_GPU[arm]):
        raise ValueError('FT/TEST physical GPU differs from explicit arm allocation')
    if os.environ.get('WORLD_SIZE')!='1':raise ValueError('single process/world1 required')
    folder = unit_path(arm, task, fold, formal)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = unit_config(arm, task, fold)
    allocation=read(OUT/'selection/gpu_map.json')
    if allocation['map'][f'{task}/{fold}']['gpu']!=cfg['physical_gpu']:raise ValueError('frozen GPU assignment mismatch')
    write(folder / 'config.json', cfg)
    state = dict(status='RUNNING', pid=os.getpid(), process=process_identity(os.getpid()),
                 started=now(), completed_epochs=0, epochs_started=0, validation_started=0,
                 validation_completed=0, attempted_updates=0, completed_updates=0,
                 test_started=0, test_graphs=0, phase='initialize', outer_test='NOT_RUN')
    io_seconds = 0.
    io_calls = 0

    def timed_write(path, value, checkpoint=False):
        nonlocal io_seconds, io_calls
        tick = time.monotonic()
        if checkpoint:
            save_checkpoint(path, value)
        else:
            write(path, value)
        io_seconds += time.monotonic() - tick
        io_calls += 1

    def record():
        state['engineering_revision'] = 'ph_descriptor-epoch-reservation-r1'
        timed_write(folder / 'consumption.json', state)

    record()
    started = time.monotonic()
    ds = None
    try:
        seed_all(42 + fold)
        if formal:
            frozen = read(frozen_path(arm))
            if frozen.get('frozen') is not True or frozen.get('ph_metadata')!=read(INPUTS/'metadata.json'):
                raise ValueError('TEST before the one-time freeze')
            if read(OUT/'selection/gpu_map.json')!=frozen['gpu_map']:raise ValueError('frozen GPU mapping changed')
            if frozen['splits'][task] != read(SPLIT / f'{task}.json'):
                raise ValueError('frozen original split changed')
            selected = frozen['units'][f'{arm}/{task}/{fold}']
            saved = torch.load(selected['best'], map_location='cpu', weights_only=False)
            if (selected['config'] != cfg or selected['best'] != str(unit_path(arm, task, fold) / 'best.pt')
                    or selected['parent'] != str(parent_path(arm)) or selected['parent_step']!=10000
                    or selected['best_epoch'] != saved['best_epoch']
                    or selected['best_stage'] != saved['best_stage'] or selected['scaler'] != saved['scaler']
                    or selected['attention'] != cfg['attention']):
                raise ValueError('frozen complete best/scaler mismatch')
            model = restore_best(saved, cfg)
            saved_feature_scaler=saved['feature_scaler']
            if selected['feature_scaler']!=saved_feature_scaler:raise ValueError('frozen feature scaler/best conflict')
            scaler_json = saved['scaler']
            scaler = TargetScaler(task, StandardScaler(), transform_mode='standard')
            scaler.scaler.mean_ = np.array(scaler_json['mean'])
            scaler.scaler.scale_ = np.array(scaler_json['scale'])
            scaler.scaler.var_ = scaler.scaler.scale_ ** 2
            scaler.scaler.n_features_in_ = 1
            best_epoch, best_stage = saved['best_epoch'], saved['best_stage']
            del saved
        else:
            parent = torch.load(parent_path(arm), map_location='cpu', weights_only=False)
            head = torch.load(cfg['head_initial'], map_location='cpu', weights_only=False)
            model = initialize(parent, head, arm)
            del parent, head
        shared_feature_scaler=read(feature_scaler_path(task,fold))
        if formal and saved_feature_scaler!=shared_feature_scaler:raise ValueError('frozen feature scaler changed')
        ds = Data(task,fold,formal,source_root=DOWN,split_root=SPLIT,graph_root=GRAPH,ph_root=INPUTS,feature_scaler=shared_feature_scaler)
        checked_scaler,reference_scaler=load_reference_scaler(ds,task,fold,formal)
        if formal:
            mapping=[ds.ph.mapping[task][row] for row in ds.indices]
            if selected['sample_mapping']!=mapping:raise ValueError('frozen PH sample mapping changed')
            if selected['test_indices']!=ds.indices or reference_scaler!=scaler_json:raise ValueError('frozen TEST/scaler changed')
        else:scaler,scaler_json=checked_scaler,reference_scaler
        ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        write(folder / 'scaler.json', scaler_json)
        model.to('cuda:0')
        validation = loader(ds, ds.validation)

        def evaluate():
            model.eval()
            targets, predictions = [], []
            with torch.inference_mode():
                for batch in validation:
                    batch = batch.to('cuda:0')
                    prediction = model(batch)
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError('nonfinite validation/TEST prediction')
                    targets.append(batch.y.cpu().numpy())
                    predictions.append(prediction.cpu().numpy())
                    if formal:
                        state['test_graphs'] += len(batch.y)
                        record()
            y = scaler.inverse_transform(np.concatenate(targets)).reshape(-1)
            p = scaler.inverse_transform(np.concatenate(predictions)).reshape(-1)
            return y, p, metrics(y, p)

        if formal:
            state.update(phase='TEST', test_started=1)
            record()
            truth, prediction, result = evaluate()
        else:
            train = loader(ds, ds.train, True, 42 + fold)
            expected = 70 * len(train)
            if expected > 56000:
                raise RuntimeError('unit update reservation impossible')
            seed_all(42 + fold)
            history, epoch_timings = [], []
            best_rmse = math.inf
            for stage, epochs in (('head', 10), ('joint', 60)):
                opt, evidence = optimizer(model, stage)
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
                             if stage == 'head' else
                             torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1,
                                                                                  eta_min=1e-6))
                for stage_epoch in range(epochs):
                    epoch_tick = time.monotonic()
                    io_before = io_seconds
                    state['phase'] = stage
                    begin_epoch_reservation(state, len(train))
                    record()
                    rates = [g['lr'] for g in opt.param_groups]
                    model.train()
                    losses = []
                    for batch in train:
                        batch = batch.to('cuda:0')
                        opt.zero_grad(set_to_none=True)
                        loss = nn.functional.mse_loss(model(batch), batch.y.view(-1))
                        if not torch.isfinite(loss):
                            raise FloatingPointError('nonfinite loss')
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        if state['attempted_updates'] >= expected:
                            raise RuntimeError('unit update limit reached')
                        state['attempted_updates'] += 1
                        opt.step()
                        state['completed_updates'] += 1
                        losses.append(float(loss.detach()))
                    scheduler.step()
                    state['validation_started'] += 1
                    y, p, met = evaluate()
                    state['validation_completed'] += 1
                    state['completed_epochs'] += 1
                    rec = dict(epoch=state['completed_epochs'], stage=stage, stage_epoch=stage_epoch + 1,
                               train_loss=float(np.mean(losses)), validation_r2=met['r2'],
                               validation_rmse=met['rmse'], learning_rates=rates, optimizer_groups=evidence,
                               elapsed_seconds=time.monotonic() - started,
                               gpu_peak_bytes=torch.cuda.max_memory_allocated())
                    if met['rmse'] < best_rmse:
                        best_rmse = met['rmse']
                        best_epoch = state['completed_epochs']
                        best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        timed_write(folder / 'best.pt', dict(schema=FT_SCHEMA, architecture=ARCHITECTURE,
                                                             attention=cfg['attention'], config=cfg,
                                                             state_dict={k: v.detach().cpu().clone()
                                                                         for k, v in model.state_dict().items()},
                                                             best_epoch=best_epoch, best_stage=best_stage,
                                                             scaler=scaler_json,feature_scaler=shared_feature_scaler, outer_test='NOT_RUN'),
                                    checkpoint=True)
                    rec['epoch_seconds_before_history'] = time.monotonic() - epoch_tick
                    rec['write_seconds_before_history'] = io_seconds - io_before
                    history.append(rec)
                    timed_write(folder / 'history.json', history)
                    close_epoch_reservation(state)
                    record()
                    epoch_timings.append(dict(epoch=state['completed_epochs'],
                                              epoch_seconds=time.monotonic() - epoch_tick,
                                              write_seconds=io_seconds - io_before,
                                              write_seconds_cumulative=io_seconds,
                                              write_calls_cumulative=io_calls,
                                              timing_scope='real epoch incl CPU/model/evaluation/write'))
                    timed_write(folder / 'epoch_timing.json', epoch_timings)
                    print(json.dumps(rec), flush=True)
            if state['attempted_updates'] != expected or state['completed_updates'] != expected:
                raise RuntimeError('incomplete optimizer budget')
            timed_write(folder / 'resume.pt', dict(schema=FT_SCHEMA, architecture=ARCHITECTURE,
                                                   attention=cfg['attention'], config=cfg,
                                                   model=model.state_dict(), optimizer=opt.state_dict(),
                                                   scheduler=scheduler.state_dict(), stage=stage,
                                                   stage_epoch=stage_epoch + 1,
                                                   epoch=state['completed_epochs'], batch_cursor='epoch_end',
                                                   best_epoch=best_epoch, best_stage=best_stage,
                                                   best_rmse=best_rmse, scaler=scaler_json,feature_scaler=shared_feature_scaler,
                                                   consumption=copy.deepcopy(state), automatic_resume=False),
                        checkpoint=True)
        np.savez(folder / ('test_predictions.npz' if formal else 'validation_predictions.npz'),
                 y_true=finite(truth), y_pred=finite(prediction),
                 indices=np.array([ds.indices[i] for i in ds.validation]),
                 sample_keys=ds.keys[ds.validation], best_epoch=best_epoch, best_stage=best_stage)
        write(folder / 'metrics.json', dict(status='COMPLETE', config=cfg, best_epoch=best_epoch,
                                            best_stage=best_stage, scaler=scaler_json,
                                            attention=cfg['attention'],feature_scaler=shared_feature_scaler, **result))
        state.update(status='COMPLETE', ended=now(), best_epoch=best_epoch, best_stage=best_stage,
                     elapsed_seconds=time.monotonic() - started,
                     gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                     write_seconds=io_seconds, write_calls=io_calls)
        record()
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc(), ended=now())
        record()
        write(folder / 'failure.json', dict(error=state['error'], automatic_retry=False, time=now()))
        raise
    finally:
        if ds is not None:
            ds.close()

def aggregate(arm, formal=False):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    tasks, units = {}, {}
    for task in TASKS:
        folds, entries = [], []
        for fold in range(5):
            folder = unit_path(arm, task, fold, formal)
            m = read(folder / 'metrics.json')
            c = read(folder / 'consumption.json')
            cfg = unit_config(arm, task, fold)
            if m.get('status') != 'COMPLETE' or m.get('config') != cfg or c.get('status') != 'COMPLETE':
                raise ValueError('unit is not COMPLETE/config-consistent: ' + str(folder))
            if m.get('attention') != cfg['attention']:
                raise ValueError('unit attention record mismatch')
            if formal:
                if c.get('test_started') != 1 or c.get('test_graphs') != len(read(SPLIT / f'{task}.json')['folds'][fold]['test_indices']):
                    raise ValueError('formal TEST graph coverage mismatch')
            else:
                if c.get('completed_epochs') != 70 or c.get('validation_started') != 70:
                    raise ValueError('FT did not run all 70 epochs/validations')
                if c.get('attempted_updates') != c.get('completed_updates'):
                    raise ValueError('FT attempted/completed update mismatch')
            entries.append(dict(config=cfg, best=str(unit_path(arm,task,fold) / 'best.pt'), best_epoch=m['best_epoch'],
                                best_stage=m['best_stage'], scaler=m['scaler'], attention=m['attention'],
                                metrics={k: m[k] for k in ('r2', 'rmse', 'mae') if k in m},
                                predictions=str(folder / ('test_predictions.npz' if formal else 'validation_predictions.npz'))))
            folds.append(float(m['r2']))
        mean = float(np.mean(folds))
        std = float(np.std(folds, ddof=1))
        tasks[task] = dict(folds=folds, mean=mean, std=std)
        for fold in range(5):
            units[f'{arm}/{task}/{fold}'] = entries[fold]
    macro = float(np.mean([tasks[t]['mean'] for t in TASKS]))
    return dict(tasks=tasks, macro=macro, units=units)


def freeze_final(validation):
    path=OUT/'selection/final_frozen.json'
    if path.exists():raise FileExistsError('already frozen; never reselect')
    ph_metadata=read(INPUTS/'metadata.json');row_mapping=read(INPUTS/'rows.json');units={};splits={}
    for arm in ARMS:
        if arm not in validation:raise ValueError('all70FT must complete before freeze')
        for task in TASKS:
            splits[task]=read(SPLIT/f'{task}.json')
            by_row={e['row']:e for e in row_mapping[task]}
            for fold in range(5):
                folder=unit_path(arm,task,fold);m=read(folder/'metrics.json');c=read(folder/'consumption.json');cfg=unit_config(arm,task,fold)
                if (m['status']!='COMPLETE' or c['status']!='COMPLETE' or c['completed_epochs']!=70
                    or c['validation_started']!=70 or m['config']!=cfg or read(folder/'config.json')!=cfg
                    or read(folder/'scaler.json')!=m['scaler'] or read(feature_scaler_path(task,fold))!=m['feature_scaler']
                    or not (folder/'best.pt').is_file()):raise ValueError('FT incomplete at freeze')
                indices=sorted(splits[task]['folds'][fold]['test_indices'])
                units[f'{arm}/{task}/{fold}']=dict(config=cfg,best=str(folder/'best.pt'),scaler=m['scaler'],
                    scaler_path=str(folder/'scaler.json'),config_path=str(folder/'config.json'),parent=str(PARENT),parent_step=10000,
                    best_epoch=m['best_epoch'],best_stage=m['best_stage'],attention=m['attention'],
                    feature_scaler=read(feature_scaler_path(task,fold)),feature_scaler_path=str(feature_scaler_path(task,fold)),physical_gpu=gpu_for(task,fold),
                    test_indices=indices,sample_mapping=[by_row[row] for row in indices])
    if len(units)!=70:raise ValueError('freeze must cover70 unique units')
    frozen=dict(plan=PLAN,revision=REVISION,frozen=True,time=now(),FINAL=FINAL,models={a:str(PARENT) for a in ARMS},units=units,
        ph_metadata=ph_metadata,gpu_map=read(OUT/'selection/gpu_map.json'),splits=splits,validation=validation,reference=read(OUT/'authorization/P10K_REF.json'),
        policy='all70FT complete before first TEST; strict RMSE best; no performance gate')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(frozen,f,ensure_ascii=False,indent=2,allow_nan=False)
    return frozen
