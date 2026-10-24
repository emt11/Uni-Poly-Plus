"""UA2D-OPT r1 finite FT units; no automatic recovery."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import traceback
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.dataset.glt_dual import build_glt_2d_sample as build
from src.dataset.ua2d_opt import collate_opt as collate
SEMANTICS="original_2hop_2d"
from src.modules.ua2d_opt import initialize, restore, optimizer, ARCHITECTURE, SCHEMA
from src.training.glt_dual_runtime import require_tmux, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'UA2D-OPT-20261021-01'
OUT = ROOT/'results/ua2d_opt_20261021/r1'
LOG = ROOT/'logs/ua2d_opt_20261021/r1'
DOWN = ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT/'data/splits/mips_paper5_outer'
PARENT = ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS = ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
ARMS = ('A_TOP2','A_MIX246','B2_E','B3_E','B2_C','B3_C','C_SCHED_E','C_SCHED_C','D_INTEGRATE')
LIMITS=dict(prepare_starts=1,initialization_starts=1,pt_starts=5,pt_updates=25000,pt_presentations=25200000,ft_starts=280,epochs_started=19600,validation_started=19600,attempted_updates=448000,test_starts=210,test_graphs=35250,cpu_suites=1)
PROFILE = dict(batch=24, amp='fp32', workers=0, drop_last=False, head_epochs=10,
    joint_epochs=60, weight_decay=.02, clip=5., encoder_dropout=0., head_dropout=.1,
    head_lr=.0003, backbone_joint_lr=.0001, head_joint_lr=.001,
    head_scheduler=dict(name='CosineAnnealingLR', T_max=10, eta_min=.00003),
    joint_scheduler=dict(name='CosineAnnealingWarmRestarts', T_0=10, T_mult=1, eta_min=.000001))


def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)


def finite(x):
    a = np.asarray(x, dtype=np.float64)
    if not np.isfinite(a).all(): raise FloatingPointError('nonfinite numerical result')
    return a


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def config(arm, task, fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):
        raise ValueError('outside declared matrix')
    return dict(plan=PLAN, revision='r1', arm=arm, task=task, fold=fold, seed=42+fold,
        parent=str(parent_path(arm)),policy=policy(arm),schedule=schedule(arm), head_initial=str(HEADS/task/f'fold{fold}.pt'),
        source=str(DOWN), split=str(SPLIT/f'{task}.json'), architecture=ARCHITECTURE,
        profile=PROFILE,input_semantics=SEMANTICS if arm.startswith('A_') else 'ua2d-opt-hop3-r1')


def logical_unit(arm, task, fold):return f'{arm}/{task}/{fold}'


def repair_attempt(arm, task, fold, require_complete=False):
    path=OUT/'repair_ft1/attempt_mapping.json'
    if not path.exists():return None
    mapping=read(path)
    if mapping.get('revision')!='r1-FT-REPAIR1':raise ValueError('attempt mapping revision mismatch')
    key=logical_unit(arm,task,fold)
    if key not in mapping['units']:raise ValueError('missing explicit attempt mapping: '+key)
    entry=mapping['units'][key]
    expected=(OUT/'repair_ft1/attempts'/arm/task/f'fold{fold}'/'attempt1' if entry.get('replacement')
        else OUT/'finetune'/arm/task/f'fold{fold}')
    if Path(entry['path'])!=expected or (entry.get('replacement') and key not in REPAIR_FAILED_UNITS):
        raise ValueError('unauthorized attempt path or replacement')
    if require_complete and entry['status']!='COMPLETE':raise ValueError('mapped attempt incomplete: '+key)
    return entry


def unit_path(arm, task, fold, formal=False):
    if formal:return OUT/'formal'/arm/task/f'fold{fold}'
    entry=repair_attempt(arm,task,fold)
    return Path(entry['path']) if entry else OUT/'finetune'/arm/task/f'fold{fold}'


REPAIR_FAILED_UNITS=tuple([f'B3_C/egc/{f}' for f in range(5)]+['B3_C/ei/4','B3_C/eps/0','B3_C/eps/1'])


def begin_epoch_reservation(state, updates):
    if state.get('epoch_reservation',{}).get('open'):raise ValueError('previous epoch reservation still open')
    state['epochs_started']+=1
    state['epoch_reservation']=dict(open=True,epoch=state['epochs_started'],updates=updates,validations=1,
        before_updates=state['attempted_updates'],before_validation=state['validation_started'])


def close_epoch_reservation(state):
    r=state['epoch_reservation']
    if (not r['open'] or state['attempted_updates']!=r['before_updates']+r['updates']
        or state['completed_updates']!=state['attempted_updates']
        or state['validation_started']!=r['before_validation']+1
        or state['completed_epochs']!=r['epoch']):raise ValueError('incomplete epoch cannot close reservation')
    r['open']=False


def conservative_ft_counts(state):
    actual={k:int(state.get(k,0)) for k in ('epochs_started','validation_started','attempted_updates','test_graphs')}
    charged=dict(actual);r=state.get('epoch_reservation',{})
    if r.get('open'):
        charged['attempted_updates']=max(actual['attempted_updates'],r['before_updates']+r['updates'])
        charged['validation_started']=max(actual['validation_started'],r['before_validation']+r['validations'])
    return dict(known_actual_lower_bound=actual,charged=charged,
        uncertain={k:charged[k]-actual[k] for k in charged},open_epoch=bool(r.get('open')))


def frozen_attempt(selected, arm, task, fold):
    entry=repair_attempt(arm,task,fold,require_complete=True)
    expected=unit_path(arm,task,fold)/'best.pt'
    if Path(selected['best'])!=expected or (entry and selected.get('attempt_job')!=entry['job_name']):
        raise ValueError('frozen checkpoint does not match successful attempt')
    return expected


def update_plan(status, phase, detail):
    p = ROOT/'CURRENT_PLAN.md'
    text = p.read_text()
    marker = '\n### UA2D-OPT 最新执行状态\n'
    end = '\n<!-- UA2D_OPT_EXECUTION_END -->'
    entry = marker+f'\n- 时间：{now()}；状态：{status}；阶段：{phase}。\n- {detail}\n- 结果根：{OUT}；日志根：{LOG}。实现者自检不构成独立审查。\n'+end
    if marker in text:
        before, block = text.split(marker, 1)
        _, after = block.split(end, 1)
        text = before+entry+after
    else: text = text.rstrip()+'\n'+entry+'\n'
    p.write_text(text)
    if OUT.exists():
        write(OUT/'status.json', dict(plan=PLAN, revision='r1', status=status, phase=phase,
            time=now(), detail=detail, independent_review='PENDING'))

class Data(torch.utils.data.Dataset):
    def __init__(self,task,fold,formal=False,arm=None):
        self.arm=arm
        from src.training.glt_dual_runtime import open_source
        from scripts.finetune_mcl_ph import compact_train_validation_indices
        if task not in TASKS or fold not in range(5):raise ValueError('unit outside matrix')
        manifest=read(SPLIT/f'{task}.json');entry=manifest['folds'][fold]
        if manifest.get('validation_is_test') is not False:raise ValueError('validation/test separation invalid')
        if formal:
            frozen=read(OUT/'selection/final_frozen.json')
            if not frozen['frozen']:raise ValueError('TEST before freeze')
            self.indices=sorted(entry['test_indices']);self.train=[];self.validation=list(range(len(self.indices)))
        else:self.indices,self.train,self.validation=compact_train_validation_indices(entry['train_indices'],entry['validation_indices'])
        self.source,frame=open_source(DOWN,DOWN,task=task,selected_indices=self.indices,split_manifest=manifest,verify_key_digest=False)
        if frame.original_row.astype(int).tolist()!=self.indices:raise ValueError('selected source row mismatch')
        self.raw_targets=finite(frame.label.to_numpy());self.targets=self.raw_targets.copy();self.cache={}
        self.keys=np.asarray([s[0].hex() for s in self.source.samples])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        from src.training.glt_dual_runtime import pretrain_source_fields
        from src.dataset.glt_dual import build_glt_2d_sample as build
        from scripts.finetune_mcl_ph import clone_clean_record
        if i not in self.cache:
            top,_,text,static,_=pretrain_source_fields(self.source,i)
            self.cache[i]=build(top,text,static=static) if self.arm.startswith('A_') else cached_downstream(self.arm,top,text,static,self.keys[i])
        d=clone_clean_record(self.cache[i]);d.y=torch.tensor([self.targets[i]],dtype=torch.float32);return d
    def set_target_override(self,y):self.targets=np.asarray(y,dtype=np.float64)
    def close(self):self.source.close()


def loader(ds, ids, shuffle=False, seed=42):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, ids), batch_size=24,
        shuffle=shuffle, num_workers=0, drop_last=False, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed) if shuffle else None)


def metrics(y, p):
    y, p = finite(y).reshape(-1), finite(p).reshape(-1)
    return dict(r2=float(r2_score(y, p)), rmse=float(mean_squared_error(y, p)**.5),
                mae=float(mean_absolute_error(y, p)))


def unit(arm, task, fold, formal=False):
    from src.utils import scale_targets, TargetScaler
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise ValueError('physical GPU2/3 workers only')
    entry=repair_attempt(arm,task,fold,require_complete=formal)
    if not formal and entry and (entry['status']!='DISPATCHED' or os.environ.get('UA2D_OPT_UNIT_JOB')!=entry['job_name']):
        raise ValueError('FT attempt not exclusively dispatched')
    folder = unit_path(arm, task, fold, formal); folder.mkdir(parents=True, exist_ok=False)
    cfg = config(arm, task, fold); write(folder/'config.json', cfg)
    state = dict(status='RUNNING', pid=os.getpid(), started=now(), completed_epochs=0,
        epochs_started=0, validation_started=0, validation_completed=0, attempted_updates=0,
        completed_updates=0, test_started=0, test_graphs=0, phase='initialize')
    io_seconds=0.;io_calls=0
    def timed_write(path,value,checkpoint=False):
        nonlocal io_seconds,io_calls
        tick=time.monotonic()
        if checkpoint:save_checkpoint(path,value)
        else:write(path,value)
        io_seconds+=time.monotonic()-tick;io_calls+=1
    def record():
        if entry:state['engineering_revision']='r1-FT-REPAIR1'
        timed_write(folder/'consumption.json',state)
    record()
    unit_started=time.monotonic()
    ds = None
    try:
        seed_all(42+fold)
        if formal:
            frozen = read(OUT/'selection/final_frozen.json')
            if frozen.get('frozen') is not True: raise ValueError('TEST before freeze')
            selected = frozen['units'][f'{arm}/{task}/{fold}']
            saved = torch.load(frozen_attempt(selected,arm,task,fold), map_location='cpu', weights_only=False)
            if (selected['config'] != cfg or selected['best_epoch'] != saved['best_epoch']
                    or selected['best_stage'] != saved['best_stage'] or selected['scaler'] != saved['scaler']):
                raise ValueError('frozen complete best/scaler mismatch')
            model = restore(saved, cfg); scaler_json = saved['scaler']
            scaler = TargetScaler(task, StandardScaler(), transform_mode='standard')
            scaler.scaler.mean_ = np.array(scaler_json['mean']); scaler.scaler.scale_ = np.array(scaler_json['scale'])
            scaler.scaler.var_ = scaler.scaler.scale_**2; scaler.scaler.n_features_in_ = 1
            best_epoch, best_stage = saved['best_epoch'], saved['best_stage']
            del saved
        else:
            parent = torch.load(parent_path(arm), map_location='cpu', weights_only=False)
            head = torch.load(cfg['head_initial'], map_location='cpu', weights_only=False)
            model = initialize(parent, head, arm); del parent
        ds = Data(task, fold, formal,arm)
        if formal:
            if selected['test_indices'] != ds.indices: raise ValueError('frozen TEST indices mismatch')
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=ds.train, transform_mode='standard')
            scaler_json = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(),
                               fit='train', transform='standard')
        reference_scaler = read(PARENT.parents[2]/'finetune/B_FP'/task/f'fold{fold}/scaler.json')
        if scaler_json != reference_scaler: raise ValueError('B_FP train-only scaler mismatch')
        write(folder/'scaler.json', scaler_json)
        model.to('cuda:0')
        validation = loader(ds, ds.validation)
        def evaluate():
            model.eval(); targets, predictions = [], []
            with torch.inference_mode():
                for batch in validation:
                    batch = batch.to('cuda:0'); p = model(batch)
                    if not torch.isfinite(p).all(): raise FloatingPointError('nonfinite validation/TEST prediction')
                    targets.append(batch.y.cpu().numpy()); predictions.append(p.cpu().numpy())
                    if formal:
                        state['test_graphs'] += len(batch.y); record()
            y = scaler.inverse_transform(np.concatenate(targets)).reshape(-1)
            p = scaler.inverse_transform(np.concatenate(predictions)).reshape(-1)
            return y, p, metrics(y, p)
        if formal:
            state.update(phase='TEST', test_started=1); record()
            truth, prediction, result = evaluate()
        else:
            train = loader(ds, ds.train, True, 42+fold)
            expected = 70*len(train)
            if expected > 56000: raise RuntimeError('unit update reservation impossible')
            seed_all(42+fold); history = []; epoch_timings=[]; best_r2 = -math.inf
            for stage, epochs in (('head', 10), ('joint', 60)):
                opt, evidence = optimizer(model, stage)
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
                    if stage == 'head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6))
                for stage_epoch in range(epochs):
                    epoch_tick=time.monotonic();io_before=io_seconds
                    state['phase']=stage;begin_epoch_reservation(state,len(train));record()
                    training_rates = [g['lr'] for g in opt.param_groups]
                    model.train(); losses = []
                    for batch in train:
                        batch = batch.to('cuda:0'); opt.zero_grad(set_to_none=True)
                        output = model(batch); loss = nn.functional.mse_loss(output, batch.y.view(-1))
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        if state['attempted_updates'] >= expected: raise RuntimeError('unit update limit reached')
                        state['attempted_updates'] += 1
                        opt.step(); state['completed_updates'] += 1
                        losses.append(float(loss.detach()))
                    scheduler.step()
                    state['validation_started'] += 1
                    y, p, met = evaluate()
                    state['validation_completed'] += 1; state['completed_epochs'] += 1
                    rec = dict(epoch=state['completed_epochs'], stage=stage, stage_epoch=stage_epoch+1,
                        train_loss=float(np.mean(losses)), validation_r2=met['r2'], validation_rmse=met['rmse'],
                        learning_rates=training_rates, optimizer_groups=evidence,elapsed_seconds=time.monotonic()-unit_started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),relations_last_batch=int(batch.lga_spd.numel()))

                    if met['r2'] > best_r2:
                        best_r2 = met['r2']; best_epoch = state['completed_epochs']; best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        timed_write(folder/'best.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE,
                            config=cfg, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            best_epoch=best_epoch, best_stage=best_stage, scaler=scaler_json, outer_test='NOT_RUN'),checkpoint=True)
                    rec['epoch_seconds_before_history']=time.monotonic()-epoch_tick
                    rec['write_seconds_before_history']=io_seconds-io_before
                    history.append(rec);timed_write(folder/'history.json',history)
                    close_epoch_reservation(state);record()
                    epoch_timings.append(dict(epoch=state['completed_epochs'],epoch_seconds=time.monotonic()-epoch_tick,
                        write_seconds=io_seconds-io_before,write_seconds_cumulative=io_seconds,write_calls_cumulative=io_calls,
                        timing_scope='real epoch incl CPU/model/evaluation/write; no added CUDA synchronization'))
                    timed_write(folder/'epoch_timing.json',epoch_timings);print(json.dumps(rec),flush=True)
            if state['attempted_updates'] != expected or state['completed_updates'] != expected:
                raise RuntimeError('incomplete optimizer budget')
            timed_write(folder/'resume.pt',dict(schema=SCHEMA,architecture=ARCHITECTURE,config=cfg,
                model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                stage=stage,stage_epoch=stage_epoch+1,epoch=state['completed_epochs'],batch_cursor='epoch_end',
                best_epoch=best_epoch,best_stage=best_stage,best_r2=best_r2,scaler=scaler_json,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),loader_rng=train.generator.get_state(),
                consumption=copy.deepcopy(state),automatic_resume=False),checkpoint=True)
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),
            y_true=finite(truth), y_pred=finite(prediction), indices=np.array([ds.indices[i] for i in ds.validation]),
            sample_keys=ds.keys[ds.validation], best_epoch=best_epoch, best_stage=best_stage)
        write(folder/'metrics.json', dict(status='COMPLETE', config=cfg, best_epoch=best_epoch,
            best_stage=best_stage, scaler=scaler_json, **result))
        state.update(status='COMPLETE', ended=now(), best_epoch=best_epoch, best_stage=best_stage,elapsed_seconds=time.monotonic()-unit_started,gpu_peak_bytes=torch.cuda.max_memory_allocated(),write_seconds=io_seconds,write_calls=io_calls); record()
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc(), ended=now()); record()
        write(folder/'failure.json', dict(error=state['error'], automatic_retry=False, time=now()))
        raise
    finally:
        if ds is not None: ds.close()



def parent_path(arm):
    if arm.startswith('A_'):return PARENT
    if arm=='D_INTEGRATE':arm=read(OUT/'selection/conditional_decisions.json')['PT_WINNER']
    return OUT/'pretrain'/arm/'deploy.pt'

def policy(arm):
    if arm.startswith('A_'):return arm[2:]
    if arm=='D_INTEGRATE':return effective_d_policy()
    return 'NONE'

def schedule(arm):
    if arm.startswith('A_'):return None
    if arm=='D_INTEGRATE':arm=read(OUT/'selection/conditional_decisions.json')['PT_WINNER']
    if arm.startswith('C_'):return [1,1,2,2,3,3]
    return [int(arm[1])]*6

def cached_downstream(arm,top,text,static,key):
    from src.dataset.ua2d_opt import read_downstream
    return read_downstream(OUT/'inputs/hop3',key,top)

CACHE=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
ELIGIBLE=ROOT/'results/ua2d_et_single_20261020/r1/sampling_repair1/eligible_indices.npy'
SOURCE=ROOT/'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
COMMON=ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/initialization/common_step0.pt'
PT_HEADS=COMMON.parent/'task_heads/V1.pt'

def aggregate(arm, formal=False):
    tasks, units, stages = {}, {}, {}
    for task in TASKS:
        values = []
        for fold in range(5):
            entry=repair_attempt(arm,task,fold,require_complete=True)
            path = unit_path(arm, task, fold, formal); m = read(path/'metrics.json'); c = read(path/'consumption.json')
            if m['status'] != 'COMPLETE' or c['status'] != 'COMPLETE' or m['config'] != config(arm, task, fold):
                raise ValueError('incomplete or inconsistent matrix')
            if not formal and (c['completed_epochs'] != 70 or c['completed_updates'] != c['attempted_updates']):
                raise ValueError('incomplete FT')
            values.append(m['r2']); key = f'{arm}/{task}/{fold}'
            stages[key] = dict(epoch=m['best_epoch'], stage=m['best_stage'])
            units[key] = dict(config=m['config'], best=str(unit_path(arm, task, fold)/'best.pt'),
                best_epoch=m['best_epoch'], best_stage=m['best_stage'], scaler=m['scaler'],
                attempt_path=str(unit_path(arm,task,fold)),attempt_job=entry['job_name'] if entry else None,
                config_path=str(unit_path(arm,task,fold)/'config.json'),scaler_path=str(unit_path(arm,task,fold)/'scaler.json'),
                test_indices=sorted(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']),
                predictions=str(path/('test_predictions.npz' if formal else 'validation_predictions.npz')))
        tasks[task] = dict(folds=values, mean=float(np.mean(finite(values))), std=float(np.std(values, ddof=1)))
    return dict(tasks=tasks, macro=float(np.mean([v['mean'] for v in tasks.values()])),
                units=units, stages=stages, head_best_count=sum(v['stage'] == 'head' for v in stages.values()))




B_ARMS=('B2_E','B3_E','B2_C','B3_C')
TIE_ORDER=('A_TOP2','A_MIX246',*B_ARMS,'C_SCHED_E','C_SCHED_C','D_INTEGRATE')

def choose(values,order):
    if not values or any(not np.isfinite(v) for v in values.values()):raise ValueError('invalid complete validation matrix')
    return max(values,key=lambda a:(values[a],-order.index(a)))

def a_decision(validation,baseline):
    winner=choose({a:validation[a]['macro'] for a in ('A_TOP2','A_MIX246')},TIE_ORDER)
    delta=validation[winner]['macro']-baseline
    return dict(winner=winner,delta=delta,trigger=delta>=.005,FT_POLICY=winner[2:] if delta>=.005 else 'NONE',threshold=.005)

def c_decision(validation):
    winner=choose({a:validation[a]['macro'] for a in ('B3_E','B3_C')},TIE_ORDER)
    reference='B2_'+winner[-1];delta=validation[winner]['macro']-validation[reference]['macro']
    return dict(U=winner[-1],winner=winner,reference=reference,delta=delta,trigger=True,required=True,
        historical_threshold=.005,historical_threshold_passed=delta>=.005,arm='C_SCHED_'+winner[-1],
        execution_policy_revision=EXECUTION_POLICY_REVISION,reason='user_requires_one_C_independent_of_validation_delta')

def factorial(values):
    pairs=dict(hop_E=('B3_E','B2_E'),hop_C=('B3_C','B2_C'),target_2=('B2_C','B2_E'),target_3=('B3_C','B3_E'))
    macro={k:values[a]['macro']-values[b]['macro'] for k,(a,b) in pairs.items()}
    macro['interaction']=macro['hop_C']-macro['hop_E']
    tasks={}
    for t in TASKS:
        effects={k:(np.array(values[a]['tasks'][t]['folds'])-np.array(values[b]['tasks'][t]['folds'])) for k,(a,b) in pairs.items()}
        effects['interaction']=effects['hop_C']-effects['hop_E']
        tasks[t]={k:dict(folds=v.tolist(),mean=float(v.mean()),std=float(v.std(ddof=1))) for k,v in effects.items()}
    return dict(macro=macro,paired_tasks=tasks,interpretation='matched 2x2 difference in differences; no independent seed stability claim')

def final_decision(validation,baseline,pt_winner):
    scores={a:v['macro'] for a,v in validation.items()}
    if set(scores)-set(TIE_ORDER) or not np.isfinite(baseline):raise ValueError('undeclared candidate or invalid reference')
    final=choose(scores,TIE_ORDER);delta=scores[final]-baseline
    c_arms=[a for a in validation if a in ('C_SCHED_E','C_SCHED_C')]
    if len(c_arms)!=1 or 'D_INTEGRATE' not in validation:raise ValueError('required C/D validation incomplete')
    c=c_arms[0]
    if pt_winner not in (*B_ARMS,c):raise ValueError('invalid required D parent')
    models=([final,c,'D_INTEGRATE','B3_'+c[-1],pt_winner] if final.startswith('A_')
        else [*B_ARMS,c,'D_INTEGRATE'])
    models=list(dict.fromkeys(models));cap=5 if final.startswith('A_') else 6
    if (any(a not in validation for a in models) or len(models)>cap
        or len(models)*35>LIMITS['test_starts'] or len(models)*5875>LIMITS['test_graphs']):
        raise ValueError('incomplete required references or TEST budget exceeded')
    return dict(FINAL=final,delta_validation_baseline=delta,test_eligible=True,
        performance_threshold_test=.010,models=models,validation_scores=scores,
        reference_validation_macro=baseline,selection='validation_only_global_new_candidate',
        test_policy='single_batch_required_C_D_no_validation_gate_no_reselection',
        execution_policy_revision=EXECUTION_POLICY_REVISION,C=c,D='D_INTEGRATE',PT_WINNER=pt_winner,D_FT_POLICY='MIX246')

def freeze_final(validation,decision,conditional):
    if not decision.get('test_eligible') or not decision.get('models'):raise ValueError('missing complete TEST nomination')
    units={}
    for arm in decision['models']:
        for key,value in validation[arm]['units'].items():
            # Strict full package restoration belongs to each real TEST unit.
            value=dict(value);value['formal_output']=str(unit_path(arm,value['config']['task'],value['config']['fold'],True));units[key]=value
    p=OUT/'selection/final_frozen.json'
    with p.open('x') as f:json.dump(dict(frozen=True,time=now(),**decision,conditional=conditional,units=units),f,ensure_ascii=False,indent=2,allow_nan=False)


def test_outcome(delta):
    if delta is None:return dict(positive_test_delta=None,performance_pass=None,performance_status='未完成／不可比较')
    if not np.isfinite(delta):raise ValueError('nonfinite formal TEST delta')
    return dict(positive_test_delta=delta>0,performance_pass=delta>=.010,
        performance_status='通过' if delta>=.010 else '正增量但未达门槛' if delta>0 else '本次未观察到提升')


def deliver(validation,formal,decision,error,cpu_exit,budget):
    review=OUT/'final_review';review.mkdir(exist_ok=True)
    old=read(PARENT.parents[2]/'final_review/results.json')
    refs=dict(eea=('Periodic-TDL',.934),egb=('Periodic-TDL (+Chem, +RC)',.923),egc=('Periodic-TDL (+Chem, +RC)',.911),ei=('Periodic-TDL',.839),eps=('Periodic-TDL',.787),nc=('Periodic-TDL',.856),xc=('Periodic-TDL',.423))
    final=decision.get('FINAL') if decision else None;selected=formal.get(final)
    table={};rows=[]
    for task,(paper,ref) in refs.items():
        v=selected['tasks'][task] if selected else None
        table[task]=dict(project=v,reference=ref,paper_model=paper,delta=v['mean']-ref if v else None,status='COMPLETE' if v else '未完成／不可比较',reason=None if v else ('执行异常停止，正式TEST未完成' if error else '正式TEST未完成；不得依据validation判定性能'))
        display=f"{v['mean']:.3f}±{v['std']:.3f} | {v['mean']-ref:+.3f}" if v else '未完成／不可比较 | —'
        rows.append(f'| {task.upper() if task!="eea" else "Eea"} | {display} | {ref:.3f} | {paper} |')
    deltas={}
    if selected:
        deltas={k:selected['macro']-old['test'][k]['macro'] for k in ('B_FP','V1')}
        r10=read(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json')
        deltas['r10']=selected['macro']-r10['macro']['global']['r2']
        if 'B2_E' in formal:deltas['B2_E']=selected['macro']-formal['B2_E']['macro']
        reference=(read(OUT/'selection/conditional_decisions.json').get('PT_WINNER') if final=='D_INTEGRATE' else 'B3_'+final[-1] if final and final.startswith('C_') else None)
        if reference in formal:deltas['direct_reference']=selected['macro']-formal[reference]['macro']
    status='需返修' if error or cpu_exit!=0 else '待审查'
    result=dict(plan=PLAN,revision='r1',status=status,error=error,validation=validation,formal=formal,decision=decision,delta_test=deltas,**test_outcome(deltas.get('B_FP')),cpu_exit=cpu_exit,budget=budget,sota=table,independent_review='PENDING',aggregation='equal task means of five fold R2; std ddof1; seed42+fold',protocol='shared development folds; no proven paper protocol equivalence; Egc/Xc3380/432 vs paper4124/430')
    conditional_path=OUT/'selection/conditional_decisions.json'
    conditional=read(conditional_path) if conditional_path.exists() else {}
    if (OUT/'selection/cd_required.json').exists():
        result['execution_policy_revision']=EXECUTION_POLICY_REVISION
        result['cd_required_comparisons']=cd_formal_comparisons(formal,old['test']['B_FP'],conditional)
        write_cd_required_report(result,conditional)
    write(review/'results.json',result)
    cache_manifest=OUT/'inputs/cache_manifest.json'
    write(review/'throughput.json',dict(cache=read(cache_manifest) if cache_manifest.exists() else None,
        pretrain_last_recorded_window={p.parent.name:read(p) for p in (OUT/'pretrain').glob('*/timing.json')},
        measurement='actual training only; data_seconds includes loading and denominator communication, not pure IO; no benchmark'))
    report=f'''# UA2D-OPT r1 实际交付

状态：{status}；独立审查：待进行。执行者按DSH职责交付，本次自检不构成Codex独立审查。

FINAL：{final or '未冻结'}；正式TEST性能结论：{result['performance_status']}；CPU唯一suite exit={cpu_exit}。

停止原因：{error or '无执行异常'}

| Task | FINAL正式R² | 项目减参考 | 论文参考R² | 参考模型 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+f'''

口径：七任务等权macro；各任务五折等权R²，std(ddof=1)，seed42+fold。共享validation/TEST参与开发，不称独立盲测。数据／split／协议等价性未证明；仅作论文数值对照。未开放TEST不以validation或smoke代填。

结果：{OUT}/final_review/results.json；实际预算：{OUT}/budget.json；日志及真实退出码：{LOG}；完整已执行FT：{OUT}/finetune；正式预测仅在实际开放时存在。

PT／FT失败及部分运行计账，未重试／恢复／换卡／改batch。C/D按r1-CD-REQUIRED固定执行，异常停止及未完成阶段如实记录，不转移或扩大预算。
'''
    (review/'report.md').write_text(report)
    if (review/'cd_required_report.md').exists():report+='\n\n'+(review/'cd_required_report.md').read_text()
    update_plan(status,'执行收尾',report)
    return status


EXECUTION_POLICY_REVISION='r1-CD-REQUIRED'
JOB_IDENTITY_FIELDS=('name','kind','arm','gpu','argv','env','command','cwd','log','started','consumption','window')


def job_identity(j):return {k:j.get(k) for k in JOB_IDENTITY_FIELDS}


def validate_cd_handoff(request,budget,manifest):
    """Accept only this single phase-specific administrative handoff, never retry."""
    if (request.get('plan')!=PLAN or request.get('execution_policy_revision')!=EXECUTION_POLICY_REVISION
        or request.get('state')!='controller_stopped_workers_untouched'
        or request.get('expected_limits')!=LIMITS or budget.get('limits')!=LIMITS
        or request.get('wallclock_started_unix')!=1792541198.175118
        or manifest.get('wallclock_started_unix')!=request['wallclock_started_unix']
        or not request.get('c_io1_deployed_at')):raise ValueError('CD-required handoff policy/clock/limits mismatch')
    jobs=budget['jobs'];names=[j['name'] for j in jobs]
    if len(names)!=len(set(names)) or request.get('jobs')!=len(jobs):raise ValueError('duplicate or missing adopted job')
    if request.get('job_identities')!=[job_identity(j) for j in jobs]:raise ValueError('adopted job identity changed')
    prior_arms=('A_TOP2','A_MIX246','B2_E','B3_E','B2_C')
    expected={f'opt_ft_{a}_{t}_{f}' for a in prior_arms for t in TASKS for f in range(5)}
    expected|={'opt_prepare','opt_initialize',*[f'opt_pt_{a}' for a in B_ARMS]}
    remaining={f'opt_ft_B3_C_{t}_{f}' for t in TASKS for f in range(5)}
    if not expected.issubset(names) or set(names)-expected-remaining:raise ValueError('handoff outside B3_C FT phase; replay forbidden')
    if any(j['exit']!=0 for j in jobs if j['name'] in expected):raise ValueError('historical unit incomplete or failed; no replay')
    inflight=[j for j in jobs if j['exit'] is None]
    if not inflight or any(j['name'] not in remaining for j in inflight) or any(j['exit'] not in (None,0) for j in jobs):raise ValueError('handoff requires healthy in-flight B3_C FT')
    used=budget['used']
    if (used['prepare_starts']!=1 or used['initialization_starts']!=1 or used['pt_starts']!=4
        or used['pt_updates']!=20000 or used['pt_presentations']!=20160000
        or used['test_starts']!=0 or used['cpu_suites']!=0
        or used['ft_starts']!=175+len(set(names)&remaining)):raise ValueError('handoff cumulative budget/phase mismatch')
    for j in jobs:
        if j['kind']=='pt':
            ranks=j.get('rank_progress',[])
            if len(ranks)!=2 or any(x.get('status')!='COMPLETE' or x.get('completed_updates')!=5000 or x.get('consumed_presentations')!=2520000 for x in ranks):raise ValueError('adopted PT not complete')
    return copy.deepcopy(jobs)


def claim_cd_handoff(path,payload):
    with Path(path).open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False)


def effective_d_policy(path=None):
    repair_policy=OUT/'repair_ft1/effective_policy.json'
    x=read(path or (repair_policy if repair_policy.exists() else OUT/'selection/cd_required.json'))
    if (x.get('execution_policy_revision') not in (EXECUTION_POLICY_REVISION,'r1-FT-REPAIR1')
        or x.get('C',{}).get('required') is not True
        or x.get('D',{}).get('required') is not True
        or x['D'].get('FT_POLICY')!='MIX246'):raise ValueError('missing or inconsistent required D policy')
    return x['D']['FT_POLICY']


def cd_formal_comparisons(formal,baseline,conditional):
    c=conditional.get('C',{}).get('arm');parent=conditional.get('PT_WINNER')
    comparisons={}
    for label,arm,ref in (('C_vs_direct',c,'B3_'+c[-1] if c else None),
        ('D_vs_direct','D_INTEGRATE',parent),('C_vs_B_FP',c,'B_FP'),('D_vs_B_FP','D_INTEGRATE','B_FP')):
        project=formal.get(arm);reference=baseline if ref=='B_FP' else formal.get(ref)
        if not project or not reference:
            comparisons[label]=dict(arm=arm,reference=ref,delta=None,**test_outcome(None));continue
        delta=project['macro']-reference['macro'];tasks={}
        for t in TASKS:
            paired=np.array(project['tasks'][t]['folds'])-np.array(reference['tasks'][t]['folds'])
            tasks[t]=dict(folds=paired.tolist(),mean=float(paired.mean()),std=float(paired.std(ddof=1)))
        comparisons[label]=dict(arm=arm,reference=ref,delta=delta,paired_tasks=tasks,**test_outcome(delta))
    return comparisons


def write_cd_required_report(result,conditional):
    folder=OUT/'final_review';lines=['# r1-CD-REQUIRED 实际执行交付','',
        f"状态：{result['status']}；独立审查：待Codex审查。DSH执行自检不构成独立审查。",'',
        '旧a_policy.json保留，旧NONE仅为历史开发决策；有效政策另存selection/cd_required.json。C无validation增量准入门槛，D固定MIX246。',
        '仅一次显式controller交接，从B3_C未完成FT接续；原准备、初始化、成功和在途单元、累计预算及72小时起点均保留。',
        'C及D使用原预留额度；无新增候选、重复准备、训练retry/resume或额外测试。命令／窗口／真实exit及消耗见budget.json和controller_handoff_cd_required/。','',
        f"FINAL：{result.get('decision',{}).get('FINAL') if result.get('decision') else '未冻结'}；CPU exit={result.get('cpu_exit')}。",'',
        '| 正式对照 | Δmacro R² | 依据未舍入值的记录 |','|---|---:|---|']
    for name,c in result.get('cd_required_comparisons',{}).items():
        delta=f"{c['delta']:+.3f}" if c['delta'] is not None else '未完成／不可比较'
        lines.append(f"| {name} | {delta} | {c['performance_status']} |")
    refs={'eea':('Periodic-TDL',.934),'egb':('Periodic-TDL (+Chem, +RC)',.923),'egc':('Periodic-TDL (+Chem, +RC)',.911),'ei':('Periodic-TDL',.839),'eps':('Periodic-TDL',.787),'nc':('Periodic-TDL',.856),'xc':('Periodic-TDL',.423)}
    for arm in (conditional.get('C',{}).get('arm'),'D_INTEGRATE'):
        lines+=['',f'## {arm or "C未确定"} 七任务论文数值对照','',
            '| Task | 正式R² | 论文参考 | 项目减参考 | 参考模型 |','|---|---:|---:|---:|---|']
        values=result['formal'].get(arm)
        for task,(model,ref) in refs.items():
            v=values['tasks'][task] if values else None
            score=f"{v['mean']:.3f}±{v['std']:.3f}" if v else '未完成／不可比较';delta=f"{v['mean']-ref:+.3f}" if v else '—'
            lines.append(f'| {task} | {score} | {ref:.3f} | {delta} | {model} |')
    lines+=['','统计口径：每任务五折R²等权mean、std(ddof=1)，seed42+fold，再七任务等权macro；差值与门槛使用完整精度。',
        '数据／split／协议等价性未证明，仅作论文数值对照；共享TEST不称独立盲测。完整B四臂未正式评估时不声称2×2 TEST因子结论。',
        '完整配置、原始预测、scaler、history、selection、预算、七任务FINAL表及完整精度配对差值见final_review/results.json、report.md及原产物路径。',
        f"异常：{result.get('error') or '无执行异常'}。"]
    (folder/'cd_required_report.md').write_text('\n'.join(lines)+'\n')
