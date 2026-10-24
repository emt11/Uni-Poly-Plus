"""Bounded two-arm atom/bond FT campaign; zero PT, validation-gated TEST."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import traceback
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.dataset.ua2d_demol import build, collate, SEMANTICS
from src.modules.ua2d_demol import initialize, restore, optimizer, ARCHITECTURE, SCHEMA
from src.training.glt_dual_runtime import require_tmux, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'UA2D-DEMOL-LITE-20261020-01'
OUT = ROOT/'results/ua2d_demol_20261020/r1'
LOG = ROOT/'logs/ua2d_demol_20261020/r1'
DOWN = ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT/'data/splits/mips_paper5_outer'
PARENT = ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS = ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
ARMS = ('BOND_SELF', 'BOND_NEIGHBOR')
LIMITS = dict(pt_starts=0, ft_starts=70, epochs_started=4900, validation_started=4900,
              attempted_updates=112000, test_starts=70, test_graphs=11750, cpu_suites=1)
PROFILE = dict(batch=24, amp='fp32', workers=0, drop_last=False, head_epochs=10,
    joint_epochs=60, weight_decay=.02, clip=5., encoder_dropout=0., head_dropout=.1,
    head_lr=.0003, inherited_joint_lr=.0001, new_joint_lr=.0003, head_joint_lr=.001,
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
        parent=str(PARENT), head_initial=str(HEADS/task/f'fold{fold}.pt'),
        source=str(DOWN), split=str(SPLIT/f'{task}.json'), architecture=ARCHITECTURE,
        profile=PROFILE, adapter_lr_policy='constant_3e-4_joint_only', input_semantics=SEMANTICS)


def unit_path(arm, task, fold, formal=False):
    return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'


def update_plan(status, phase, detail):
    p = ROOT/'CURRENT_PLAN.md'
    text = p.read_text()
    marker = '\n### DEMOL 最新执行登记\n'
    end = '\n<!-- DEMOL_EXECUTION_END -->'
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
    def __init__(self,task,fold,formal=False):
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
        from src.dataset.ua2d_demol import build
        from scripts.finetune_mcl_ph import clone_clean_record
        if i not in self.cache:
            top,_,text,static,_=pretrain_source_fields(self.source,i)
            self.cache[i]=build(top,text,static=static)
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
    folder = unit_path(arm, task, fold, formal); folder.mkdir(parents=True, exist_ok=False)
    cfg = config(arm, task, fold); write(folder/'config.json', cfg)
    state = dict(status='RUNNING', pid=os.getpid(), started=now(), completed_epochs=0,
        epochs_started=0, validation_started=0, validation_completed=0, attempted_updates=0,
        completed_updates=0, test_started=0, test_graphs=0, phase='initialize')
    def record(): write(folder/'consumption.json', state)
    record()
    ds = None
    try:
        seed_all(42+fold)
        if formal:
            frozen = read(OUT/'selection/final_frozen.json')
            if frozen.get('frozen') is not True: raise ValueError('TEST before freeze')
            selected = frozen['units'][f'{arm}/{task}/{fold}']
            saved = torch.load(unit_path(arm, task, fold)/'best.pt', map_location='cpu', weights_only=False)
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
            parent = torch.load(PARENT, map_location='cpu', weights_only=False)
            head = torch.load(cfg['head_initial'], map_location='cpu', weights_only=False)
            model = initialize(parent, head, arm); del parent
        ds = Data(task, fold, formal)
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
            if expected > 112000: raise RuntimeError('unit update reservation impossible')
            seed_all(42+fold); history = []; best_r2 = -math.inf
            for stage, epochs in (('head', 10), ('joint', 60)):
                opt, evidence = optimizer(model, stage)
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
                    if stage == 'head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6))
                for stage_epoch in range(epochs):
                    state.update(phase=stage, epochs_started=state['epochs_started']+1); record()
                    if stage == 'joint':
                        for group in opt.param_groups:
                            if group['name'] == 'adapter': group['lr'] = 3e-4
                    training_rates = [g['lr'] for g in opt.param_groups]
                    model.train(); losses = []
                    for batch in train:
                        batch = batch.to('cuda:0'); opt.zero_grad(set_to_none=True)
                        output = model(batch); loss = nn.functional.mse_loss(output, batch.y.view(-1))
                        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
                        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                        if state['attempted_updates'] >= expected: raise RuntimeError('unit update limit reached')
                        state['attempted_updates'] += 1; record()
                        opt.step(); state['completed_updates'] += 1; record()
                        losses.append(float(loss.detach()))
                    scheduler.step()
                    state['validation_started'] += 1; record()
                    y, p, met = evaluate()
                    state['validation_completed'] += 1; state['completed_epochs'] += 1; record()
                    rec = dict(epoch=state['completed_epochs'], stage=stage, stage_epoch=stage_epoch+1,
                        train_loss=float(np.mean(losses)), validation_r2=met['r2'], validation_rmse=met['rmse'],
                        learning_rates=training_rates, optimizer_groups=evidence)
                    history.append(rec); write(folder/'history.json', history); print(json.dumps(rec), flush=True)
                    if met['r2'] > best_r2:
                        best_r2 = met['r2']; best_epoch = state['completed_epochs']; best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        save_checkpoint(folder/'best.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE,
                            config=cfg, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            best_epoch=best_epoch, best_stage=best_stage, scaler=scaler_json, outer_test='NOT_RUN'))
                    save_checkpoint(folder/'resume.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE, config=cfg,
                        model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        stage=stage, stage_epoch=stage_epoch+1, epoch=state['completed_epochs'], batch_cursor='epoch_end',
                        best_epoch=best_epoch, best_stage=best_stage, best_r2=best_r2, scaler=scaler_json,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        numpy_rng=np.random.get_state(), python_rng=random.getstate(), loader_rng=train.generator.get_state(),
                        consumption=copy.deepcopy(state), automatic_resume=False))
            if state['attempted_updates'] != expected or state['completed_updates'] != expected:
                raise RuntimeError('incomplete optimizer budget')
        np.savez(folder/('test_predictions.npz' if formal else 'validation_predictions.npz'),
            y_true=finite(truth), y_pred=finite(prediction), indices=np.array([ds.indices[i] for i in ds.validation]),
            sample_keys=ds.keys[ds.validation], best_epoch=best_epoch, best_stage=best_stage)
        write(folder/'metrics.json', dict(status='COMPLETE', config=cfg, best_epoch=best_epoch,
            best_stage=best_stage, scaler=scaler_json, **result))
        state.update(status='COMPLETE', ended=now(), best_epoch=best_epoch, best_stage=best_stage); record()
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc(), ended=now()); record()
        write(folder/'failure.json', dict(error=state['error'], automatic_retry=False, time=now()))
        raise
    finally:
        if ds is not None: ds.close()


def aggregate(arm, formal=False):
    tasks, units, stages = {}, {}, {}
    for task in TASKS:
        values = []
        for fold in range(5):
            path = unit_path(arm, task, fold, formal); m = read(path/'metrics.json'); c = read(path/'consumption.json')
            if m['status'] != 'COMPLETE' or c['status'] != 'COMPLETE' or m['config'] != config(arm, task, fold):
                raise ValueError('incomplete or inconsistent matrix')
            if not formal and (c['completed_epochs'] != 70 or c['completed_updates'] != c['attempted_updates']):
                raise ValueError('incomplete FT')
            values.append(m['r2']); key = f'{arm}/{task}/{fold}'
            stages[key] = dict(epoch=m['best_epoch'], stage=m['best_stage'])
            units[key] = dict(config=m['config'], best=str(unit_path(arm, task, fold)/'best.pt'),
                best_epoch=m['best_epoch'], best_stage=m['best_stage'], scaler=m['scaler'],
                test_indices=sorted(read(SPLIT/f'{task}.json')['folds'][fold]['test_indices']),
                predictions=str(path/('test_predictions.npz' if formal else 'validation_predictions.npz')))
        tasks[task] = dict(folds=values, mean=float(np.mean(finite(values))), std=float(np.std(values, ddof=1)))
    return dict(tasks=tasks, macro=float(np.mean([v['mean'] for v in tasks.values()])),
                units=units, stages=stages, head_best_count=sum(v['stage'] == 'head' for v in stages.values()))




def reference_validation():
    tasks = {}; units = {}
    for task in TASKS:
        values = []
        for fold in range(5):
            path = PARENT.parents[2]/'finetune/B_FP'/task/f'fold{fold}'
            m = read(path/'metrics.json')
            cfg = m['config']
            if m['status'] != 'COMPLETE' or cfg['arm'] != 'B_FP' or cfg['parent'] != str(PARENT):
                raise ValueError('B_FP reference identity mismatch')
            if cfg['seed'] != 42+fold: raise ValueError('B_FP seed mismatch')
            # Verify actual validation indices/labels, without another inference.
            pred = np.load(path/'validation_predictions.npz')
            entry = read(SPLIT/f'{task}.json')['folds'][fold]
            if pred['indices'].tolist() != entry['validation_indices']:
                raise ValueError('reference validation split mismatch')
            values.append(m['r2']); units[f'{task}/{fold}'] = str(path)
        tasks[task] = dict(folds=values, mean=float(np.mean(finite(values))), std=float(np.std(values, ddof=1)))
    return dict(tasks=tasks, macro=float(np.mean([x['mean'] for x in tasks.values()])), units=units)


def development_gate(neighbor, self_score, baseline):
    finite([neighbor, self_score, baseline])
    return dict(neighbor=float(neighbor), self_score=float(self_score), baseline=float(baseline),
        delta_self=float(neighbor-self_score), delta_baseline=float(neighbor-baseline), threshold=.010,
        passed=bool(neighbor-self_score >= .010 and neighbor-baseline >= .010),
        selection='validation_only_full_precision')


def decide_and_freeze():
    results = {a: aggregate(a) for a in ARMS}
    reference = reference_validation()
    decision = development_gate(results['BOND_NEIGHBOR']['macro'], results['BOND_SELF']['macro'], reference['macro'])
    decision.update(time=now(), validation=results, reference=reference)
    write(OUT/'selection/development_decision.json', decision)
    if decision['passed']:
        path = OUT/'selection/final_frozen.json'
        if path.exists(): raise FileExistsError('freeze already exists')
        write(path, dict(frozen=True, time=now(), plan=PLAN, revision='r1', parent=str(PARENT),
            architecture=ARCHITECTURE, decision=decision, policy='both_arms_once_after_joint_validation_gate',
            units={k:v for r in results.values() for k,v in r['units'].items()}))
    return decision


def initialize_sources():
    reference = reference_validation()
    if not PARENT.is_file(): raise FileNotFoundError(PARENT)
    for task in TASKS:
        manifest = read(SPLIT/f'{task}.json')
        if manifest.get('validation_is_test') is not False: raise ValueError('invalid validation separation')
        for fold in range(5):
            if not (HEADS/task/f'fold{fold}.pt').is_file(): raise FileNotFoundError('missing head')
            e = manifest['folds'][fold]
            a,b,c = (set(e[k]) for k in ('train_indices','validation_indices','test_indices'))
            if a&b or a&c or b&c: raise ValueError('overlapping split')
    write(OUT/'initialization/sources.json', dict(parent=str(PARENT), heads=str(HEADS), split=str(SPLIT),
        reference=reference, adapter_seed=20261020, public_adapter='isolated_rng_same_constructor_both_arms',
        input_semantics=SEMANTICS, no_hash_validation=True, no_new_cache=True))


def final_report(status, failure=None):
    folder = OUT/'final_review'; folder.mkdir(exist_ok=True)
    result = dict(plan=PLAN, revision='r1', status=status, time=now(), failure=failure,
        independent_review='PENDING', budget=read(OUT/'budget.json'))
    decision_path = OUT/'selection/development_decision.json'
    decision = read(decision_path) if decision_path.exists() else None
    result['development_decision'] = decision
    for kind in ('validation', 'test'):
        result[kind] = {}
        for arm in ARMS:
            try: result[kind][arm] = aggregate(arm, formal=kind == 'test')
            except (FileNotFoundError, ValueError) as error:
                reason = 'validation双门槛未通过，正式TEST=0' if kind == 'test' and decision and not decision['passed'] else str(error)
                result[kind][arm] = dict(status='未完成／不可比较', reason=reason)
    if all('macro' in result['test'][a] for a in ARMS):
        reference = read(ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json')
        def macro(x):
            if isinstance(x, (int,float)): return float(x)
            for key in ('macro','global','r2'):
                if key in x: return macro(x[key])
            raise ValueError('unsupported macro')
        result['formal_gate'] = development_gate(result['test']['BOND_NEIGHBOR']['macro'],
            result['test']['BOND_SELF']['macro'], macro(reference['test']['B_FP']))
    refs = dict(eea=(.934,'Periodic-TDL'), egb=(.923,'Periodic-TDL (+Chem, +RC)'),
        egc=(.911,'Periodic-TDL (+Chem, +RC)'), ei=(.839,'Periodic-TDL'), eps=(.787,'Periodic-TDL'),
        nc=(.856,'Periodic-TDL'), xc=(.423,'Periodic-TDL'))
    result['sota'] = {}
    lines = [f'# {PLAN} / r1 执行交付', '', f'状态：{status}；独立审查未完成。', '',
        '五折seed42—46，std ddof=1，七任务等权macro；完整精度判定，展示三位小数。',
        '共享validation／TEST不是独立盲测；论文协议未证明一致，仅数值对照。', '',
        '|任务|论文模型|参考R²|BOND_SELF R²／差值|BOND_NEIGHBOR R²／差值|', '|---|---|---:|---|---|']
    if decision:
        lines[4:4] = [f"validation：NEIGHBOR−SELF={decision['delta_self']:+.3f}；NEIGHBOR−B_FP={decision['delta_baseline']:+.3f}；门槛通过={decision['passed']}。", '']
    for task,(ref,paper) in refs.items():
        row = dict(reference=ref, reference_model=paper, arms={}); cells = []
        for arm in ARMS:
            r = result['test'][arm]
            if 'tasks' in r:
                m = r['tasks'][task]
                row['arms'][arm] = dict(r2=m['mean'], std=m['std'], delta=m['mean']-ref,
                    formal_path=str(OUT/'formal'/arm/task), protocol='5fold seed42-46 ddof1 shared_test')
                cells.append(f"{m['mean']:.3f}±{m['std']:.3f}／{m['mean']-ref:+.3f}")
            else:
                row['arms'][arm] = r; cells.append('未完成／不可比较（'+r['reason']+'）')
        result['sota'][task] = row
        lines.append(f'|{task}|{paper}|{ref:.3f}|{cells[0]}|{cells[1]}|')
    cpu = [j for j in result['budget']['jobs'] if j['kind']=='cpu']
    result['cpu'] = cpu
    write(folder/'results.json', result)
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    return result
