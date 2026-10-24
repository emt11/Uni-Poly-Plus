"""Isolated frozen-3D single candidate; explicit read-only r1 inputs."""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import traceback
import numpy as np
import torch
from torch import nn
from src.training.ua_hop_ph import ROOT, DOWN, SPLIT, PARENT, HEADS, TASKS, PROFILE, now, read, write, finite, seed_all, loader, metrics
from src.training.glt_dual_runtime import require_tmux, save_checkpoint
from src.dataset.ua_hop_ph import SEMANTICS
from src.modules.ua_hop_ph_frozen3d import initialize, restore, optimizer, ARCHITECTURE, SCHEMA, CANDIDATE, POLICY

PLAN = 'UA-HOP-PH-FROZEN3D-20261020-01'
OUT = ROOT/'results/ua_hop_ph_20261020/frozen3d_r1'
LOG = ROOT/'logs/ua_hop_ph_20261020/frozen3d_r1'
REFERENCE = ROOT/'results/ua_hop_ph_20261020/r1'
SIDECAR = REFERENCE/'sidecar'
ARMS = (CANDIDATE,)
LIMITS = dict(pt_starts=0, ft_starts=35, epochs_started=2450, validation_started=2450,
              attempted_updates=56000, test_starts=35, test_graphs=5875, cpu_suites=1)


def config(arm, task, fold):
    if arm != CANDIDATE or task not in TASKS or fold not in range(5):
        raise ValueError('outside single-candidate matrix')
    return dict(plan=PLAN, revision='r1', arm=arm, candidate_id=CANDIDATE,
        trainability_policy=POLICY, task=task, fold=fold, seed=42+fold,
        parent=str(PARENT), head_initial=str(HEADS/task/f'fold{fold}.pt'),
        adapter_initial=str(REFERENCE/'initialization/common_adapter.pt'),
        router_initial=str(REFERENCE/'initialization/ph_router.pt'),
        source=str(DOWN), sidecar=str(SIDECAR), split=str(SPLIT/f'{task}.json'),
        architecture=ARCHITECTURE, profile=PROFILE, input_semantics=SEMANTICS)


def unit_path(arm, task, fold, formal=False):
    return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'


def update_plan(status, phase, detail):
    p=ROOT/'CURRENT_PLAN.md'
    before, block=p.read_text().split('## UA-HOP-PH-FROZEN3D-20261020-01 / r1',1)
    lines=block.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('- **计划ID：UA-HOP-PH-FROZEN3D'):
            lines[i]=f'- **计划ID：{PLAN}；revision：r1；状态：{status}／{phase}。** 已获用户明确执行授权；DSH自检不代替独立审查。'
            break
    block='\n'.join(lines).split('\n\n### FROZEN3D DSH最新执行记录')[0].rstrip()
    block+=f'\n\n### FROZEN3D DSH最新执行记录\n\n- 时间：{now()}；状态：{status}；阶段：{phase}。\n- {detail}\n- 结果根：{OUT}；日志根：{LOG}。实际消耗见budget.json；历史r1只读保留。\n'
    p.write_text(before+'## UA-HOP-PH-FROZEN3D-20261020-01 / r1'+block)
    write(OUT/'status.json',dict(plan=PLAN,revision='r1',status=status,phase=phase,time=now(),detail=detail,independent_review='PENDING'))

class Data(torch.utils.data.Dataset):
    def __init__(self, task, fold, formal=False):
        m = read(SPLIT/f'{task}.json'); entry = m['folds'][fold]
        if m.get('validation_is_test') is not False: raise ValueError('validation/test mixed')
        self.rows = read(SIDECAR/'tasks'/f'{task}.json')
        if formal:
            self.indices = sorted(entry['test_indices']); self.train = []
            self.validation = list(range(len(self.indices)))
        else:
            self.indices = sorted(set(entry['train_indices']) | set(entry['validation_indices']))
            at = {index: j for j, index in enumerate(self.indices)}
            self.train = [at[i] for i in entry['train_indices']]
            self.validation = [at[i] for i in entry['validation_indices']]
        self.selected = [self.rows[i] for i in self.indices]
        if any(r['original_row'] != i for i, r in zip(self.indices, self.selected)):
            raise ValueError('task row binding mismatch')
        self.raw_targets = finite([r['label'] for r in self.selected])
        self.keys = np.array([r['sample_key'] for r in self.selected])
        self.targets = self.raw_targets.copy(); self.cache = {}

    def set_target_override(self, y): self.targets = np.asarray(y, dtype=np.float64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        key = self.keys[i]
        if key not in self.cache:
            x = torch.load(SIDECAR/'samples'/f'{key}.pt', map_location='cpu', weights_only=False)
            if (x['schema'] != SEMANTICS or x['key'] != key
                    or x['normalized_smiles'] != self.selected[i]['normalized_smiles']):
                raise ValueError('sidecar schema/identity mismatch')
            self.cache[key] = x['data']
        d = self.cache[key].clone(); d.y = torch.tensor([self.targets[i]], dtype=torch.float32)
        return d


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
            common = torch.load(REFERENCE/'initialization/common_adapter.pt', map_location='cpu', weights_only=False)
            model.adapter.load_state_dict(common, strict=True)
            if arm == CANDIDATE:
                model.router.load_state_dict(torch.load(REFERENCE/'initialization/ph_router.pt', map_location='cpu', weights_only=False), strict=True)
        ds = Data(task, fold, formal)
        if formal:
            if selected['test_indices'] != ds.indices: raise ValueError('frozen TEST indices mismatch')
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=ds.train, transform_mode='standard')
            scaler_json = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(),
                               fit='train', transform='standard')
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
            seed_all(42+fold); history = []; best_r2 = -math.inf
            for stage, epochs in (('head', 10), ('joint', 60)):
                opt, evidence = optimizer(model, stage)
                scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=3e-5)
                    if stage == 'head' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6))
                for stage_epoch in range(epochs):
                    state.update(phase=stage, epochs_started=state['epochs_started']+1); record()
                    model.train(); losses = []
                    for batch in train:
                        batch = batch.to('cuda:0'); opt.zero_grad(set_to_none=True)
                        output = model(batch); loss = nn.functional.mse_loss(output, batch.y)
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
                        learning_rates=[g['lr'] for g in opt.param_groups], optimizer_groups=evidence)
                    history.append(rec); write(folder/'history.json', history); print(json.dumps(rec), flush=True)
                    if met['r2'] > best_r2:
                        best_r2 = met['r2']; best_epoch = state['completed_epochs']; best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        save_checkpoint(folder/'best.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE, candidate_id=CANDIDATE, trainability_policy=POLICY,
                            config=cfg, state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            best_epoch=best_epoch, best_stage=best_stage, scaler=scaler_json, outer_test='NOT_RUN'))
                    save_checkpoint(folder/'resume.pt', dict(schema=SCHEMA, architecture=ARCHITECTURE, candidate_id=CANDIDATE, trainability_policy=POLICY, config=cfg,
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
    """The development reference comes exclusively from the 35 old FT metrics."""
    tasks = {}; units = {}
    for task in TASKS:
        values = []
        for fold in range(5):
            path = REFERENCE/'finetune/HOP_PH'/task/f'fold{fold}/metrics.json'
            m = read(path)
            if (m['status'] != 'COMPLETE' or m['config']['arm'] != 'HOP_PH'
                    or m['config']['parent'] != str(PARENT) or m['config']['seed'] != 42+fold):
                raise ValueError('r1 FT reference identity mismatch')
            values.append(float(m['r2'])); units[f'{task}/{fold}'] = str(path)
        finite(values)
        tasks[task] = dict(folds=values, mean=float(np.mean(values)), std=float(np.std(values, ddof=1)))
    return dict(tasks=tasks, macro=float(np.mean([v['mean'] for v in tasks.values()])), units=units)


def development_gate(candidate_macro, reference_macro):
    delta = float(candidate_macro)-float(reference_macro)
    finite([candidate_macro, reference_macro, delta])
    return dict(candidate_macro=float(candidate_macro), reference_macro=float(reference_macro),
                delta=delta, threshold=.010, passed=delta >= .010, selection='validation_only_full_precision')


def decide_and_freeze():
    candidate = aggregate(CANDIDATE)
    reference = reference_validation()
    decision = development_gate(candidate['macro'], reference['macro'])
    decision.update(plan=PLAN, candidate_id=CANDIDATE, time=now(),
        reference_units=reference['units'], test_units_authorized=35 if decision['passed'] else 0,
        no_test_policy='validation gate failure implies no formal predictions')
    path = OUT/'selection/development_decision.json'
    if path.exists(): raise FileExistsError('development decision already exists; no rerun')
    write(path, decision)
    if decision['passed']:
        frozen_path = OUT/'selection/final_frozen.json'
        if frozen_path.exists(): raise FileExistsError('freeze already exists')
        write(frozen_path, dict(frozen=True, plan=PLAN, revision='r1', candidate_id=CANDIDATE,
            trainability_policy=POLICY, architecture=ARCHITECTURE, parent=str(PARENT),
            time=now(), decision=decision, units=candidate['units']))
    return decision


def initialize_sources():
    """Only bind existing inputs; no preparation job, scan or embedding cache."""
    if read(SIDECAR/'semantics.json') != SEMANTICS:
        raise ValueError('read-only r1 input semantics mismatch')
    old = read(REFERENCE/'initialization/sources.json')
    if old['parent'] != str(PARENT) or old['step'] != 5000:
        raise ValueError('r1 common initialization parent mismatch')
    paths = [PARENT, REFERENCE/'initialization/common_adapter.pt', REFERENCE/'initialization/ph_router.pt']
    paths += [HEADS/task/f'fold{fold}.pt' for task in TASKS for fold in range(5)]
    paths += [SIDECAR/'tasks'/f'{task}.json' for task in TASKS]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    write(OUT/'initialization/sources.json', dict(parent=str(PARENT), sidecar=str(SIDECAR),
        reference=str(REFERENCE), candidate_id=CANDIDATE, trainability_policy=POLICY,
        adapter=str(REFERENCE/'initialization/common_adapter.pt'),
        router=str(REFERENCE/'initialization/ph_router.pt'), head_sources=old['head_sources'],
        input_semantics=SEMANTICS, preparation_jobs=0, embedding_cache=False,
        historical_ft_reference=reference_validation()))


def final_report(status, failure=None):
    folder = OUT/'final_review'; folder.mkdir(exist_ok=True)
    result = dict(plan=PLAN, revision='r1', candidate_id=CANDIDATE, status=status,
        time=now(), failure=failure, independent_review='PENDING', budget=read(OUT/'budget.json'),
        trainability_policy=POLICY, reference_validation=reference_validation())
    try:
        result['validation'] = aggregate(CANDIDATE)
        write(folder/'best_stage_summary.json', dict(units=result['validation']['stages'],
              head_best_count=result['validation']['head_best_count'], head_best_interpretation='zero residual; no 3D gain claim'))
    except (FileNotFoundError, ValueError) as e:
        result['validation'] = dict(status='未完成／不可比较', reason=str(e))
    decision_path = OUT/'selection/development_decision.json'
    decision = read(decision_path) if decision_path.exists() else dict(passed=False, reason='FT incomplete')
    result['development_decision'] = decision
    if decision['passed']:
        try: result['test'] = aggregate(CANDIDATE, formal=True)
        except (FileNotFoundError, ValueError) as e:
            result['test'] = dict(status='未完成／不可比较', reason=str(e))
    else:
        result['test'] = dict(status='未执行／不可比较', reason='validation门槛未通过' if decision_path.exists() else 'FT未完成')
    ref = dict(eea=(.934, 'Periodic-TDL'), egb=(.923, 'Periodic-TDL (+Chem, +RC)'),
        egc=(.911, 'Periodic-TDL (+Chem, +RC)'), ei=(.839, 'Periodic-TDL'),
        eps=(.787, 'Periodic-TDL'), nc=(.856, 'Periodic-TDL'), xc=(.423, 'Periodic-TDL'))
    result['sota'] = {}
    for task, (value, paper) in ref.items():
        row = dict(reference=value, reference_model=paper, protocol='5fold seed42-46;ddof1;shared validation/test')
        if 'tasks' in result['test']:
            t = result['test']['tasks'][task]
            row.update(r2=t['mean'], std=t['std'], delta=t['mean']-value,
                       formal_path=str(OUT/'formal'/CANDIDATE/task))
        else: row.update(status=result['test']['status'], reason=result['test']['reason'])
        result['sota'][task] = row
    if 'macro' in result['test']:
        # Historical TEST enters only the post-freeze reporting path, never the development decision.
        history = read(REFERENCE/'final_review/results.json')
        bfp = read(ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json')
        r10 = read(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json')
        r10macro = r10['macro']['global']['r2']
        dt = result['test']['macro']-history['test']['HOP_PH']['macro']
        db = result['test']['macro']-bfp['test']['B_FP']['macro']
        result['formal_gate'] = dict(test_minus_r1_ph=dt, test_minus_b_fp=db,
            test_minus_r10=result['test']['macro']-r10macro, threshold=.010,
            passed=dt >= .010 and db >= .010)
    else: result['formal_gate'] = dict(passed=None, reason='candidate formal TEST not complete or not authorized')
    write(folder/'results.json', result)
    lines = [f'# {PLAN} / r1 DSH执行交付', '', f'状态：{status}；独立审查未完成。', '',
        '唯一受控变化为encoder3d冻结，保留真实三维前向；不重跑reference。',
        '五折seed42—46、std ddof=1、七任务等权宏平均；共享validation/TEST不是独立盲测。', '']
    if 'macro' in result['validation']:
        shown_delta = result['validation']['macro']-result['reference_validation']['macro']
        lines.extend([f"validation宏R²：{result['validation']['macro']:.3f}；r1 HOP_PH：{result['reference_validation']['macro']:.3f}；差值：{shown_delta:+.3f}；门槛≥0.010：{'通过' if decision['passed'] else '未通过'}。",
            f"head阶段best：{result['validation']['head_best_count']}/35，不能解释为三维收益。", '',
            '|任务|候选validation R²|r1 PH validation R²|差值|', '|---|---:|---:|---:|'])
        for task in TASKS:
            a = result['validation']['tasks'][task]; b = result['reference_validation']['tasks'][task]
            lines.append(f"|{task}|{a['mean']:.3f}±{a['std']:.3f}|{b['mean']:.3f}±{b['std']:.3f}|{a['mean']-b['mean']:+.3f}|")
    lines.extend(['', '|任务|论文模型|参考R²|候选正式R²／差值|', '|---|---|---:|---|'])
    for task, row in result['sota'].items():
        cell = f"{row['r2']:.3f}±{row['std']:.3f}／{row['delta']:+.3f}" if 'r2' in row else f"{row['status']}（{row['reason']}）"
        lines.append(f"|{task}|{row['reference_model']}|{row['reference']:.3f}|{cell}|")
    lines.extend(['', '协议等价未证明，仅数值对照；未开放TEST不填入r1 TEST冒充候选结果。',
        f'原始结果：{folder / "results.json"}；CPU证据：{LOG / "cpu"}；失败：{failure or "无"}。'])
    report = '\n'.join(lines)+'\n'
    (folder/'report.md').write_text(report)
    (folder/'validation_report.md').write_text(report)
    write(folder/'seven_task_sota.json', result['sota'])
    cpu = [j for j in result['budget']['jobs'] if j['kind'] == 'cpu']
    write(folder/'cpu_result.json', dict(suites=len(cpu), jobs=cpu, limit_seconds=300))
    return result

