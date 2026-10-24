"""Fixed two-arm UA-HOP-PH campaign. No legacy controller is invoked."""
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

from src.dataset.ua_hop_ph import prepare_sample, collate, SEMANTICS
from src.modules.ua_hop_ph import initialize, restore, optimizer, ARCHITECTURE, SCHEMA
from src.training.glt_dual_runtime import require_tmux, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'UA-HOP-PH-20261020-01'
OUT = ROOT/'results/ua_hop_ph_20261020/r1'
LOG = ROOT/'logs/ua_hop_ph_20261020/r1'
DOWN = ROOT/'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT/'data/splits/mips_paper5_outer'
PARENT = ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/pretrain/B_FP/deploy.pt'
HEADS = ROOT/'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
ARMS = ('HOP_EQ', 'HOP_PH')
LIMITS = dict(pt_starts=0, ft_starts=70, epochs_started=4900, validation_started=4900,
              attempted_updates=112000, test_starts=70, cpu_suites=1)
PROFILE = dict(batch=24, amp='fp32', workers=0, drop_last=False, head_epochs=10,
    joint_epochs=60, weight_decay=.02, clip=5., encoder_dropout=0., head_dropout=.1,
    head_lr=.0003, inherited_joint_lr=.0001, new_joint_lr=.001, head_joint_lr=.001,
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
        profile=PROFILE, input_semantics=SEMANTICS)


def unit_path(arm, task, fold, formal=False):
    return OUT/('formal' if formal else 'finetune')/arm/task/f'fold{fold}'


def update_plan(status, phase, detail):
    p = ROOT/'CURRENT_PLAN.md'; text = p.read_text()
    marker = '## UA-HOP-PH-20261020-01 / r1'
    before, block = text.split(marker, 1)
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('- **计划 ID：UA-HOP-PH'):
            lines[i] = f'- **计划 ID：{PLAN}；revision：r1；状态：{status}／{phase}。** DSH实现自检不构成独立审查。'
            break
    entry = (f'\n\n### DSH 最新执行登记\n\n- 时间：{now()}；状态：{status}；阶段：{phase}。\n'
             f'- {detail}\n- 结果根：{OUT}；日志根：{LOG}。真实消耗以budget.json及各单元consumption.json为准；'
             '未完成项不可比较，不据此归档PROJECT_HISTORY。\n')
    block = '\n'.join(lines).split('\n\n### DSH 最新执行登记')[0].rstrip()+entry
    p.write_text(before+marker+block)
    if OUT.exists():
        write(OUT/'status.json', dict(plan=PLAN, revision='r1', status=status, phase=phase,
            time=now(), detail=detail, independent_review='PENDING'))


def prepare():
    """One normal input preparation, shared by both arms and all folds."""
    from src.dataset.mcl_ph_direct import DirectMCLSource
    require_tmux()
    folder = OUT/'sidecar'; folder.mkdir(exist_ok=False)
    (folder/'samples').mkdir(); (folder/'tasks').mkdir()
    progress = dict(status='RUNNING', unique_samples=0, rows=0, pid=os.getpid(), started=now())
    path = folder/'consumption.json'; write(path, progress)
    seen = {}; totals = dict(roots=0, complete=[0]*4, empty_shells=[0]*4,
        truncated_roots=[0]*4, affected_atoms=[0]*4, cross_image_relations=[0]*4,
        valid_samples=0, invalid_samples=0, identity_samples=0)
    try:
        write(folder/'semantics.json', SEMANTICS)
        for task in TASKS:
            manifest = read(SPLIT/f'{task}.json')
            if manifest.get('validation_is_test') is not False:
                raise ValueError('validation/test separation missing')
            source = DirectMCLSource(DOWN, task=task, selected_indices=None,
                                     split_manifest=manifest, verify_key_digest=False)
            rows = []
            try:
                for i, entry in enumerate(source.entries):
                    key = entry['sample_key']
                    identity = entry['normalized_smiles']
                    if key not in seen:
                        record = source.record(i)
                        data, cov = prepare_sample(record.topology, record.trimer,
                                                   entry['source_smiles'], record.static)
                        save_checkpoint(folder/'samples'/f'{key}.pt', dict(schema=SEMANTICS,
                            key=key, normalized_smiles=identity, data=data, coverage=cov))
                        seen[key] = identity
                        totals['identity_samples'] += 1
                        totals['valid_samples' if cov['geometry_valid'] else 'invalid_samples'] += 1
                        totals['roots'] += cov['roots']
                        for name in ('complete', 'empty_shells', 'truncated_roots', 'affected_atoms', 'cross_image_relations'):
                            totals[name] = [a+b for a, b in zip(totals[name], cov[name])]
                    elif seen[key] != identity:
                        raise ValueError('shared key has inconsistent normalized identity')
                    # Labels are reader-owned and never enter PH or shell construction.
                    rows.append(dict(original_row=i, sample_key=key, normalized_smiles=identity,
                                     label=float(entry['label'])))
                    progress.update(rows=progress['rows']+1, unique_samples=len(seen), task=task, row=i)
                    if i % 25 == 0: write(path, progress); print(json.dumps(progress), flush=True)
                write(folder/'tasks'/f'{task}.json', rows)
            finally:
                source.close()
        totals['complete_rate'] = [v/max(1, totals['roots']) for v in totals['complete']]
        totals.update(semantics=SEMANTICS, population='unique_seven_task_inputs_central_incidence_roots',
                      preparation='normal_input_preparation_only', coordinate_generation_calls=0)
        write(folder/'coverage_summary.json', totals)
        # Normal initialization: prove declared shared state and bind original heads.
        parent = torch.load(PARENT, map_location='cpu', weights_only=False)
        head_path = HEADS/'eea/fold0.pt'
        head = torch.load(head_path, map_location='cpu', weights_only=False)
        seed_all(42); eq = initialize(parent, head, 'HOP_EQ')
        seed_all(42); ph = initialize(parent, head, 'HOP_PH')
        common = eq.adapter.state_dict()
        if any(not torch.equal(v, ph.adapter.state_dict()[k]) for k, v in common.items()):
            raise ValueError('common adapter initialization drift')
        if sum(p.numel() for p in ph.router.parameters()) != 20484:
            raise ValueError('router capacity mismatch')
        init = OUT/'initialization'; init.mkdir(exist_ok=False)
        save_checkpoint(init/'common_adapter.pt', common)
        save_checkpoint(init/'ph_router.pt', ph.router.state_dict())
        binding = dict(parent=str(PARENT), schema=parent['schema'], step=parent['step'],
            config=parent['config'], common_seed=20261020, head_sources={
                f'{task}/{fold}': str(HEADS/task/f'fold{fold}.pt') for task in TASKS for fold in range(5)},
            shared_adapter=str(init/'common_adapter.pt'), router=str(init/'ph_router.pt'),
            router_extra_parameters=20484, optimizer_restoration='FT resume.pt, not PT deploy.pt')
        for p in binding['head_sources'].values():
            if not Path(p).is_file(): raise FileNotFoundError(p)
        write(init/'sources.json', binding)
        progress.update(status='COMPLETE', completed=now()); write(path, progress)
    except BaseException:
        progress.update(status='FAILED', error=traceback.format_exc(), ended=now()); write(path, progress)
        raise


class Data(torch.utils.data.Dataset):
    def __init__(self, task, fold, formal=False):
        m = read(SPLIT/f'{task}.json'); entry = m['folds'][fold]
        if m.get('validation_is_test') is not False: raise ValueError('validation/test mixed')
        self.rows = read(OUT/'sidecar/tasks'/f'{task}.json')
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
            x = torch.load(OUT/'sidecar/samples'/f'{key}.pt', map_location='cpu', weights_only=False)
            if (x['schema'] != SEMANTICS or x['key'] != key
                    or x['normalized_smiles'] != self.selected[i]['normalized_smiles']):
                raise ValueError('sidecar schema/identity mismatch')
            self.cache[key] = x['data']
        d = self.cache[key].clone(); d.y = torch.tensor([self.targets[i]], dtype=torch.float32)
        return d


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
            common = torch.load(OUT/'initialization/common_adapter.pt', map_location='cpu', weights_only=False)
            model.adapter.load_state_dict(common, strict=True)
            if arm == 'HOP_PH':
                model.router.load_state_dict(torch.load(OUT/'initialization/ph_router.pt', map_location='cpu', weights_only=False), strict=True)
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
            if expected > 112000: raise RuntimeError('unit update reservation impossible')
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


def freeze_final():
    results = {a: aggregate(a) for a in ARMS}
    path = OUT/'selection/final_frozen.json'
    if path.exists(): raise FileExistsError('freeze already exists')
    write(path, dict(frozen=True, time=now(), plan=PLAN, revision='r1', parent=str(PARENT),
        architecture=ARCHITECTURE, policy='both_predeclared_arms_once;validation_strict_best;earlier_tie',
        validation=results, units={k: v for r in results.values() for k, v in r['units'].items()}))


def final_report(status, failure=None):
    folder = OUT/'final_review'; folder.mkdir(exist_ok=True)
    result = dict(plan=PLAN, revision='r1', status=status, failure=failure, time=now(),
                  budget=read(OUT/'budget.json'), independent_review='PENDING')
    stage_summary = {}; summary = []
    for kind in ('validation', 'test'):
        values = {}
        for arm in ARMS:
            try:
                r = aggregate(arm, formal=kind == 'test'); values[arm] = r
                stage_summary[f'{kind}/{arm}'] = dict(head_best_count=r['head_best_count'], units=r['stages'])
            except (FileNotFoundError, ValueError) as e:
                values[arm] = dict(status='未完成／不可比较', reason=str(e))
        result[kind] = values
    if all('macro' in result[k].get(a, {}) for k in ('validation', 'test') for a in ARMS):
        dv = result['validation']['HOP_PH']['macro']-result['validation']['HOP_EQ']['macro']
        dt = result['test']['HOP_PH']['macro']-result['test']['HOP_EQ']['macro']
        result['gate'] = dict(validation_delta=dv, test_delta=dt, validation_positive=dv > 0,
                              test_delta_at_least_001=dt >= .010, passed=dv > 0 and dt >= .010)
    ref = dict(eea=(.934, 'Periodic-TDL'), egb=(.923, 'Periodic-TDL (+Chem, +RC)'),
        egc=(.911, 'Periodic-TDL (+Chem, +RC)'), ei=(.839, 'Periodic-TDL'),
        eps=(.787, 'Periodic-TDL'), nc=(.856, 'Periodic-TDL'), xc=(.423, 'Periodic-TDL'))
    sota = {}
    for task, (value, paper) in ref.items():
        sota[task] = dict(reference=value, reference_model=paper, arms={})
        for arm in ARMS:
            r = result['test'][arm]
            if 'tasks' in r:
                mean = r['tasks'][task]['mean']
                sota[task]['arms'][arm] = dict(r2=mean, std=r['tasks'][task]['std'], delta=mean-value,
                    formal_path=str(OUT/'formal'/arm/task), protocol='5fold seed42-46;ddof1;shared_test')
            else: sota[task]['arms'][arm] = dict(status='未完成／不可比较', reason='formal matrix incomplete')
    result['sota'] = sota
    historical = read(ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json')
    r10 = read(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json')
    # Preserve the actual source documents for review, never substitute rounded references.
    result['historical_sources'] = dict(b_fp=str(ROOT/'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json'),
                                       r10=str(ROOT/'results/fce_tune_20261016/global_blend/final_review/seven_task_results.json'))
    def macro(x):
        if isinstance(x, (int, float)): return float(x)
        if 'macro' in x: return macro(x['macro'])
        if 'global' in x: return macro(x['global'])
        if 'r2' in x: return macro(x['r2'])
        raise ValueError('unsupported historical macro structure')
    result['historical_comparison'] = {}
    for arm in ARMS:
        if 'macro' in result['test'][arm]:
            result['historical_comparison'][arm] = dict(
                test_minus_b_fp=result['test'][arm]['macro']-macro(historical['test']['B_FP']),
                validation_minus_b_fp=result['validation'][arm]['macro']-macro(historical['validation']['B_FP']),
                test_minus_r10=result['test'][arm]['macro']-macro(r10['macro']['global']))
    write(folder/'results.json', result); write(folder/'best_stage_summary.json', stage_summary)
    lines = [f'# {PLAN} / r1 DSH执行交付', '', f'状态：{status}；独立审查未完成。', '',
             '五折seed42—46，std ddof=1；七任务等权宏平均，完整精度判定，展示三位小数。',
             '共享TEST不是独立盲测；协议未证明与论文一致，仅数值对照；head阶段best不是三维／PH收益。', '',
             '|任务|论文模型|参考R²|HOP_EQ R²／差值|HOP_PH R²／差值|', '|---|---|---:|---|---|']
    for task, row in sota.items():
        cells = []
        for arm in ARMS:
            a = row['arms'][arm]
            cells.append(f"{a['r2']:.3f}±{a['std']:.3f}／{a['delta']:+.3f}" if 'r2' in a else '未完成／不可比较')
        lines.append(f"|{task}|{row['reference_model']}|{row['reference']:.3f}|{cells[0]}|{cells[1]}|")
    lines.extend(['', '原始配置、预测与消耗分别见各finetune／formal单元；CPU证据见日志cpu目录及budget.json。',
                  f'结果：{folder / "results.json"}；失败：{failure or "无"}。'])
    (folder/'report.md').write_text('\n'.join(lines)+'\n')
    return result
