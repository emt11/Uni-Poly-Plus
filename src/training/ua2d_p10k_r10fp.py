"""Isolated, array-only P10K / frozen r9 KRR policy. No model imports."""
import copy
import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'UA2D-P10K-R10FP-20261023-01'
REVISION = 'r1'
SCHEMA = 'ua2d-p10k-r10fp-raw-arrays-r1'
TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
WEIGHTS = (0., .25, .5, .75, 1.)
UNITS = {(t, f) for t in TASKS for f in range(5)}
PARENT_PLAN = 'UA2D-PTSCALE-20261022-01'
PT = ROOT / 'results/ua2d_ptscale_20261022/r1'
R9 = ROOT / 'results/fce_tune_20261016/all_task_kernel'
R10 = ROOT / 'results/fce_tune_20261016/global_blend'
OUT = ROOT / 'results/ua2d_p10k_r10fp_20261023/r1'
LOG = ROOT / 'logs/ua2d_p10k_r10fp_20261023/r1'
PARENT = PT / 'pretrain/checkpoints/step10000/deploy.pt'
SOURCE = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT / 'data/splits/mips_paper5_outer'
ENV = dict(CUDA_VISIBLE_DEVICES='', WORLD_SIZE='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
LIMITS = dict(prepare_starts=1, validation_units=35, validation_presentations=4714,
              weight_candidates=5, validation_scores=175, freezes=1, test_starts=35,
              test_graphs=5875, outputs=35, cpu_suites=1, pt_starts=0, ft_starts=0,
              epochs=0, optimizer_updates=0, krr_fit=0, refit=0, kernel_queries=0,
              fp_generation=0, graph_generation=0, conformer_generation=0,
              model_inference=0, gpu_tasks=0, workers=0, initialization_starts=0)
POLICY = dict(candidate='P10K_R10FP_GLOBAL', grid=list(WEIGHTS),
              objective='35 equal-weight validation R2', tie='larger P10K weight, exact tie',
              units='original property units', alpha='unchanged frozen r9',
              task_switching=False)


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open('x') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)


def resource_guard(env=None):
    env = os.environ if env is None else env
    if any(env.get(k) != v for k, v in ENV.items()):
        raise ValueError('CPU-only / world1 / single-thread resource mismatch')


def check_budget(used, reservation=None):
    reservation = {} if reservation is None else reservation
    if set(used) != set(LIMITS) or set(reservation) - set(LIMITS):
        raise ValueError('budget schema mismatch')
    for k, limit in LIMITS.items():
        a, b = used[k], reservation.get(k, 0)
        if type(a) is not int or type(b) is not int or min(a, b) < 0 or a + b > limit:
            raise ValueError('budget exceeded: ' + k)


def finite(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('empty/nonfinite raw prediction array')
    return a


def metrics(y, p):
    y, p = finite(y), finite(p)
    if y.shape != p.shape or len(y) < 2:
        raise ValueError('prediction shape mismatch')
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        raise ValueError('constant target R2 undefined')
    error = y - p
    return dict(r2=float(1 - np.sum(error ** 2) / denom),
                rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))))


def check_metrics(actual, saved):
    for k in ('r2', 'rmse'):
        if not np.isclose(actual[k], saved[k], rtol=1e-12, atol=1e-12):
            raise ValueError('existing metric mismatch: ' + k)


def scaler_check(scaler, other=None):
    if scaler.get('fit') != 'train' or scaler.get('transform') != 'standard':
        raise ValueError('target scaler units / train-only mismatch')
    m, s = finite(scaler['mean']), finite(scaler['scale'])
    if m.shape != (1,) or s.shape != (1,) or s[0] <= 0:
        raise ValueError('target scaler shape mismatch')
    if other is not None and scaler != other:
        raise ValueError('branch target scaler mismatch')


def validate_split(unit):
    ids = []
    for role in ('train', 'validation', 'test'):
        a = unit[role + '_indices']
        if not a or any(type(i) is not int for i in a):
            raise ValueError('split index schema mismatch')
        ids.extend(a)
    if len(ids) != unit['sample_count'] or set(ids) != set(range(unit['sample_count'])):
        raise ValueError('split overlap/coverage mismatch')


def bind(values, unit, role, prediction_key, *, krr=False):
    """Bind by both original row and existing sample key, never array position."""
    if role not in ('validation', 'test') or unit.get('units') != 'original property units':
        raise ValueError('prediction units/role mismatch')
    scaler_check(unit['scaler'])
    rows = np.asarray(values['indices'])
    if rows.ndim != 1 or rows.dtype.kind not in 'iu':
        raise ValueError('original row index type mismatch')
    ids = rows.tolist(); expected = unit[role + '_indices']
    keys = np.asarray(values['sample_keys']).astype(str).reshape(-1).tolist()
    if (len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected)
            or len(keys) != len(ids) or len(set(keys)) != len(keys)):
        raise ValueError('duplicate/missing/extra row or sample key')
    lookup = {i: n for n, i in enumerate(ids)}
    order = [lookup[i] for i in expected]
    if any(keys[lookup[i]] != unit['identities'][str(i)]['sample_key'] for i in expected):
        raise ValueError('row/sample key identity conflict')
    if krr:
        for k, field in [('scaler_mean', 'mean'), ('scaler_scale', 'scale')]:
            if not np.array_equal(np.asarray(values[k]).reshape(-1), unit['scaler'][field]):
                raise ValueError('stored KRR scaler mismatch')
    if 'alpha' in values and float(values['alpha']) != unit['alpha']:
        raise ValueError('stored alpha mismatch')
    result = {}
    for name in ('y_true', prediction_key):
        a = finite(values[name])
        if len(a) != len(ids):
            raise ValueError('prediction array length mismatch')
        result[name] = a[order]
    return result


def blend(a, b, weight):
    if type(weight) not in (float, int) or weight not in WEIGHTS:
        raise ValueError('outside fixed weight grid')
    a, b = finite(a), finite(b)
    if a.shape != b.shape:
        raise ValueError('branch shape mismatch')
    return finite(weight * a + (1 - weight) * b)


def choose(validation):
    if len(validation) != 35 or {(r['task'], r['fold']) for r in validation} != UNITS:
        raise ValueError('exactly 35 validation units required')
    scores = []
    for w in WEIGHTS:
        units = [dict(task=r['task'], fold=r['fold'],
                      r2=metrics(r['y_true'], blend(r['p10k_pred'], r['krr_pred'], w))['r2'])
                 for r in validation]
        scores.append(dict(weight=w, mean_validation_r2=float(np.mean([r['r2'] for r in units])), units=units))
    winner = max(scores, key=lambda s: (s['mean_validation_r2'], s['weight']))
    return dict(policy=copy.deepcopy(POLICY), weight=winner['weight'], scores=scores)


class FrozenGate:
    def __init__(self):
        self.payload = None

    def freeze(self, payload):
        if self.payload is not None:
            raise ValueError('policy already frozen; no reselection')
        if (payload.get('plan'), payload.get('revision'), payload.get('schema')) != (PLAN, REVISION, SCHEMA):
            raise ValueError('frozen identity mismatch')
        if (not payload.get('frozen') or payload.get('policy') != POLICY
                or payload.get('test_processed_by_this_cycle') is not False):
            raise ValueError('frozen policy / TEST ordering mismatch')
        units = payload['units']
        if len(units) != 35 or {(u['task'], u['fold']) for u in units} != UNITS:
            raise ValueError('frozen unit coverage mismatch')
        for u in units:
            validate_split(u); scaler_check(u['scaler'])
            if (u['p10k']['config']['parent'] != str(PARENT) or u['alpha'] not in (.1, 1.)):
                raise ValueError('frozen parent/alpha mismatch')
        scores = payload['scores']
        if len(scores) != 5 or [s['weight'] for s in scores] != list(WEIGHTS):
            raise ValueError('fixed score grid mismatch')
        for s in scores:
            if len(s['units']) != 35 or {(r['task'], r['fold']) for r in s['units']} != UNITS:
                raise ValueError('validation score coverage mismatch')
            a = finite([r['r2'] for r in s['units']])
            if float(a.mean()) != s['mean_validation_r2']:
                raise ValueError('validation equal weighting mismatch')
        best = max(scores, key=lambda s: (s['mean_validation_r2'], s['weight']))
        if payload['weight'] != best['weight']:
            raise ValueError('validation winner/tie mismatch')
        self.payload = copy.deepcopy(payload)

    def require_test(self):
        if self.payload is None:
            raise ValueError('formal TEST before freeze')


def load_npz(path, *, gate=None, formal=False):
    if formal:
        if gate is None:
            raise ValueError('formal array access before freeze')
        gate.require_test()
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def path_check(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p.resolve())


def stage_counter(stage, counts, status='RUNNING', **extra):
    write(OUT / f'{stage}_consumption.json', dict(time=now(), status=status, counts=counts, **extra), replace=True)


def prepare():
    resource_guard(); start = time.monotonic()
    counts = dict(validation_units=0, validation_presentations=0)
    stage_counter('prepare', counts)
    old = read(R9 / 'selection_before_test.json')
    old_formal = read(R9 / 'final_review/seven_task_results.json')
    pts = read(PT / 'selection/final_frozen.json')
    meta = read(PARENT.parent / 'metadata.json')
    if (meta['plan'], meta['revision'], meta['schema'], meta['step'], meta['status']) != (
            PARENT_PLAN, 'r1', 'ua2d-ptscale-pretrain-r1', 10000, 'COMPLETE'):
        raise ValueError('P10K PT identity mismatch')
    if meta['deploy'] != str(PARENT) or not pts['frozen'] or pts['plan'] != PARENT_PLAN or pts['revision'] != 'r1':
        raise ValueError('P10K freeze identity mismatch')
    policy = old['policy']; spec = policy['feature_spec']
    if (not old['frozen'] or old_formal['policy'] != policy or policy['name'] != 'G05_MKRR50'
            or policy['alphas'] != [.1, 1., 10.] or spec['kind'] != 'MorganBinary'
            or spec['radius'] != 2 or spec['fpSize'] != 2048 or not spec['includeChirality']
            or not spec['useBondTypes'] or spec['countSimulation'] or not spec['includeRingMembership']
            or spec['kernel'] != 'Tanimoto similarity' or spec['g05_weight'] != .5 or spec['krr_weight'] != .5):
        raise ValueError('historical KRR policy mismatch')
    if len(old['units']) != 35 or {(u['task'], u['fold']) for u in old['units']} != UNITS:
        raise ValueError('historical KRR coverage mismatch')
    records = {(r['task'], r['fold']): r for r in old_formal['folds']}
    if len(old_formal['folds']) != 35 or set(records) != UNITS:
        raise ValueError('historical formal index coverage mismatch')
    cache = OUT / 'inputs'; cache.mkdir(exist_ok=False)
    units = []
    for old_u in old['units']:
        task, fold = old_u['task'], old_u['fold']
        selected = pts['units'][f'P10K/{task}/{fold}']
        cfg = selected['config']; ft = PT / f'finetune/P10K/{task}/fold{fold}'
        formal = PT / f'formal/P10K/{task}/fold{fold}'
        entry = read(SPLIT / f'{task}.json')['folds'][fold]
        unit = dict(task=task, fold=fold, seed=42 + fold, sample_count=old_u['sample_count'],
                    units='original property units', alpha=old_u['alpha'], scaler=old_u['scaler'],
                    identities=old['identities'][task],
                    **{r + '_indices': entry[r + '_indices'] for r in ('train', 'validation', 'test')})
        validate_split(unit); scaler_check(unit['scaler'], selected['scaler'])
        if set(unit['identities']) != {str(i) for i in range(unit['sample_count'])}:
            raise ValueError('original complete row/key coverage mismatch')
        for role in ('train', 'validation'):
            samples = old_u[role + '_samples']
            if [s['original_row'] for s in samples] != unit[role + '_indices']:
                raise ValueError('old sample row order mismatch')
            for s in samples:
                identity = unit['identities'][str(s['original_row'])]
                if any(s[k] != identity[k] for k in ('sample_key', 'normalized_smiles')):
                    raise ValueError('old sample identity/source mismatch')
        for role in ('train', 'validation', 'test'):
            if unit[role + '_indices'] != old_u[role + '_indices'] or unit[role + '_indices'] != pts['splits'][task]['folds'][fold][role + '_indices']:
                raise ValueError('original split mismatch')
        if selected['test_indices'] != unit['test_indices']:
            raise ValueError('frozen test indices mismatch')
        if (cfg['plan'], cfg['revision'], cfg['arm'], cfg['task'], cfg['fold'], cfg['seed'], cfg['parent_step']) != (
                PARENT_PLAN, 'r1', 'P10K', task, fold, 42 + fold, 10000):
            raise ValueError('P10K FT identity mismatch')
        if (cfg['parent'] != str(PARENT) or cfg['parent_schema'] != meta['schema']
                or cfg['parent_plan'] != PARENT_PLAN or cfg['source'] != str(SOURCE)
                or cfg['source'] != old_u['reference']['config']['source']
                or cfg['split'] != str(SPLIT / f'{task}.json')):
            raise ValueError('parent/source/split path mismatch')
        if cfg != read(ft / 'config.json') or cfg != read(formal / 'config.json'):
            raise ValueError('P10K config mismatch')
        if read(ft / 'scaler.json') != unit['scaler'] or read(formal / 'scaler.json') != unit['scaler']:
            raise ValueError('P10K scaler mismatch')
        m = read(ft / 'metrics.json')
        if (m['status'] != 'COMPLETE' or m['config'] != cfg or m['scaler'] != unit['scaler']
                or m['best_epoch'] != selected['best_epoch'] or m['best_stage'] != selected['best_stage']):
            raise ValueError('P10K validation BEST mismatch')
        valpath = ft / 'validation_predictions.npz'
        kv = Path(old_u['model']).parent / 'validation' / f"alpha_{unit['alpha']:g}" / 'predictions.npz'
        trial = read(kv.parent / 'metrics.json')
        if trial != old_u['chosen_validation'] or trial['alpha'] != unit['alpha']:
            raise ValueError('old frozen alpha/validation mismatch')
        rec = records[(task, fold)]
        if rec['alpha'] != unit['alpha'] or rec['validation'] != trial:
            raise ValueError('old formal alpha provenance mismatch')
        pv = load_npz(valpath); bv = load_npz(kv)
        p = bind(pv, unit, 'validation', 'y_pred')
        b = bind(bv, unit, 'validation', 'y_krr', krr=True)
        if not np.array_equal(p['y_true'], b['y_true']):
            raise ValueError('validation original-unit label mismatch')
        if int(pv['best_epoch']) != selected['best_epoch'] or str(pv['best_stage']) != selected['best_stage']:
            raise ValueError('validation prediction BEST metadata mismatch')
        check_metrics(metrics(p['y_true'], p['y_pred']), m)
        check_metrics(metrics(b['y_true'], b['y_krr']), trial['branch'])
        output = cache / f'{task}_fold{fold}.npz'
        np.savez(output, y_true=p['y_true'], p10k_pred=p['y_pred'], krr_pred=b['y_krr'],
                 indices=np.asarray(unit['validation_indices']),
                 sample_keys=[unit['identities'][str(i)]['sample_key'] for i in unit['validation_indices']],
                 alpha=unit['alpha'], scaler_mean=unit['scaler']['mean'], scaler_scale=unit['scaler']['scale'])
        unit.update(validation_cache=str(output),
                    p10k=dict(config=cfg, config_path=str(ft / 'config.json'), best=selected['best'],
                              best_epoch=selected['best_epoch'], best_stage=selected['best_stage'],
                              scaler_path=str(ft / 'scaler.json'), validation=str(valpath),
                              test=path_check(formal / 'test_predictions.npz')),
                    krr=dict(model=old_u['model'], validation=str(kv),
                             test=path_check(rec['paths']['branch']), feature_spec=spec,
                             alpha=unit['alpha'], chosen_validation=trial,
                             test_branch_metrics=rec['branch']), split=str(SPLIT / f'{task}.json'))
        # Models are provenance only; existing arrays are the computational dependencies.
        unit['krr']['model_exists'] = Path(old_u['model']).exists()
        unit['p10k']['best_exists'] = Path(selected['best']).exists()
        units.append(unit)
        counts['validation_units'] += 1; counts['validation_presentations'] += len(p['y_true'])
        stage_counter('prepare', counts)
        print('prepare', task, fold, counts, flush=True)
    if counts != dict(validation_units=35, validation_presentations=4714):
        raise ValueError('validation coverage budget mismatch')
    metadata = dict(plan=PLAN, revision=REVISION, schema=SCHEMA, status='COMPLETE',
                    units=units, counts=counts, dtype='float64', prediction_units='original property units',
                    source=str(SOURCE), split=str(SPLIT), parent=meta, policy=POLICY,
                    r9_freeze=str(R9 / 'selection_before_test.json'),
                    p10k_freeze=str(PT / 'selection/final_frozen.json'),
                    r10_freeze=str(R10 / 'selection_before_test.json'),
                    test_values_read=False, time=now(), seconds=time.monotonic() - start)
    write(cache / 'metadata.json', metadata)
    write(cache / 'COMPLETE', dict(schema=SCHEMA, plan=PLAN, revision=REVISION, counts=counts, time=now()))
    stage_counter('prepare', counts, 'COMPLETE', elapsed_seconds=time.monotonic() - start)


def select():
    resource_guard(); meta = read(OUT / 'inputs/metadata.json')
    if (read(OUT / 'inputs/COMPLETE')['schema'] != SCHEMA or meta['status'] != 'COMPLETE'
            or meta['plan'] != PLAN or meta['revision'] != REVISION or meta['schema'] != SCHEMA):
        raise ValueError('validation cache not COMPLETE / identity mismatch')
    # Reserve all 175 evaluations before selection; no candidate computed on TEST.
    stage_counter('select', dict(weight_candidates=5, validation_scores=175))
    validation = []
    for u in meta['units']:
        a = load_npz(u['validation_cache'])
        if float(a['alpha']) != u['alpha']:
            raise ValueError('cached alpha mismatch')
        for k in ('p10k_pred', 'krr_pred'):
            bind(a, u, 'validation', k, krr=True)
        validation.append(dict(task=u['task'], fold=u['fold'], **a))
    selected = choose(validation)
    write(OUT / 'selection/validation_selection.json', selected)
    stage_counter('select', dict(weight_candidates=5, validation_scores=175), 'COMPLETE')
    print('validation-selected global P10K weight', selected['weight'], flush=True)


def freeze():
    resource_guard(); meta = read(OUT / 'inputs/metadata.json')
    payload = dict(read(OUT / 'selection/validation_selection.json'), plan=PLAN, revision=REVISION,
                   schema=SCHEMA, frozen=True, frozen_at=now(), units=meta['units'],
                   parent=meta['parent'], source=meta['source'],
                   test_processed_by_this_cycle=False, existing_shared_test_development_exposed=True,
                   budget_at_freeze=read(OUT / 'budget.json'), provenance=meta)
    gate = FrozenGate(); gate.freeze(payload)
    write(OUT / 'selection/final_frozen.json', payload)
    stage_counter('freeze', dict(freezes=1), 'COMPLETE')


def formal():
    resource_guard(); start = time.monotonic()
    gate = FrozenGate(); gate.freeze(read(OUT / 'selection/final_frozen.json'))
    old = read(R9 / 'selection_before_test.json')
    old_units = {(u['task'], u['fold']): u for u in old['units']}
    old_records = {(r['task'], r['fold']): r for r in read(R9 / 'final_review/seven_task_results.json')['folds']}
    original_pts = read(PT / 'selection/final_frozen.json')
    counts = dict(test_starts=0, test_graphs=0, outputs=0)
    stage_counter('formal', counts)
    rows = []
    for u in gate.payload['units']:
        gate.require_test()
        key = (u['task'], u['fold']); prior = old_units[key]; rec = old_records[key]
        selected = original_pts['units'][f"P10K/{u['task']}/{u['fold']}"]
        if (prior['alpha'] != u['alpha'] or prior['model'] != u['krr']['model']
                or prior['scaler'] != u['scaler'] or rec['paths']['branch'] != u['krr']['test']
                or rec['alpha'] != u['alpha'] or prior['chosen_validation'] != u['krr']['chosen_validation']
                or selected['config'] != u['p10k']['config'] or selected['best'] != u['p10k']['best']
                or selected['scaler'] != u['scaler'] or old['identities'][u['task']] != u['identities']):
            raise ValueError('frozen original provenance changed')
        entry = read(u['split'])['folds'][u['fold']]
        if any(entry[r+'_indices'] != u[r+'_indices'] or prior[r+'_indices'] != u[r+'_indices']
               for r in ('train', 'validation', 'test')):
            raise ValueError('frozen split changed')
        folder = OUT / f"formal/{u['task']}/fold{u['fold']}"
        folder.mkdir(parents=True, exist_ok=False)
        counts['test_starts'] += 1; counts['test_graphs'] += len(u['test_indices'])
        stage_counter('formal', counts, current=dict(task=u['task'], fold=u['fold']))
        write(folder / 'consumption.json', dict(status='RUNNING', test_starts=1,
                                               test_graphs=len(u['test_indices']), start=now()))
        pvals = load_npz(u['p10k']['test'], gate=gate, formal=True)
        bvals = load_npz(u['krr']['test'], gate=gate, formal=True)
        p = bind(pvals, u, 'test', 'y_pred')
        b = bind(bvals, u, 'test', 'y_pred', krr=True)
        if not np.array_equal(p['y_true'], b['y_true']):
            raise ValueError('formal original-unit labels conflict')
        pm = read(Path(u['p10k']['test']).parent / 'metrics.json')
        if (pm['status'] != 'COMPLETE' or pm['config'] != u['p10k']['config']
                or pm['scaler'] != u['scaler'] or pm['best_epoch'] != u['p10k']['best_epoch']
                or pm['best_stage'] != u['p10k']['best_stage']
                or int(pvals['best_epoch']) != u['p10k']['best_epoch']
                or str(pvals['best_stage']) != u['p10k']['best_stage']):
            raise ValueError('formal P10K BEST/scaler/config conflict')
        ref = metrics(p['y_true'], p['y_pred'])
        check_metrics(ref, pm)
        check_metrics(metrics(b['y_true'], b['y_pred']), u['krr']['test_branch_metrics'])
        w = gate.payload['weight']
        prediction = blend(p['y_pred'], b['y_pred'], w)
        values = metrics(p['y_true'], prediction)
        np.savez(folder / 'predictions.npz', indices=np.asarray(u['test_indices']),
                 sample_keys=[u['identities'][str(i)]['sample_key'] for i in u['test_indices']],
                 y_true=p['y_true'], y_pred=prediction, y_p10k=p['y_pred'],
                 weight=np.float64(w), alpha=np.float64(u['alpha']),
                 scaler_mean=u['scaler']['mean'], scaler_scale=u['scaler']['scale'])
        row = dict(task=u['task'], fold=u['fold'], alpha=u['alpha'], weight=w,
                   metrics=values, reference=ref, delta_r2=values['r2'] - ref['r2'],
                   predictions=str(folder / 'predictions.npz'),
                   source=dict(p10k=u['p10k']['test'], krr=u['krr']['test']))
        write(folder / 'metrics.json', row); rows.append(row)
        write(folder / 'consumption.json', dict(status='COMPLETE', test_starts=1,
                                               test_graphs=len(u['test_indices']), end=now()), replace=True)
        counts['outputs'] += 1; stage_counter('formal', counts)
        print('formal', u['task'], u['fold'], 'R2', values['r2'], flush=True)
    if counts != dict(test_starts=35, test_graphs=5875, outputs=35):
        raise ValueError('formal coverage budget mismatch')
    write(OUT / 'formal/results.json', dict(rows=rows, weight=gate.payload['weight']))
    stage_counter('formal', counts, 'COMPLETE', elapsed_seconds=time.monotonic() - start)
