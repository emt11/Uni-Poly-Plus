"""Isolated UA2D-3DSHIFT campaign: prepare/PT evidence, three-arm FT matrix and frozen TEST.

All paths are explicit under ``results/ua2d_3dshift_20261022/r1``.  This module is
only imported by the new prepare/PT/controller entries and never by a legacy run.
"""
import copy
import datetime
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch import nn

from src.modules.glt_dual import Pure2DGLTModel, strict_module_state
from src.modules.ua2d_galformer_pretrain import (PretrainConfig, attention_record, package_attention,
                                               UNIFORM_OBJECTIVE_POLICY)
from src.modules.ua2d_3dshift import (SHIFT_SCHEMA, VARIANTS, PRETRAIN_UPDATES, architecture_record, check_shift_package,
                                       shift_record, restore_downstream_shift)
from src.training.glt_dual_runtime import require_tmux, save_checkpoint
from src.training.ua2d_fpdecay_sg import (Data, MeanModel, begin_epoch_reservation, close_epoch_reservation,
                                          conservative_ft_counts, finite, loader, metrics, optimizer,
                                          process_identity, seed_all, summarize_timing)
from src.training.ua2d_ptscale import ARCHITECTURE as REFERENCE_ARCHITECTURE, PROFILE as REFERENCE_PROFILE

ROOT = Path(__file__).resolve().parents[2]
PLAN = 'UA2D-3DSHIFT-20261022-01'
REVISION = 'r1.5-ALL1-PCTRL-GPU2'
ORIGINAL_OUT = ROOT / 'results/ua2d_3dshift_20261022/r1'
ORIGINAL_LOG = ROOT / 'logs/ua2d_3dshift_20261022/r1'
OUT = ORIGINAL_OUT / 'all1_pcontrol_gpu2'
LOG = ORIGINAL_LOG / 'all1_pcontrol_gpu2'
INPUTS = ORIGINAL_OUT / 'inputs'
SIDECAR = ORIGINAL_OUT / 'inputs/sidecar'
CACHE = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/prepared_cache'
PSCALE = ROOT / 'results/ua2d_ptscale_20261022/r1/inputs'
TRANSFER = ROOT / 'results/ua2d_transfer_20261022/r1/inputs'
INITIALIZATION = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/initialization'
BASE = ROOT / 'results/ua2d_fpdecay_20261022/sg_r2/pretrain/BASE/deploy.pt'
BASE_CONFIG = BASE.parent / 'config.json'
BASE_RESULTS = ROOT / 'results/ua2d_fpdecay_20261022/sg_r2/final_review/results.json'
P10K_RESULTS = ROOT / 'results/ua2d_ptscale_20261022/r1/final_review/results.json'
BFP_RESULTS = ROOT / 'results/ua2d_pretrain_fp_20261019/fast_r4/final_review/results.json'
DOWN = ROOT / 'data/processed/mcl_ph_r4_20260930/downstream_direct_01.staging'
SPLIT = ROOT / 'data/splits/mips_paper5_outer'
HEADS = ROOT / 'results/glt_optimization_20261011/r4/U1/initialization/heads_2d'
SOURCE = ROOT / 'data/processed/mcl_ph_r4_20260930/pi1m_direct_01.staging'
INDEX = ROOT / 'results/mcl_ph_next_20260929/p0/direct_indices_v1/summary.json'

TASKS = ('eea', 'egb', 'egc', 'ei', 'eps', 'nc', 'xc')
ARMS = ('F_G10K', 'P_G')
STEPS = {arm: PRETRAIN_UPDATES for arm in ARMS}
FINAL = 'P_G'
EXECUTION_ARMS = ARMS
EVAL_ARMS = (*ARMS,'P_G5K')
STEPS['P_G5K']=5000
FT_GPU = {'F_G10K':3,'P_G':2,'P_G5K':3}
ORIGINAL_WALLCLOCK_STARTED = 1792682501.581892
ORIGINAL_DEADLINE = ORIGINAL_WALLCLOCK_STARTED + 129600
PT_SCHEMA = SHIFT_SCHEMA
FT_SCHEMA = 'ua2d-3dshift-best-r1'
ARCHITECTURE = copy.deepcopy(REFERENCE_ARCHITECTURE)
PROFILE = copy.deepcopy(REFERENCE_PROFILE)
LIMITS = dict(prepare_starts=1, initialization_starts=0, pt_starts=4, pt_updates=30630,
              pt_presentations=30875040, ft_starts=105, epochs_started=7350,
              validation_started=7350, attempted_updates=168000, test_starts=105,
              test_graphs=17625, cpu_suites=1)
THRESHOLD = .010
SOTA = {'eea': ('Eea', 'Periodic-TDL', .934), 'egb': ('Egb', 'Periodic-TDL (+Chem, +RC)', .923),
        'egc': ('Egc', 'Periodic-TDL (+Chem, +RC)', .911), 'ei': ('Ei', 'Periodic-TDL', .839),
        'eps': ('EPS', 'Periodic-TDL', .787), 'nc': ('Nc', 'Periodic-TDL', .856),
        'xc': ('Xc', 'Periodic-TDL', .423)}


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def unit_path(arm, task, fold, formal=False):
    return OUT / ('formal' if formal else 'finetune') / arm / task / f'fold{fold}'


def variant_of(arm):
    return 'P_G' if arm=='P_G5K' else arm


def parent_path(arm):
    if arm=='P_G5K':return OUT/'pretrain/P_G/preserved/step05000.pt'
    return OUT / 'pretrain' / arm / 'deploy.pt'


def unit_config(arm, task, fold):
    if arm not in EVAL_ARMS or task not in TASKS or fold not in range(5):
        raise ValueError('outside the 3dshift three-arm matrix')
    return dict(plan=PLAN, revision=REVISION, arm=arm, task=task, fold=fold, seed=42 + fold,
                variant=variant_of(arm), local_alignment=VARIANTS[variant_of(arm)]['local'],
                parent=str(parent_path(arm)), parent_step=STEPS[arm], parent_plan=PLAN,
                parent_schema=RECOVERY_CHECKPOINT_SCHEMA if arm=='P_G5K' else PT_SCHEMA, parent_arm=variant_of(arm), parent_world_size=1,
                head_initial=str(HEADS / task / f'fold{fold}.pt'), source=str(DOWN),
                split=str(SPLIT / f'{task}.json'), cache=str(TRANSFER),physical_gpu=FT_GPU[arm],world_size=1,objective_policy='uniform1_align_mean_v1',
                attention=dict(revision='ua2d_attention_scale_r1', scale_denominator=512,
                               scale=512 ** -.5, bias_scaled=False),
                architecture=ARCHITECTURE, profile=PROFILE,
                input_semantics='pure_2d_from_periodic_3d_relative_shift')


def update_plan(status, phase, detail):
    p = ROOT / 'CURRENT_PLAN.md'
    text = p.read_text()
    start = '\n### UA2D-3DSHIFT r1 最新执行状态\n'
    end = '\n<!-- UA2D_3DSHIFT_EXECUTION_END -->'
    entry = (start + f'\n- {now()}；状态：{status}；阶段：{phase}。\n- {detail}\n'
             f'- 交付：{OUT}；日志：{LOG}；执行自检不构成 Codex 独立审查。\n' + end)
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + entry + after
    else:
        text = text.rstrip() + '\n' + entry + '\n'
    p.write_text(text)
    write(OUT / 'status.json', dict(plan=PLAN, revision=REVISION, status=status, phase=phase,
                                    time=now(), detail=detail, independent_review='PENDING'))


def pt_values():
    """Retain BASE data/model settings; explicitly select the new all-one losses."""
    reference = read(BASE_CONFIG)
    values = dict(reference['config'])
    values.update(attention_scale_denominator=512, objective_policy=UNIFORM_OBJECTIVE_POLICY,
                  weight_warmup=0, mask_2d_weight=1., mask_3d_weight=1.,
                  contrast_weight=1., fp_weight=1., coordinate_weight=1.,
                  length_weight=1., angle_weight=1.)
    values = dict(sorted(values.items()))
    config = PretrainConfig(**values)
    if (config.strategy != 'mask_align' or config.fusion != 'none' or config.fp_enabled is not True
            or config.attention_scale_denominator != 512 or config.mask_ratio != .4
            or config.noise_sigma != .03 or config.dropout != .1):
        raise ValueError('BASE recipe changed')
    return values


def pt_attention():
    return attention_record(PretrainConfig(**pt_values()))


def pt_learning_rate(update):
    from src.training.ua2d_fpdecay_sg import pt_learning_rate as base_learning_rate
    if type(update) is not int or not 0 <= update < PRETRAIN_UPDATES:
        raise ValueError('outside fixed10k PT')
    return base_learning_rate(update % 5000)


def speed_reference():
    """Existing BASE steps501--5000, full-precision median, no new measurement."""
    records=[json.loads(line) for line in (BASE.parent/'step_timing.jsonl').read_text().splitlines() if line]
    values=[r['step_seconds'] for r in records if 501<=r['step']<=5000]
    if len(values)!=4500:raise ValueError('BASE speed reference incomplete')
    finite(values);return float(np.median(values))


def shift_objective(result, weights, denominators):
    """Six-microbatch window normalization on one GPU (no world/accumulation multiplier)."""
    total = None
    for key, weight in weights.items():
        value = result['sums'][key] / denominators[key].clamp_min(1.)
        term = weight * value
        total = term if total is None else total + term
    if total is None:
        raise ValueError('empty objective set')
    return total


def cache_references(require_sidecar=True):
    """Read-only identity of every input the new campaign depends on."""
    cache = read(CACHE / 'metadata.json')
    if (cache.get('schema') != 'ua2d_prepared_templates_r4' or cache.get('status') != 'COMPLETE'
            or cache.get('train_rows') != 911391 or cache.get('source') != str(SOURCE)
            or cache.get('index') != str(INDEX)):
        raise ValueError('prepared template cache identity mismatch')
    mapping = read(PSCALE / 'metadata.json')
    if (mapping.get('plan') != 'UA2D-PTSCALE-20261022-01' or mapping.get('schema') != 'ua2d-ptscale-logical-map-r1'
            or mapping.get('status') != 'COMPLETE' or mapping.get('records') != 910756
            or mapping.get('excluded') != 635 or mapping.get('parent_cache') != str(CACHE)):
        raise ValueError('PTSCALE logical mapping identity mismatch')
    transfer = read(TRANSFER / 'metadata.json')
    if (transfer.get('schema') != 'ua2d-transfer-input-2hop-r1' or transfer.get('status') != 'COMPLETE'
            or transfer.get('records') != 5875 or transfer.get('source') != str(DOWN)
            or transfer.get('split') != str(SPLIT) or transfer.get('max_hop') != 2
            or transfer.get('online_fallback') is not False):
        raise ValueError('downstream 2-hop cache identity mismatch')
    sidecar = None
    if require_sidecar:
        from src.dataset.ua2d_3dshift import sidecar_metadata
        sidecar = sidecar_metadata(SIDECAR)
    base = torch.load(BASE, map_location='cpu', weights_only=False)
    if (base.get('schema') != 'ua2d-fpdecay-sg-pretrain-r2' or base.get('step') != 5000
            or base.get('config', {}).get('strategy') != 'mask_align'
            or base['config'].get('fp_enabled') is not True
            or package_attention(base)['scale_denominator'] != 512):
        raise ValueError('frozen BASE F_G checkpoint mismatch')
    execution = base.get('execution', {})
    if (execution.get('world_size') != 1 or execution.get('microbatch') != 168
            or execution.get('accumulation') != 6 or execution.get('global_batch') != 1008
            or execution.get('eligible_rows') != 910756):
        raise ValueError('frozen BASE F_G execution mismatch')
    if not BASE_RESULTS.exists() or not P10K_RESULTS.exists() or not BFP_RESULTS.exists():
        raise ValueError('a read-only reference result is missing')
    for path in (INITIALIZATION / 'common_step0.pt', INITIALIZATION / 'task_heads' / 'V1.pt'):
        if not path.exists():
            raise ValueError('frozen common initialisation is missing: ' + str(path))
    return dict(cache=cache, mapping=mapping, transfer=transfer, sidecar=sidecar,sidecar_required=require_sidecar,
                base=dict(path=str(BASE), schema=base['schema'], step=base['step'],
                          arm=base.get('arm'), consumption=base.get('consumption')),
                references=dict(BASE=str(BASE_RESULTS), P10K=str(P10K_RESULTS), B_FP=str(BFP_RESULTS)),
                initialization=str(INITIALIZATION))


def validate_parent(package, arm):
    if arm=='P_G5K':return validate_pinned_5k_parent(package)
    config = check_shift_package(package, 'PT parent ' + arm)
    if package['variant'] != arm:
        raise ValueError('PT parent variant/arm mismatch')
    expected = asdict_config(PretrainConfig(**pt_values()))
    if package['config'] != expected:
        raise ValueError('PT parent objective config mismatch')
    execution = package.get('execution', {})
    if (execution.get('plan') != PLAN or execution.get('revision') != REVISION
            or execution.get('arm') != arm or execution.get('world_size') != 1
            or execution.get('microbatch') != 168 or execution.get('accumulation') != 6
            or execution.get('global_batch') != 1008 or execution.get('total_updates') != STEPS[arm]
            or execution.get('eligible_rows') != 910756 or execution.get('physical_gpus')!=[2]
            or execution.get('objective_policy')!='uniform1_align_mean_v1'
            or execution.get('loss_weight_warmup_updates')!=0 or execution.get('align_multiplier')!=1.):
        raise ValueError('PT parent execution metadata mismatch')
    if package_attention(package)['scale_denominator'] != 512:
        raise ValueError('PT parent attention scale mismatch')
    return config


def asdict_config(config):
    from dataclasses import asdict
    return asdict(config)


def initialize(parent, head, arm):
    validate_parent(parent, arm)
    model = MeanModel()
    parent_state=parent['model'] if arm=='P_G5K' else parent['state_dict']
    for name in ('o8', 'norm2'):
        prefix = name + '.'
        subset = {k[len(prefix):]: v for k, v in parent_state.items() if k.startswith(prefix)}
        strict_module_state(getattr(model, name), subset)
    strict_module_state(model.head, head)
    return model


def restore_best(package, cfg):
    if (package.get('schema') != FT_SCHEMA or package.get('config') != cfg
            or package.get('architecture') != ARCHITECTURE
            or package.get('attention') != cfg['attention']):
        raise ValueError('frozen FT best/source mismatch')
    model = MeanModel()
    strict_module_state(model, package['state_dict'])
    return model


def unit(arm, task, fold, formal=False):
    from src.utils import TargetScaler, scale_targets
    from sklearn.preprocessing import StandardScaler
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(FT_GPU[arm]):
        raise ValueError('FT/TEST physical GPU differs from explicit arm allocation')
    if os.environ.get('WORLD_SIZE')!='1':raise ValueError('single process/world1 required')
    folder = unit_path(arm, task, fold, formal)
    folder.mkdir(parents=True, exist_ok=False)
    cfg = unit_config(arm, task, fold)
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
        state['engineering_revision'] = 'ua2d-3dshift-epoch-reservation-r1'
        timed_write(folder / 'consumption.json', state)

    record()
    started = time.monotonic()
    ds = None
    try:
        seed_all(42 + fold)
        if formal:
            frozen = read(frozen_path(arm))
            if frozen.get('frozen') is not True:
                raise ValueError('TEST before the one-time freeze')
            if frozen['splits'][task] != read(SPLIT / f'{task}.json'):
                raise ValueError('frozen original split changed')
            selected = frozen['units'][f'{arm}/{task}/{fold}']
            saved = torch.load(selected['best'], map_location='cpu', weights_only=False)
            if (selected['config'] != cfg or selected['best'] != str(unit_path(arm, task, fold) / 'best.pt')
                    or selected['parent'] != str(parent_path(arm)) or selected['parent_step']!=STEPS[arm]
                    or selected['best_epoch'] != saved['best_epoch']
                    or selected['best_stage'] != saved['best_stage'] or selected['scaler'] != saved['scaler']
                    or selected['attention'] != cfg['attention']):
                raise ValueError('frozen complete best/scaler mismatch')
            model = restore_best(saved, cfg)
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
        ds = Data(task, fold, formal)
        if formal:
            if selected['test_indices'] != ds.indices:
                raise ValueError('frozen TEST indices mismatch')
            ds.set_target_override(scaler.transform(ds.raw_targets).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=ds.train, transform_mode='standard')
            scaler_json = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(),
                               fit='train', transform='standard')
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
            best_r2 = -math.inf
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
                    if met['r2'] > best_r2:
                        best_r2 = met['r2']
                        best_epoch = state['completed_epochs']
                        best_stage = stage
                        truth, prediction, result = y.copy(), p.copy(), met.copy()
                        timed_write(folder / 'best.pt', dict(schema=FT_SCHEMA, architecture=ARCHITECTURE,
                                                             attention=cfg['attention'], config=cfg,
                                                             state_dict={k: v.detach().cpu().clone()
                                                                         for k, v in model.state_dict().items()},
                                                             best_epoch=best_epoch, best_stage=best_stage,
                                                             scaler=scaler_json, outer_test='NOT_RUN'),
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
                                                   best_r2=best_r2, scaler=scaler_json,
                                                   consumption=copy.deepcopy(state), automatic_resume=False),
                        checkpoint=True)
        np.savez(folder / ('test_predictions.npz' if formal else 'validation_predictions.npz'),
                 y_true=finite(truth), y_pred=finite(prediction),
                 indices=np.array([ds.indices[i] for i in ds.validation]),
                 sample_keys=ds.keys[ds.validation], best_epoch=best_epoch, best_stage=best_stage)
        write(folder / 'metrics.json', dict(status='COMPLETE', config=cfg, best_epoch=best_epoch,
                                            best_stage=best_stage, scaler=scaler_json,
                                            attention=cfg['attention'], **result))
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
    if arm not in EVAL_ARMS:
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


def frozen_path(arm):
    separate=OUT/'selection'/f'frozen_{arm}.json'
    return separate if separate.is_file() else OUT/'selection/final_frozen.json'


def freeze_final(validation, arms=None):
    arms=ARMS if arms is None else tuple(arms)
    if arms not in (ARMS,EVAL_ARMS,('F_G10K',),('P_G',),('P_G5K',)):raise ValueError('unregistered freeze subset')
    path = OUT / 'selection' / ('all_evaluations_frozen.json' if arms==EVAL_ARMS else 'final_frozen.json' if arms==ARMS else f'frozen_{arms[0]}.json')
    if path.exists():
        raise ValueError('final freeze already exists')
    units, scores, splits, models = {}, {}, {}, {}
    for arm in arms:
        if not parent_path(arm).is_file():raise FileNotFoundError('missing frozen PT parent '+str(parent_path(arm)))
        result = validation[arm] if arm in validation else aggregate(arm)
        scores[arm] = dict(tasks={t: dict(mean=result['tasks'][t]['mean'], std=result['tasks'][t]['std'])
                                  for t in TASKS}, macro=result['macro'])
        models[arm] = str(parent_path(arm))
        for task in TASKS:
            splits[task] = read(SPLIT / f'{task}.json')
            for fold in range(5):
                key = f'{arm}/{task}/{fold}'
                folder = unit_path(arm, task, fold)
                m = read(folder / 'metrics.json')
                consumption=read(folder/'consumption.json')
                expected_config=unit_config(arm,task,fold)
                if (m.get('status')!='COMPLETE' or consumption.get('status')!='COMPLETE'
                    or consumption.get('completed_epochs')!=70 or m.get('config')!=expected_config
                    or read(folder/'config.json')!=expected_config or read(folder/'scaler.json')!=m['scaler']
                    or not (folder/'best.pt').is_file()):raise ValueError('incomplete FT evidence at freeze')
                entry = dict(config=unit_config(arm, task, fold), best=str(folder / 'best.pt'),
                             best_epoch=m['best_epoch'], best_stage=m['best_stage'], scaler=m['scaler'],
                             attention=m['attention'], validation_r2=float(m['r2']),
                             test_indices=sorted(read(SPLIT / f'{task}.json')['folds'][fold]['test_indices']),
                             variant=variant_of(arm),parent=str(parent_path(arm)),parent_step=STEPS[arm],split_path=str(SPLIT/f'{task}.json'),evaluation='five equal folds std(ddof1); seven equal task means; raw precision',scaler_path=str(folder/'scaler.json'),config_path=str(folder/'config.json'))
                if key in units:
                    raise ValueError('duplicate frozen unit')
                units[key] = entry
    if len(units) != len(arms)*35:
        raise ValueError('incomplete registered freeze')
    payload = dict(plan=PLAN, revision=REVISION, time=now(), frozen=True, FINAL=FINAL,
                   models=models, units=units, validation_scores=scores, splits=splits,
                   selection='immutable per-arm freeze before first TEST; final union references unchanged prior freeze; no reselection',freeze_arms=list(arms),
                   shift={a:shift_record(variant_of(a)) for a in arms},
                   architecture={a:architecture_record(variant_of(a)) for a in arms},
                   references=reference_scores()['paths'])
    if arms in (ARMS,EVAL_ARMS):
        for arm in arms:
            prior=OUT/'selection'/f'frozen_{arm}.json'
            if prior.exists():
                original=read(prior)
                if any(units[k]!=v for k,v in original['units'].items()):raise ValueError('prior formal freeze modified')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return payload


def reference_scores():
    base=read(BASE_RESULTS);p10k=read(P10K_RESULTS);bfp=read(BFP_RESULTS)
    def block(root,section,arm):
        value=root[section][arm]
        if set(value['tasks'])!=set(TASKS) or any(len(value['tasks'][t]['folds'])!=5 for t in TASKS):
            raise ValueError('incomplete seven-task five-fold reference')
        finite([value['macro'],*[x for t in TASKS for x in value['tasks'][t]['folds']]])
        return value
    validation=block(base,'validation','BASE')
    formal={'F_G':block(base,'formal','BASE'),'P10K':block(p10k,'formal','P10K'),'B_FP':block(bfp,'test','B_FP')}
    return dict(validation=dict(F_G=validation['macro'],tasks={t:validation['tasks'][t]['mean'] for t in TASKS}),
        validation_blocks=dict(F_G=validation),formal_blocks=formal,
        test={a:v['macro'] for a,v in formal.items()},tasks={a:{t:v['tasks'][t]['mean'] for t in TASKS} for a,v in formal.items()},
        paths=dict(F_G=str(BASE_RESULTS),P10K=str(P10K_RESULTS),B_FP=str(BFP_RESULTS)))


def comparisons(validation,formal):
    reference=reference_scores();v=dict(reference['validation_blocks'],**validation);f=dict(reference['formal_blocks'],**formal)
    specs={'P_G_minus_F_G10K':{'P_G':1,'F_G10K':-1},'P_G_minus_F_G':{'P_G':1,'F_G':-1},
           'P_G_minus_P10K':{'P_G':1,'P10K':-1},'P_G_minus_B_FP':{'P_G':1,'B_FP':-1}}
    def contrast(values,coefficients):
        if any(a not in values for a in coefficients):return dict(status='未完成／不可比较',delta=None,tasks={})
        tasks={}
        for t in TASKS:
            x=sum(c*finite(values[a]['tasks'][t]['folds']) for a,c in coefficients.items())
            tasks[t]=dict(folds=x.tolist(),mean=float(x.mean()),std=float(x.std(ddof=1)))
        return dict(status='COMPLETE',delta=float(sum(c*values[a]['macro'] for a,c in coefficients.items())),tasks=tasks)
    result={name:dict(coefficients=coef,validation=contrast(v,coef),test=contrast(f,coef),
                     interpretation='controlled 10k contrast' if name=='P_G_minus_F_G10K' else
                                    'historical numerical reference; training budget/protocol confounded')
            for name,coef in specs.items()}
    return result,None  # No matched F_G at 10k: the interaction is not identifiable.


RECOVERY_CHECKPOINT_SCHEMA='ua2d-3dshift-all1-pcontrol-intermediate-r1'
CHECKPOINT_INTERVAL=1000
CHECKPOINT_KEEP=2


def atomic_recovery_checkpoint(path,payload):
    """Exclusive new checkpoint -> flush/fsync -> atomic rename -> directory fsync."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('checkpoint overwrite prohibited: '+str(path))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def validate_recovery_checkpoint(package):
    step=package.get('completed_updates');arm=package.get('arm')
    if (package.get('schema')!=RECOVERY_CHECKPOINT_SCHEMA or package.get('plan')!=PLAN
        or package.get('revision')!=REVISION or arm not in ARMS or type(step) is not int
        or step not in range(CHECKPOINT_INTERVAL,PRETRAIN_UPDATES+1,CHECKPOINT_INTERVAL)
        or package.get('attempted_updates')!=step or package.get('next_global_position')!=step*1008
        or package.get('loss_warmup_position')!=step or package.get('lr_position')!=step
        or package.get('next_micro_index')!=step*6 or package.get('lr_cycle_position')!=step%5000
        or package.get('consumed_presentations')!=step*1008):raise ValueError('recovery checkpoint counter/position/schema conflict')
    for key in ('python','numpy','torch','cuda'):
        if key not in package.get('rng',{}):raise ValueError('incomplete recovery RNG')
    for key in ('model','optimizer','loader_state','execution','config'):
        if key not in package:raise ValueError('incomplete recovery checkpoint: '+key)
    if package['loader_state'].get('next_consumed_micro')!=step*6:raise ValueError('loader consumption cursor mismatch')
    return step


def save_recovery_checkpoint(output,model,optimizer,loader,consumption,metadata):
    """Preserve last2 complete snapshots; no retry/resume is dispatched here."""
    from src.training.glt_dual_runtime import rng_state
    output=Path(output);step=consumption['completed_updates'];tick=time.monotonic()
    folder=output/'checkpoints';path=folder/f'intermediate_{step:05d}.pt'
    package=dict(schema=RECOVERY_CHECKPOINT_SCHEMA,plan=PLAN,revision=REVISION,arm=metadata['arm'],
        step=step,completed_updates=step,attempted_updates=consumption['attempted_updates'],
        consumed_presentations=consumption['consumed_presentations'],model=model.state_dict(),optimizer=optimizer.state_dict(),
        rng=rng_state(),config=metadata['config'],execution=metadata,consumption=copy.deepcopy(consumption),
        lr_position=step,lr_cycle_position=step%5000,loss_warmup_position=step,
        next_micro_index=step*6,next_global_position=step*1008,
        loader_state=dict(generator_rng=loader.generator.get_state(),next_consumed_micro=step*6,
            ordered=True,num_workers=4,prefetch_factor=2,sampler_seed=42,
            next_global_position=step*1008,prefetched_unconsumed=consumption['prefetched_not_consumed'],
            replay_rule='discard prefetched queue; regenerate deterministic samples at next consumed cursor'),
        automatic_resume=False,restore_requires_new_explicit_authorization=True)
    validate_recovery_checkpoint(package);atomic_recovery_checkpoint(path,package)
    snapshots=sorted(folder.glob('intermediate_[0-9][0-9][0-9][0-9][0-9].pt'))
    kept=snapshots[-CHECKPOINT_KEEP:];removed=snapshots[:-CHECKPOINT_KEEP]
    # These are only new all1_pcontrol_gpu2 intermediate snapshots, never deployment/freeze sources.
    for previous in removed:
        if previous.parent!=folder or previous==path:raise ValueError('unsafe retention target')
        previous.unlink()
    descriptor=os.open(str(folder),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    event=dict(time=now(),step=step,checkpoint=str(path),retained=[str(p) for p in kept],
        removed=[str(p) for p in removed],seconds=time.monotonic()-tick,
        retention_basis='explicit recovery authorization: latest2 new intermediate files only; no formal/old artifact reads these',
        checkpoint_purpose='future explicitly authorized recovery only; no automatic resume')
    write(output/'checkpoint_manifest.json',event)
    with (output/'checkpoint_retention.jsonl').open('a') as handle:
        handle.write(json.dumps(event,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
    return event


def reused_cache_evidence():
    references=cache_references(require_sidecar=True)
    coverage=read(SIDECAR/'coverage.json')
    if coverage.get('records')!=910756 or coverage.get('unique_keys')!=910756 or coverage.get('complete') is not True:
        raise ValueError('original sidecar coverage conflict')
    exit_path=ORIGINAL_LOG/'3dshift_prepare/exit.txt'
    if int(exit_path.read_text().strip())!=0:raise ValueError('original preparation lacks normal exit0')
    return dict(references=references,coverage=coverage,original_prepare_exit=0,
        original_prepare_exit_path=str(exit_path),prepare_starts_added=0,read_only_reuse=True)


def validate_pinned_5k_parent(package):
    # This is the genuine intermediate schema/step, never a fabricated10k deployment.
    if (package.get('schema')!=RECOVERY_CHECKPOINT_SCHEMA or package.get('plan')!=PLAN
        or package.get('revision')!=REVISION or package.get('arm')!='P_G'
        or package.get('completed_updates')!=5000 or package.get('step')!=5000):
        raise ValueError('pinned P_G5K source identity mismatch')
    validate_recovery_checkpoint(package)
    expected=pt_values()
    if package.get('config')!=expected:raise ValueError('P_G5K objective/model config mismatch')
    e=package['execution']
    if (e.get('arm')!='P_G' or e.get('world_size')!=1 or e.get('physical_gpus')!=[2]
        or e.get('microbatch')!=168 or e.get('accumulation')!=6 or e.get('global_batch')!=1008
        or e.get('total_updates')!=10000 or e.get('objective_policy')!='uniform1_align_mean_v1'
        or e.get('loss_weight_warmup_updates')!=0 or e.get('align_multiplier')!=1.):
        raise ValueError('P_G5K original PT provenance mismatch')
    if e.get('architecture')!=architecture_record('P_G') or e.get('shift')!=shift_record('P_G'):
        raise ValueError('P_G5K P geometry identity mismatch')
    return PretrainConfig(**expected)
