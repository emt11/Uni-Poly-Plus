"""G1 single-process, single-GPU FT or one frozen formal prediction."""
import argparse
import math
import os
import signal
import time
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from src.training.ua2d_fp_gate import *
from src.training.glt_dual_runtime import require_tmux, open_source, save_checkpoint
from src.modules.glt_dual import strict_module_state
from src.modules.glt_adaptation import AdaptationAdapter
from src.utils import set_global_seed, scale_targets, TargetScaler, evaluate
import scripts.finetune_mcl_ph as old


def run(arm, task, fold, formal):
    require_tmux()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise ValueError('only physical GPU2/3 authorized')
    binding = read(OUT / 'initialization/bindings' / task / f'fold{fold}.json')
    entry = binding['manifest']['folds'][fold]
    source_unit = binding['unit']
    frozen_unit = None
    if formal:
        payload = require_freeze()
        frozen_unit = next(u for u in payload['units'] if (u['arm'], u['task'], u['fold']) == (arm, task, fold))
    folder = unit_path(arm, task, fold, formal)
    folder.mkdir(parents=True, exist_ok=False)
    source = cache = None
    tracker = None
    stop_requested = [False]
    def request_stop(signum, frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    def should_stop():
        return stop_requested[0]
    step_state = dict(sequence=0, inflight=False, boundary='initializing')
    history = []
    updates = dict(attempted=0, completed=0)
    nsteps = math.ceil(len(entry['train_indices']) / 24)
    update_limit = 70 * nsteps
    try:
        set_global_seed(42 + fold)
        package = torch.load(PARENT, map_location='cpu', weights_only=False)
        head = torch.load(head_path(task, fold), map_location='cpu', weights_only=False)
        initial = torch.load(OUT / 'initialization/gate_initial.pt', map_location='cpu', weights_only=False)
        model = build_model(package, head, arm, initial)
        del package
        write(folder / 'parameters.json', dict(total=sum(p.numel() for p in model.parameters()),
            head=sum(p.numel() for p in model.head.parameters()),
            gate=sum(p.numel() for p in model.head.fusion.parameters()),
            module_initial=str(OUT / 'initialization/gate_initial.pt')))
        if hasattr(model, 'feature_cache_context'):
            raise ValueError('head feature cache must be absent')
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA context created before controlled model initialization')
        gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
        write(folder / 'cuda_request.json', dict(pid=os.getpid(), gpu=gpu, time=now()))
        deadline = time.monotonic() + 180
        def await_controller(name):
            while not (folder / name).exists():
                if should_stop() or time.monotonic() >= deadline:
                    raise InterruptedError('CUDA identity handoff stopped or timed out')
                time.sleep(.1)
        await_controller('cuda_grant.json')
        model.to('cuda:0')
        torch.cuda.synchronize()
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        write(folder / 'cuda_ready.json', dict(pid=os.getpid(), gpu=gpu,
            start_ticks=int(stat[19]), time=now(), operation='required model.to(cuda) completed'))
        await_controller('cuda_mapped.json')
        old.configure_periodic_tdl_dropout(model, task, profile='uniform_u1')
        if formal:
            best = torch.load(frozen_unit['best'], map_location='cpu', weights_only=False)
            for k in ('arm', 'task', 'fold', 'seed', 'parent', 'head_initial', 'scaler', 'best_epoch', 'policy', 'optimizer_groups'):
                if best[k] != frozen_unit[k]:
                    raise ValueError('formal checkpoint/freeze mismatch ' + k)
            if best['optimizer_updates'] != update_limit or best['module_initial'] != frozen_unit['module_initial']:
                raise ValueError('checkpoint update/initial mismatch')
            strict_module_state(model, best['state_dict'])
            selected = sorted(entry['test_indices'])
            local_val = list(range(len(selected)))
            scaler_record = frozen_unit['scaler']
        else:
            selected, local_train, local_val = old.compact_train_validation_indices(entry['train_indices'], entry['validation_indices'])
        source, frame = open_source(DOWN, DOWN, task=task, selected_indices=selected,
            split_manifest=binding['manifest'], verify_key_digest=False)
        if frame['original_row'].astype(int).tolist() != selected:
            raise ValueError('selected original rows changed')
        identities = binding['identities']
        for i, idx in enumerate(selected):
            if source.samples[i][0].hex() != identities[str(idx)]['sample_key'] or frame.iloc[i]['normalized_smiles'] != identities[str(idx)]['normalized_smiles']:
                raise ValueError('source sample identity mismatch')
        cache = FPReader(OUT / 'fingerprints.sqlite')
        tracker = InputTracker(folder, 'test' if formal else 'train_validation')
        ds = make_dataset(source, frame, selected, identities, cache, tracker=tracker, should_stop=should_stop)
        set_global_seed(42 + fold)
        if formal:
            scaler = TargetScaler(task, StandardScaler(), transform_mode='standard')
            scaler.scaler.mean_ = np.asarray(scaler_record['mean'])
            scaler.scaler.scale_ = np.asarray(scaler_record['scale'])
            scaler.scaler.var_ = scaler.scaler.scale_ ** 2
            scaler.scaler.n_features_in_ = 1
            ds.set_target_override(scaler.transform(frame['label'].to_numpy(dtype=np.float64)).reshape(-1))
        else:
            scaler = scale_targets(ds, task, train_indices=local_train, transform_mode='standard')
            scaler_record = dict(mean=scaler.scaler.mean_.tolist(), scale=scaler.scaler.scale_.tolist(), fit='train', transform='standard')
            if scaler_record != source_unit['scaler'] or scaler_record != binding['ua']['scaler']:
                raise ValueError('original train-only scaler mismatch')
        validation = DataLoader(Subset(ds, local_val), batch_size=24, shuffle=False, num_workers=0, collate_fn=fp_collate)
        if formal:
            _, _, truth, prediction = evaluate(AdaptationAdapter(model), validation, nn.MSELoss(), 'cuda:0', scaler=scaler, amp_dtype='fp32')
            epoch = best['best_epoch']
            groups = best['optimizer_groups']
        else:
            train = DataLoader(Subset(ds, local_train), batch_size=24, shuffle=True, num_workers=0,
                collate_fn=fp_collate, generator=torch.Generator().manual_seed(42 + fold))
            gradients = {}
            groups = {}
            def observe(kind):
                stage = 'head' if updates['attempted'] < 10 * nsteps else 'joint'
                step = updates['attempted'] if stage == 'head' else updates['attempted'] - 10 * nsteps
                if kind == 'attempted':
                    if should_stop():
                        raise InterruptedError('controlled stop before optimizer update')
                    if step_state['inflight']:
                        raise RuntimeError('overlapping optimizer update')
                    step_state.update(sequence=updates['attempted'] + 1, inflight=True, boundary='before_update')
                    if step == 0:
                        named = dict(model.named_parameters())
                        write(folder / ('trainability_' + stage + '.json'),
                            dict(stage=stage, trainable_names=[n for n, p in named.items() if p.requires_grad],
                                 trainable_parameters=sum(p.numel() for p in named.values() if p.requires_grad),
                                 optimizer_evidence='full group evidence published at epoch completion'))
                if kind == 'attempted' and step in (0, 1):
                    fusion = model.head.fusion
                    for name, module in [('U', fusion.u), ('V', fusion.v), ('gate', fusion.gate)]:
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):
                            raise RuntimeError('GATE gradient path missing/nonfinite ' + name)
                    encoder_has = any(p.grad is not None for module in (model.o8, model.norm2) for p in module.parameters())
                    if encoder_has != (stage == 'joint'):
                        raise RuntimeError('encoder head/joint gradient path mismatch')
                    values = {name: any(bool(torch.count_nonzero(p.grad)) for p in module.parameters())
                              for name, module in [('U', fusion.u), ('V', fusion.v), ('gate', fusion.gate)]}
                    if not values['V'] or (stage == 'head' and step == 1 and not all(values.values())):
                        raise RuntimeError('GATE expected live gradient missing')
                    if stage == 'head' and (model.o8.training or model.norm2.training or not model.head.training):
                        raise RuntimeError('frozen encoder eval/head train mismatch')
                    gradients[f'{stage}_{step}'] = dict(nonzero=values, encoder_gradient=encoder_has)
                    write(folder / 'gradient_paths.json', gradients, True)
                if updates[kind] >= update_limit:
                    raise RuntimeError('FT unit budget exhausted')
                updates[kind] += 1
                if kind == 'completed':
                    step_state.update(inflight=False, boundary='after_update')
                write(folder / 'updates.json', dict(updates, **step_state), True)
                if kind == 'completed' and should_stop():
                    raise InterruptedError('controlled stop after optimizer update')
            def progress(record):
                if record['stage'] not in groups:
                    groups[record['stage']] = stage_groups(model, record['stage'], record['optimizer_groups'])
                    write(folder / 'parameter_groups.json', groups, True)
                history.append(record)
                write(folder / 'progress.json', dict(history=history, updates=updates, outer_test='NOT_RUN'), True)
                print(json.dumps(record), flush=True)
            result = old.run_periodic_tdl_epochs(model, train, validation, nn.MSELoss(), 'cuda:0',
                task=task, scaler=scaler, progress=progress, profile='uniform_u1', step_observer=observe)
            if len(history) != 70 or updates != dict(attempted=update_limit, completed=update_limit):
                raise ValueError('FT budget completion mismatch')
            truth, prediction = result['predictions']
            epoch = result['best_epoch']
            save_checkpoint(folder / 'best.pt', dict(state_dict=result['state'], policy=POLICY,
                arm=arm, task=task, fold=fold, seed=42 + fold, parent=str(PARENT), head_initial=str(head_path(task, fold)),
                module_initial=str(OUT / 'initialization/gate_initial.pt'), scaler=scaler_record, best_epoch=epoch,
                optimizer_groups=groups, optimizer_updates=update_limit, outer_test='NOT_RUN'))
            write(folder / 'scaler.json', scaler_record)
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        values = metrics(truth, prediction)
        if len(ds.materialized) != len(selected):
            raise ValueError('input materialization incomplete')
        ids = selected if formal else entry['validation_indices']
        path = folder / ('test_predictions.npz' if formal else 'validation_predictions.npz')
        if path.exists():
            raise FileExistsError(path)
        np.savez(path, indices=ids, sample_keys=[identities[str(i)]['sample_key'] for i in ids],
            y_true=truth, y_pred=prediction, scaler_mean=scaler_record['mean'], scaler_scale=scaler_record['scale'], best_epoch=epoch)
        write(folder / 'input_rows.json', dict(rows=ds.rows, graphs=len(ds.materialized), fp_read_hits=cache.hits,
            roles='test' if formal else 'train_validation', feature_cache='disabled'))
        write(folder / 'metrics.json', dict(status='PASS', arm=arm, task=task, fold=fold, seed=42 + fold,
            parent=str(PARENT), head_initial=str(head_path(task, fold)), policy=POLICY, scaler=scaler_record,
            best_epoch=epoch, parameters=sum(p.numel() for p in model.parameters()),
            new_parameters=sum(p.numel() for p in model.head.fusion.parameters()),
            optimizer_groups=groups, expected_updates=update_limit, updates=updates, epochs=len(history),
            outer_test='RUN' if formal else 'NOT_RUN', **values))
    except BaseException as error:
        write(folder / 'interrupted_state.json', dict(updates=updates, step=step_state,
            input=None if tracker is None else dict(attempted=tracker.attempted, completed=tracker.completed, inflight=tracker.inflight),
            exact_physical_update_count='unknown if hard interrupted; hooks record boundaries', time=now()))
        write(folder / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            updates=updates, completed_epochs=len(history), time=now()))
        raise
    finally:
        if source is not None:
            source.close()
        if cache is not None:
            cache.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--formal', action='store_true')
    args = p.parse_args()
    run(args.arm, args.task, args.fold, args.formal)
