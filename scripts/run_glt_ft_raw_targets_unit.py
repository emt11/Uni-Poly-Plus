#!/usr/bin/env python3
"""N1: original F_CE / UA_2D parents, uniform FT, untransformed labels."""
import argparse
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from scripts import run_glt_align_2d_unit as u1
from scripts import finetune_mcl_ph as old
from src.dataset.glt_dual import build_dual_sample, build_glt_2d_sample, dual_glt_collate, glt_2d_collate
from src.modules.glt_adaptation import AdaptationAdapter
from src.training.glt_dual_runtime import open_source, pretrain_source_fields, save_checkpoint
from src.utils import evaluate, set_global_seed

OUT = ROOT / 'results/glt_optimization_20261011/r7/N1'
LOG = ROOT / 'logs/glt_optimization_20261011/r7/N1'
REVIEW = OUT / 'final_review/recovery1'
U1 = u1.OUT
BASE = u1.BASE
DOWN, SPLIT, PROFILE = u1.DOWN, u1.SPLIT, u1.PROFILE
ARMS = ('F_CE', 'UA_2D')
TASKS = u1.TASKS
REFERENCE_ARMS = {'F_CE': 'U0_FUSED', 'UA_2D': 'UA_2D'}
PARAMETERS = {'F_CE': 39876344, 'UA_2D': 19262233}
read, write, now, strict_restore = u1.read, u1.write, u1.now, u1.strict_restore


def reference_config(arm):
    if arm not in ARMS:
        raise ValueError('invalid N1 arm')
    ref = REFERENCE_ARMS[arm]
    return u1.validate_config(read(U1 / 'initialization' / f'{ref}.json'), ref)


def parent(arm):
    return Path(reference_config(arm)['parent'])


def head_path(arm, task, fold):
    if arm not in ARMS or task not in TASKS or fold not in range(5):
        raise ValueError('invalid N1 unit')
    return u1.head_path(REFERENCE_ARMS[arm], task, fold)


def target_contract():
    return dict(transform='identity', fit='none', scaler=None, inverse_transform=False,
                training_units='original', label_dtype='float32', head_affine_conversion=False)


def make_config(arm):
    ref = reference_config(arm)
    return dict(ref, plan='GLT-OPTIMIZATION-20261011-01', revision='r7', cycle='N1', arm=arm,
                target_transform='identity', scaler_fit='none', target_contract=target_contract(),
                reference_arm=REFERENCE_ARMS[arm], reference_root=str(U1),
                total_parameters=PARAMETERS[arm], new_parameters=0,
                ft_budget=dict(starts=35, epochs=2450, updates=56000, formal_predictions=35))


def validate_config(cfg, arm):
    if cfg != make_config(arm):
        raise ValueError('invalid N1 identity-target configuration')
    return cfg


def ft_folder(arm, task, fold):
    return OUT / 'finetune' / arm / task / f'fold{fold}'


def build_arm(package, arm, fold, cfg):
    validate_config(cfg, arm)
    model = u1.build_arm(package, REFERENCE_ARMS[arm], fold, reference_config(arm))
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
        raise ValueError('model parameter count changed')
    return model


class RawCleanDataset(old.CleanDualDataset):
    """Keep original labels; reuse clean input preparation and its per-unit cache."""
    def __init__(self, source, targets, cfg):
        super().__init__(source, targets)
        self.cfg = cfg
        if self.raw_targets.shape != (len(source),) or not np.isfinite(self.raw_targets).all():
            raise ValueError('raw label schema or finite values mismatch')

    def set_target_override(self, values):
        raise ValueError('N1 forbids target overrides or scaling')

    def __getitem__(self, index):
        index = int(index)
        if index not in self._clean_cache:
            top, tri, text, static, _ = pretrain_source_fields(self.source, index)
            self._clean_cache[index] = (
                build_glt_2d_sample(top, text, static=static)
                if self.cfg['prediction_path'] == 'pure_2d'
                else build_dual_sample(top, tri, text, static=static,
                                       geometry_mode=self.cfg['geometry_mode']))
        data = old.clone_clean_record(self._clean_cache[index])
        data.y = torch.tensor([float(self.raw_targets[index])], dtype=torch.float32)
        if not torch.isfinite(data.y).all():
            raise FloatingPointError('original label cannot be represented in FP32')
        if self.cfg['prediction_path'] == 'fused' and not hasattr(data, 'md200'):
            data.md200 = torch.zeros((1, 200))
            data.md200_valid = torch.zeros(1, dtype=torch.bool)
        return data


def validate_best(best, arm, task, fold, cfg):
    validate_config(cfg, arm)
    expected = dict(config=cfg, task=task, fold=fold, seed=42 + fold,
                    parent=str(parent(arm)), head_initial=str(head_path(arm, task, fold)),
                    prediction_path=cfg['prediction_path'], input_dim=cfg['input_dim'],
                    target_transform='identity', scaler_fit='none', target_contract=target_contract())
    if any(best.get(k) != v for k, v in expected.items()):
        raise ValueError('N1 best metadata mismatch')
    if any(k in best for k in ('scaler_mean', 'scaler_scale', 'scaler')):
        raise ValueError('scaled best cannot be loaded as raw-target best')
    if type(best.get('best_epoch')) is not int or not 1 <= best['best_epoch'] <= 70:
        raise ValueError('invalid validation-selected epoch')


def frozen_binding(selection, arm, task, fold, cfg, best):
    expected_units = {(a, t, f) for a in ARMS for t in TASKS for f in range(5)}
    actual_units = [(u['arm'], u['task'], u['fold']) for u in selection.get('units', [])]
    if (selection.get('status') != 'FROZEN' or selection.get('validation_units') != 70
            or len(actual_units) != 70 or set(actual_units) != expected_units):
        raise ValueError('all 70 FT units must be frozen before any formal prediction')
    matches = [u for u in selection['units'] if (u['arm'], u['task'], u['fold']) == (arm, task, fold)]
    if len(matches) != 1 or selection['configs'][arm] != cfg:
        raise ValueError('frozen unit/config mismatch')
    binding = matches[0]
    expected = dict(best=str(ft_folder(arm, task, fold) / 'best.pt'), parent=str(parent(arm)),
                    head_initial=str(head_path(arm, task, fold)), best_epoch=best['best_epoch'],
                    target_contract=target_contract(), prediction_path=cfg['prediction_path'],
                    input_dim=cfg['input_dim'])
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError('frozen best/parent/target/path mismatch')
    return binding


def initialize():
    folder = OUT / 'initialization'
    folder.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        cfg = make_config(arm)
        if not parent(arm).is_file():
            raise FileNotFoundError('required existing PT parent: ' + str(parent(arm)))
        for task in TASKS:
            for fold in range(5):
                if not head_path(arm, task, fold).is_file():
                    raise FileNotFoundError('required original head: ' + str(head_path(arm, task, fold)))
        write(folder / f'{arm}.json', cfg)
    write(folder / 'summary.json', dict(arms=ARMS, parameters=PARAMETERS, new_parameters=0,
                                      pt_starts=0, target_contract=target_contract(), time=now()))


def run_ft(arm, task, fold, formal=False):
    folder = OUT / ('formal' if formal else 'finetune') / arm / task / f'fold{fold}'
    folder.mkdir(parents=True, exist_ok=False)
    updates, history, source = {'attempted': 0, 'completed': 0}, [], None
    try:
        cfg = validate_config(read(OUT / 'initialization' / f'{arm}.json'), arm)
        set_global_seed(42 + fold)
        package = torch.load(parent(arm), map_location='cpu', weights_only=False)
        model = build_arm(package, arm, fold, cfg)
        head = torch.load(head_path(arm, task, fold), map_location='cpu', weights_only=False)
        strict_restore(old.periodic_tdl_head(model), head)
        model.to('cuda:0')
        old.configure_periodic_tdl_dropout(model, task, profile='uniform_u1')
        manifest = read(SPLIT / f'{task}.json')
        entry = manifest['folds'][fold]
        if formal:
            best = torch.load(ft_folder(arm, task, fold) / 'best.pt', map_location='cpu', weights_only=False)
            validate_best(best, arm, task, fold, cfg)
            binding = frozen_binding(read(OUT / 'evaluation_before_test.json'), arm, task, fold, cfg, best)
            if read(ft_folder(arm, task, fold) / 'target.json') != binding['target_contract']:
                raise ValueError('frozen target contract changed')
            strict_restore(model, best['state_dict'])
            selected = sorted(entry['test_indices'])
            local_val = list(range(len(selected)))
            if binding['test_indices'] != selected:
                raise ValueError('frozen test indices changed')
        else:
            selected, local_train, local_val = old.compact_train_validation_indices(
                entry['train_indices'], entry['validation_indices'])
        source, frame = open_source(DOWN, DOWN, task=task, dual_static_root=DOWN,
                                   selected_indices=selected, split_manifest=manifest,
                                   record_index_path=DOWN / f'{task}_offsets.npy',
                                   expected_task_rows=manifest['sample_count'], verify_key_digest=False)
        if frame['original_row'].astype(int).tolist() != selected:
            raise ValueError('selected source rows differ')
        ds = RawCleanDataset(source, frame['label'].to_numpy(dtype=np.float64), cfg)
        set_global_seed(42 + fold)
        collate = glt_2d_collate if cfg['prediction_path'] == 'pure_2d' else dual_glt_collate
        loader = DataLoader(Subset(ds, local_val), batch_size=24, shuffle=False,
                            num_workers=0, collate_fn=collate)
        if formal:
            _, r2, truth, prediction = evaluate(AdaptationAdapter(model), loader, nn.MSELoss(),
                                              'cuda:0', scaler=None, amp_dtype='fp32')
            epoch, total_epochs = best['best_epoch'], 0
        else:
            save_checkpoint(folder / 'head_initial.pt', head)
            write(folder / 'target.json', target_contract())
            train = DataLoader(Subset(ds, local_train), batch_size=24, shuffle=True,
                               num_workers=0, collate_fn=collate,
                               generator=torch.Generator().manual_seed(42 + fold))

            def observe(kind):
                updates[kind] += 1
                write(folder / 'updates.json', updates)

            def progress(rec):
                rec = dict(rec, prediction_path=cfg['prediction_path'], target_transform='identity')
                history.append(rec)
                print(u1.json.dumps(rec, allow_nan=False), flush=True)
                write(folder / 'progress.json', dict(history=history, updates=updates, outer_test='NOT_RUN'))

            result = old.run_periodic_tdl_epochs(model, train, loader, nn.MSELoss(), 'cuda:0',
                                                task=task, scaler=None, progress=progress,
                                                profile='uniform_u1', step_observer=observe)
            if updates['attempted'] != updates['completed'] or updates['completed'] != 70 * len(train):
                raise ValueError('FT update budget mismatch')
            truth, prediction = result['predictions']
            epoch, total_epochs, r2 = result['best_epoch'], len(history), result['best_r2']
            save_checkpoint(folder / 'best.pt', dict(state_dict=result['state'], config=cfg, task=task,
                fold=fold, seed=42 + fold, parent=str(parent(arm)), best_epoch=epoch, history=history,
                optimizer_groups=result['optimizer_groups_by_stage'], optimizer_updates=updates['completed'],
                head_initial=str(head_path(arm, task, fold)), prediction_path=cfg['prediction_path'],
                input_dim=cfg['input_dim'], target_transform='identity', scaler_fit='none',
                target_contract=target_contract(), outer_test='NOT_RUN'))
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        raw = ds.raw_targets[local_val]
        if not np.array_equal(truth, raw.astype(np.float32).astype(np.float64)):
            raise ValueError('evaluation truth differs from untransformed FP32 labels')
        if not np.isfinite(prediction).all() or not math.isfinite(float(r2)):
            raise FloatingPointError('nonfinite predictions/metrics')
        ids = selected if formal else entry['validation_indices']
        np.savez(folder / ('test_predictions.npz' if formal else 'validation_predictions.npz'),
                 y_true=truth, y_pred=prediction, raw_labels=raw,
                 sample_keys=np.asarray([source.samples[i][0].hex() for i in local_val]),
                 indices=np.asarray(ids), best_epoch=epoch, target_transform='identity',
                 geometry_valid=np.asarray([source.entries[i]['geometry_status'] == 'PASS' for i in local_val]),
                 outer_test='RUN' if formal else 'NOT_RUN')
        # Recompute on the exact saved arrays, as in existing formal summaries.
        from sklearn.metrics import r2_score
        write(folder / 'metrics.json', dict(status='PASS', arm=arm, task=task, fold=fold, modes=cfg,
              parameters=PARAMETERS[arm], epochs=total_epochs, updates=updates,
              r2=float(r2_score(truth, prediction)), rmse=float(np.sqrt(np.mean((truth - prediction) ** 2))),
              target_contract=target_contract(), outer_test='RUN' if formal else 'NOT_RUN'))
    except BaseException as error:
        write(folder / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                          updates=updates, completed_epochs=len(history), time=now()))
        raise
    finally:
        if source is not None:
            source.close()


if __name__ == '__main__':
    from src.training.glt_dual_runtime import require_tmux
    require_tmux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    if args.initialize and not any((args.arm, args.task, args.fold is not None, args.formal)):
        initialize()
    elif not args.initialize and args.arm and args.task and args.fold is not None:
        run_ft(args.arm, args.task, args.fold, args.formal)
    else:
        raise ValueError('invalid N1 unit arguments')
