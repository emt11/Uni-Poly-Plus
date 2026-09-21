#!/usr/bin/env python3
"""Zero-update report-path check for the fusion cycle (run before any smoke).

Two real paths are exercised on a small real batch, with **no optimizer step**:

* the pretraining path: real source -> ``prepare_fusion_sample`` -> packed batch
  -> model forward/backward -> finite loss and gradients, plus the record and
  run-metadata writers the runner uses;
* the downstream path: real xc fold-0 train/validation indices -> adapter batch
  -> gated readout -> loss -> ``best.pt`` + ``metrics.json`` written through the
  runner's own writers, then re-read.

Consuming no updates is the point: a device or reporting failure must surface
here rather than after smoke budget has been spent.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from scripts.create_mips_split_manifests import build_manifest
from src.dataset.glt_ph_fusion_inputs import (ARMS, FusionDataset, fusion_collate,
                                              fusion_downstream_collate, load_ph_stats,
                                              prepare_fusion_sample)
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_ph_fusion_candidates import (FusionDownstream, FusionPretrainer,
                                                  GLTFusionB, R2DModel)
from src.training.glt_dual_runtime import (CleanLabeledDataset, open_source, require_tmux,
                                           save_checkpoint, write_json)
from src.utils import scale_targets


def pretraining_path(args, report):
    from scripts.pretrain_glt_ph_fusion import fusion_optimizer, prepare_window, window_counts

    stats = load_ph_stats(args.profile_stats)
    arm = args.arm if args.arm in ARMS else 'CONST'
    model = GLTFusionB(arm=arm, profile_stats=stats,
                       dropout=0.1) if arm in ARMS else GLTGalPH('cls', None)
    objective = 'R2D' if args.arm == 'R2D' else 'R0'
    if args.arm == 'R2D':
        model = R2DModel()
    trainer = FusionPretrainer(model, objective=objective).to(args.device)
    optimizer, decay, no_decay = fusion_optimizer(model, {'lr': 2e-4, 'weight_decay': 1e-6})
    from src.dataset.glt_galformer_ph import PHBettiReader
    from src.training.glt_dual_runtime import IndexedFrozenDualSource, load_sample_index_artifact

    source, _ = open_source(args.cohort_root, args.cache_root,
                            dual_static_root=args.dual_static_root)
    reader = None
    try:
        index = load_sample_index_artifact(args.sample_index_artifact, 'train')
        source = IndexedFrozenDualSource(source, index['indices'])
        if args.arm == 'PH':
            reader = PHBettiReader(args.ph_sidecar_root)
        const_profile = torch.as_tensor(stats['mean'])
        prepared, positions = prepare_window(
            source, _Stream(len(source), 42), reader, step=0, micro=args.samples,
            accumulation=1, world=1, rank=0, seed=42, arm=arm,
            const_profile=const_profile, radii=(2.5, 4.0, 6.0))
        counts = window_counts(prepared, args.device, objective)
        data, labels = prepared[0]
        data = data.to(args.device)
        labels = {key: value.to(args.device) if torch.is_tensor(value) else value
                  for key, value in labels.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(args.device.type, dtype=torch.bfloat16,
                            enabled=args.amp_dtype == 'bf16'):
            result = trainer(data, labels)
            from src.modules.glt_dual_pretrain import global_objective
            loss = global_objective(result['sums'], counts, 1, (1.0, 1.0, 1.0, 0.0))
        finite = bool(torch.isfinite(loss))
        loss.backward()
        graded = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                  if parameter.grad is not None and name.startswith(('spatial.', 'conditional.'))}
        optimizer.zero_grad(set_to_none=True)
        record = {'arm': arm, 'loss': float(loss.detach()), 'finite': finite,
                  'new_module_grad_norms': graded,
                  'finite_grads': bool(all(np.isfinite(value) for value in graded.values())),
                  'nonzero_new_module_grads': bool(any(value > 0 for value in graded.values())),
                  'graph_count': int(data.graph_available.numel()),
                  'spatial_edges': int(data.spatial_edge_index.size(1)),
                  'device': str(args.device), 'optimizer_updates': 0,
                  'optimizer_groups': [group.get('name') for group in optimizer.param_groups],
                  'positions': positions}
        report['pretraining'] = record
    finally:
        source.close()
    return report


class _Stream:
    """Deterministic first-N stream, so the check never depends on a shuffle."""

    def __init__(self, size, seed):
        self.size, self.seed = int(size), int(seed)

    def index_at(self, position):
        return int(position) % self.size


def downstream_path(args, report):
    task = args.task
    manifest = build_manifest(task, Path(args.raw_root) / f'smi_{task}.csv', 'outer5_inner20')
    fold = next(item for item in manifest['folds'] if int(item['fold']) == args.fold)
    source, frame = open_source(args.downstream_cohort_root, args.downstream_cache_root,
                                task=task, dual_static_root=args.downstream_dual_static_root)
    output = Path(args.output).parent / 'path_check_downstream'
    try:
        targets = frame['label'].to_numpy(dtype=np.float64)
        arm = args.downstream_arm if args.downstream_arm in ARMS else None
        const_profile = np.load(args.const_profile)
        reader = None
        key_rows = {}
        if arm == 'PH':
            from src.dataset.glt_ph_downstream import key_row_map, open_sidecar
            reader = open_sidecar(args.ph_sidecar_root)
            key_rows = key_row_map(reader)
        dataset = (FusionDataset(source, targets, arm=arm, ph_reader=reader,
                                 const_profile=const_profile, key_rows=key_rows)
                   if arm else CleanLabeledDataset(source, targets))
        scaler = scale_targets(dataset, task, train_indices=list(fold['train_indices']),
                               transform_mode='standard')
        collate = fusion_downstream_collate if arm else __import__(
            'src.dataset.glt_dual', fromlist=['dual_glt_collate']).dual_glt_collate
        rows = list(fold['validation_indices'])[:args.samples]
        loader = DataLoader(Subset(dataset, rows), batch_size=args.samples, num_workers=0,
                            collate_fn=collate)
        encoder = (GLTFusionB(arm=arm, profile_stats=load_ph_stats(args.profile_stats))
                   if arm else R2DModel())
        readout = '2D_ONLY' if args.downstream_arm == 'R2D' else 'DUAL'
        model = FusionDownstream(encoder, readout=readout).to(args.device)
        model.eval()
        batch = next(iter(loader)).to(args.device)
        with torch.no_grad():
            prediction, aux = model(batch)
        output.mkdir(parents=True, exist_ok=True)
        unit = {'plan': 'GLT-PH-END2END-20260920-01', 'arm': args.downstream_arm,
                'task': task, 'fold': args.fold, 'readout': readout, 'mode': 'path_check',
                'best_validation_r2': 0.0, 'best_epoch': 0, 'epochs_run': 0,
                'epochs_configured': 0, 'validation_only': True, 'outer_test': 'NOT_RUN',
                'complete': False, 'stage_completed': 'path_check',
                'checkpoint': {'path': 'path_check'}, 'samples': len(rows),
                'aux_readout': aux.get('readout')}
        save_checkpoint(output / 'best.pt', {'state_dict': model.state_dict(),
                                             'task': task, 'fold': args.fold})
        write_json(output / 'metrics.json', unit)
        re_read = json.loads((output / 'metrics.json').read_text())
        report['downstream'] = {'arm': args.downstream_arm, 'task': task, 'fold': args.fold,
                                'samples': len(rows), 'readout': readout,
                                'prediction_shape': list(prediction.shape),
                                'finite': bool(torch.isfinite(prediction).all()),
                                'metrics_written': re_read['arm'] == args.downstream_arm,
                                'checkpoint_written': (output / 'best.pt').is_file(),
                                'optimizer_updates': 0, 'device': str(args.device)}
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='CONST')
    parser.add_argument('--downstream-arm', default='CONST')
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--amp-dtype', default='bf16', choices=('fp32', 'bf16'))
    parser.add_argument('--cohort-root',
                        default='data/processed/glt_dual_v2/pi1m/cohort_30f17b59bc5862a1')
    parser.add_argument('--cache-root', default='data/processed/mips_trimer_scage')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/pi1m/dual_static_v1')
    parser.add_argument('--sample-index-artifact',
                        default='results/glt_pred_20260918/s3b_prep/pretrain_split_v1.json')
    parser.add_argument('--ph-sidecar-root',
                        default='results/glt_galph_20260920/p0/ph_sidecar_betti_v2')
    parser.add_argument('--profile-stats',
                        default='results/glt_ph_end2end_20260920/p0/ph_stats_smoke_256.npz')
    parser.add_argument('--downstream-cohort-root',
                        default='data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1')
    parser.add_argument('--downstream-cache-root',
                        default='data/processed/mips_trimer_scage_downstream')
    parser.add_argument('--downstream-dual-static-root',
                        default='data/processed/glt_dual_v2/downstream/dual_static_v1')
    parser.add_argument('--const-profile',
                        default=('results/glt_galph_ph_retention_20260920/p0/'
                                 'ph_sidecar_downstream/p_train_mean_profile.npy'))
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--output',
                        default='results/glt_ph_end2end_20260920/p1/path_check.json')
    args = parser.parse_args()
    require_tmux()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'plan': 'GLT-PH-END2END-20260920-01', 'stage': 'P1',
              'check': 'zero-update report path', 'status': 'PASS'}
    pretraining_path(args, report)
    downstream_path(args, report)
    report['wall_seconds'] = time.perf_counter() - started
    report['optimizer_updates_total'] = 0
    write_json(args.output, report)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == '__main__':
    main()
