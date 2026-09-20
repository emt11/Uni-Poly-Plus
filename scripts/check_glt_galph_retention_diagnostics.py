#!/usr/bin/env python3
"""Check the reporting path before spending any epoch: real batch, real deployment.

Builds the F_REAL arm on the recorded C1_REPAIR_5K deployment, takes the first few
validation samples of one fold through the production collate, and calls the
runner's own ``ph_diagnostics`` (the function that failed in r5).  No optimizer is
created and no step is taken: this decides whether the smoke can be attempted.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.finetune_glt_galformer_ph import fixed_manifest
from scripts.finetune_glt_galformer_ph_retention import (GROUPS_TO_MODE,
                                                         ph_diagnostics, sha256_file)
from src.dataset.glt_ph_downstream import (PHRetentionDataset, key_row_map,
                                           load_const_profile, open_sidecar,
                                           retention_collate)
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.modules.glt_galformer_ph_retention import GalformerPHDownstream
from src.modules.glt_galph_checkpoint_identity import load_identity, verify_checkpoint
from src.training.glt_dual_runtime import open_source, require_tmux, write_json
from src.utils import set_global_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint',
                        default='results/glt_galph_ph_retention_20260920/p1/'
                                 'pretrain_C1_REPAIR_5K/deploy_05000.pt')
    parser.add_argument('--checkpoint-identity',
                        default='configs/mts/glt_galph_c1_repair_5k_identity.json')
    parser.add_argument('--ph-sidecar',
                        default='results/glt_galph_ph_retention_20260920/p0/'
                                 'ph_sidecar_downstream')
    parser.add_argument('--const-profile',
                        default='results/glt_galph_ph_retention_20260920/p0/'
                                 'ph_sidecar_downstream/p_train_mean_profile.npy')
    parser.add_argument('--raw-root', default='data/raw')
    parser.add_argument('--split-root', default='data/splits/mips_outer5_inner20')
    parser.add_argument('--cohort-root',
                        default='data/processed/glt_dual_v2/downstream/cohort_1545eda5a8f6a1')
    parser.add_argument('--cache-root',
                        default='data/processed/mips_trimer_scage_downstream')
    parser.add_argument('--dual-static-root',
                        default='data/processed/glt_dual_v2/downstream/dual_static_v1')
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    require_tmux()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity = load_identity(args.checkpoint_identity)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    verified = verify_checkpoint(identity, args.checkpoint, package,
                                 sha256=sha256_file(args.checkpoint))
    manifest = fixed_manifest(args.task, Path(args.raw_root) / f'smi_{args.task}.csv',
                              Path(args.split_root) / f'{args.task}.json')
    fold = [item for item in manifest['folds'] if item['fold'] == args.fold][0]
    indices = list(fold['validation_indices'])[:args.samples]
    reader = open_sidecar(args.ph_sidecar)
    const = load_const_profile(args.const_profile)
    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
                                dual_static_root=args.dual_static_root)
    payload = {'checkpoint_identity': verified, 'task': args.task, 'fold': args.fold,
               'validation_samples_used': len(indices), 'optimizer_updates': 0, 'arms': {}}
    try:
        targets = frame['label'].to_numpy(dtype=np.float64)
        for group in ('F_OFF', 'F_CONST', 'F_REAL'):
            dataset = PHRetentionDataset(
                source, targets, group=group, reader=reader,
                const_profile=(const if group == 'F_CONST' else None),
                key_rows=key_row_map(reader))
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False,
                                collate_fn=retention_collate)
            set_global_seed(42 + args.fold)
            encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'))
            load_galformer_deployment(encoder, package, int(identity['step']))
            model = GalformerPHDownstream(encoder, readout='DUAL',
                                          ph_residual=GROUPS_TO_MODE[group]).to(device)
            model.freeze_training_only_heads()
            with torch.no_grad():
                model.gamma.fill_(0.02)
            payload['arms'][group] = ph_diagnostics(model, loader, device, const)
            del model, encoder
    finally:
        source.close()
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
