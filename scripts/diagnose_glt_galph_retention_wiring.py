#!/usr/bin/env python3
"""Read-only wiring check for the three PH retention arms (0 optimizer updates).

The r5 smoke budget was consumed by three runs that trained their two epochs and
then died in reporting, so this script answers the smoke's decision-relevant
question without any training: on the real xc/fold0 fold, do the three arms see
different PH inputs, and does the frozen encoder turn them into different
representations?  It loads the recorded deployment, builds the three arms in the
same order with the same seed, and compares the encoder summaries sample by
sample.

Nothing is trained, no optimizer exists here, and the forced-gate section is an
explicit counterfactual whose model state is restored bit-exactly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.finetune_glt_galformer_ph import fixed_manifest
from scripts.finetune_glt_galformer_ph_retention import (GROUPS_TO_MODE, sha256_file)
from src.dataset.glt_ph_downstream import (PHRetentionDataset, key_row_map,
                                           load_const_profile, open_sidecar,
                                           retention_collate)
from src.modules.glt_dual import mean_pool
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment
from src.modules.glt_galformer_ph_retention import GalformerPHDownstream
from src.modules.glt_galph_checkpoint_identity import load_identity, verify_checkpoint
from src.training.glt_dual_runtime import open_source, require_tmux, write_json
from src.utils import set_global_seed

FORCED_GATE = 0.02
GROUPS = ('F_OFF', 'F_CONST', 'F_REAL')


def digest(tensor):
    array = np.ascontiguousarray(tensor.detach().float().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def arm_pass(model, loader, const, device, forced_gate):
    """Encoder summaries, the model's own residual and the input it read."""
    model.eval()
    original = model.gamma.detach().clone()
    summaries, summary_vs_const, residuals, references = [], [], [], []
    input_norms, input_spreads, input_vs_const, profiles_seen = [], [], [], []
    valid_flags, proportions, keys = [], [], []
    try:
        with torch.no_grad():
            model.gamma.fill_(float(forced_gate))
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                graphs = int(batch.graph_available.numel())
                profiles = batch.ph_profile.float()
                mask = batch.ph_mask.bool()
                summary = model.encoder.ph_encoder.summarize(
                    model.encoder.ph_encoder(profiles, mask)).float()
                const_rows = const.to(device=profiles.device).unsqueeze(0) \
                    .expand_as(profiles).contiguous()
                fixed = model.encoder.ph_encoder.summarize(
                    model.encoder.ph_encoder(const_rows, mask)).float()
                out = model.encoder(batch)
                mean3 = mean_pool(out['bond_states'], batch.bond_batch.long(), graphs)
                reference = model.readout3(torch.cat([out['cls3'], mean3], -1))
                contribution = model.ph_contribution(batch, reference)
                summaries.append(summary.cpu())
                summary_vs_const.append(float((summary - fixed).abs().max()))
                profiles_seen.append(profiles.cpu())
                residuals.append(contribution.cpu())
                references.append(reference.cpu())
                input_norms.append(float(profiles.norm(dim=(1, 2)).mean()))
                input_spreads.append(float((profiles.max(0).values
                                            - profiles.min(0).values).max()))
                input_vs_const.append(float((profiles - const_rows).abs().max()))
                valid_flags.append(batch.ph_valid.bool().cpu())
                proportions.append(float(model.last_ph_stats['ph_valid_fraction']))
                keys.extend(batch.sample_keys)
    finally:
        with torch.no_grad():
            model.gamma.copy_(original)
    summary = torch.cat(summaries, 0)
    residual = torch.cat(residuals, 0)
    reference = torch.cat(references, 0)
    return {
        'samples': int(summary.size(0)),
        'summary_digest': digest(summary),
        'profile_digest': digest(torch.cat(profiles_seen, 0)),
        'summary_norm_mean': float(summary.norm(dim=-1).mean()),
        'summary_cross_sample_spread': float((summary.max(0).values
                                              - summary.min(0).values).max()),
        'summary_vs_const_max_abs_mean': float(np.mean(summary_vs_const)),
        'input_norm_mean': float(np.mean(input_norms)),
        'input_cross_sample_spread': float(np.mean(input_spreads)),
        'input_vs_const_max_abs': float(np.mean(input_vs_const)),
        'residual_norm_mean': float(residual.norm(dim=-1).mean()),
        'residual_relative_norm': float(residual.norm() / reference.norm().clamp_min(1e-30)),
        'ph_valid_fraction': float(torch.cat(valid_flags).float().mean()),
        'reported_ph_valid_fraction': float(np.mean(proportions)),
        'sample_keys_digest': hashlib.sha256(''.join(keys).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='xc')
    parser.add_argument('--fold', type=int, default=0)
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
    parser.add_argument('--eval-batch', type=int, default=64)
    args = parser.parse_args()
    require_tmux()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity = load_identity(args.checkpoint_identity)
    checkpoint_sha = sha256_file(args.checkpoint)
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    verified = verify_checkpoint(identity, args.checkpoint, package, sha256=checkpoint_sha)

    manifest = fixed_manifest(args.task, Path(args.raw_root) / f'smi_{args.task}.csv',
                              Path(args.split_root) / f'{args.task}.json')
    fold = [item for item in manifest['folds'] if item['fold'] == args.fold][0]
    indices = list(fold['train_indices']) + list(fold['validation_indices'])
    reader = open_sidecar(args.ph_sidecar)
    key_rows = key_row_map(reader)
    const = load_const_profile(args.const_profile)
    source, frame = open_source(args.cohort_root, args.cache_root, task=args.task,
                                dual_static_root=args.dual_static_root)
    payload = {'task': args.task, 'fold': args.fold,
               'samples': f'train+validation ({len(indices)})',
               'checkpoint_identity': verified, 'forced_gate': FORCED_GATE,
               'optimizer_updates': 0, 'arms': {}}
    try:
        targets = frame['label'].to_numpy(dtype=np.float64)
        for group in GROUPS:
            dataset = PHRetentionDataset(
                source, targets, group=group, reader=reader,
                const_profile=(const if group == 'F_CONST' else None),
                key_rows=key_rows)
            loader = DataLoader(Subset(dataset, indices), batch_size=args.eval_batch,
                                shuffle=False, collate_fn=retention_collate)
            set_global_seed(42 + args.fold)
            encoder = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'))
            load_galformer_deployment(encoder, package, int(identity['step']))
            model = GalformerPHDownstream(encoder, readout='DUAL',
                                          ph_residual=GROUPS_TO_MODE[group]).to(device)
            frozen = model.freeze_training_only_heads()
            payload['arms'][group] = arm_pass(model, loader, const, device, FORCED_GATE)
            payload['arms'][group]['frozen_training_only_heads'] = frozen
            payload['arms'][group]['gamma_before_any_update'] = float(model.gamma.detach())
            del model, encoder
        payload['summary_digest_unique'] = len({payload['arms'][group]['summary_digest']
                                                for group in GROUPS}) == len(GROUPS)
        payload['profile_digest_unique'] = len({payload['arms'][group]['profile_digest']
                                                for group in GROUPS}) == len(GROUPS)
        payload['sample_order_identical'] = len({payload['arms'][group]['sample_keys_digest']
                                                 for group in GROUPS}) == 1
        payload['pairwise'] = {f'{left}_vs_{right}': {
            'encoder_summary_digest_equal': payload['arms'][left]['summary_digest']
            == payload['arms'][right]['summary_digest'],
            'encoder_summary_cross_sample_spread_equal': payload['arms'][left][
                'summary_cross_sample_spread']
            == payload['arms'][right]['summary_cross_sample_spread'],
            'residual_norm_equal': payload['arms'][left]['residual_norm_mean']
            == payload['arms'][right]['residual_norm_mean']}
            for left in GROUPS for right in GROUPS if left < right}
    finally:
        source.close()
    write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in
                      ('checkpoint_identity', 'forced_gate', 'summary_digest_unique',
                       'profile_digest_unique', 'sample_order_identical', 'pairwise')},
                     indent=2))
    for group in GROUPS:
        arm = payload['arms'][group]
        print(json.dumps({group: {key: arm[key] for key in
                                  ('profile_digest', 'summary_digest',
                                   'summary_cross_sample_spread',
                                   'summary_vs_const_max_abs_mean', 'input_norm_mean',
                                   'input_cross_sample_spread', 'input_vs_const_max_abs',
                                   'residual_relative_norm', 'ph_valid_fraction',
                                   'gamma_before_any_update')}}))


if __name__ == '__main__':
    main()
