#!/usr/bin/env python3
"""Read-only diagnostics for the PH retention blocker (no training, no writes).

Records why F_CONST and F_REAL are identical under the frozen C1 encoder:

1. parameter statistics of the PH encoder in the pretrained deployments
2. the decay mechanism of Adam(coupled weight_decay) on a zero-gradient tensor
3. the collapse timeline from the C1 resume checkpoints
4. input independence of the frozen encoder over real downstream profiles

Writes one JSON file under ``results/glt_galph_ph_retention_20260920/p1/``.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn

from src.dataset.glt_ph_downstream import (load_const_profile, key_row_map, open_sidecar)
from src.modules.glt_galformer_ph import GLTGalPH
from src.modules.glt_galformer_ph_pretrain import load_galformer_deployment

DECAY_STEPS = 5000
TRACKED = ('ph_encoder.patch.0.weight', 'ph_encoder.patch.2.weight',
           'ph_encoder.scale_fuse.0.weight', 'ph_encoder.scale_embedding',
           'ph_to_summary.weight')


def deployment_stats(path):
    state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    return {name: round(float(state[name].norm()), 8) for name in TRACKED} | {
        'alpha_ph': float(state['alpha_ph']),
        'tanh_alpha_ph': float(torch.tanh(state['alpha_ph']))}


def adam_decay_check():
    torch.manual_seed(0)
    plain = nn.Parameter(torch.randn(4096) * 0.02)
    initial = float(plain.norm())
    optimizer = torch.optim.Adam([plain], lr=2e-4, weight_decay=1e-6)
    for _ in range(DECAY_STEPS):
        optimizer.zero_grad(set_to_none=True)
        (plain.sum() * 0.0).backward()          # zero loss gradient
        optimizer.step()
    torch.manual_seed(0)
    decoupled = nn.Parameter(torch.randn(4096) * 0.02)
    optimizer_w = torch.optim.AdamW([decoupled], lr=2e-4, weight_decay=1e-6)
    for _ in range(DECAY_STEPS):
        optimizer_w.zero_grad(set_to_none=True)
        (decoupled.sum() * 0.0).backward()
        optimizer_w.step()
    return {'initial_norm': round(initial, 6),
            'adam_coupled_norm_after_5000': float(plain.norm()),
            'adam_coupled_max_abs_after_5000': float(plain.abs().max()),
            'adamw_decoupled_norm_after_5000': round(float(decoupled.norm()), 6),
            'steps': DECAY_STEPS, 'lr': 2e-4, 'weight_decay': 1e-6}


def timeline(root):
    rows = {}
    for step in (1000, 2000, 3000, 4000, 5000):
        path = Path(root) / f'resume_{step:05d}.pt'
        if not path.is_file():
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        rows[str(step)] = {name: round(float(state[f'model.{name}'].norm()), 10)
                           for name in TRACKED} | {
            'alpha_ph': float(state['model.alpha_ph'])}
    return rows


def encoder_input_dependence(deployment, sidecar, const_path, samples=64):
    package = torch.load(deployment, map_location='cpu', weights_only=False)
    model = GLTGalPH(str(package['summary_mode']), package.get('ph_mode'))
    load_galformer_deployment(model, package, int(package['step']))
    encoder = model.ph_encoder.eval()
    reader = open_sidecar(sidecar)
    rows = key_row_map(reader)
    keys = list(rows)[:samples]
    profiles = torch.stack([
        torch.as_tensor(reader.get(rows[key], key)[0], dtype=torch.float32) for key in keys])
    const = load_const_profile(const_path)
    mask = torch.zeros(len(keys), 8, dtype=torch.bool)
    with torch.no_grad():
        real = encoder.summarize(encoder(profiles, mask))
        average = encoder.summarize(
            encoder(const.unsqueeze(0).expand(len(keys), -1, -1).contiguous(), mask))
        pairwise = float((real[0:1] - real).abs().max())
    return {'samples': len(keys),
            'max_pairwise_output_diff': pairwise,
            'real_vs_const_bit_identical': bool(torch.equal(real, average)),
            'per_dim_spread_across_samples': float((real.max(0).values
                                                    - real.min(0).values).max()),
            'p_ph_norm': round(float(real[0].norm()), 8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deployments', nargs='*',
                        default=['results/glt_galph_20260920/p2/c1/pretrain/deploy_05000.pt',
                                 'results/glt_galph_20260920/p2/n1/pretrain/deploy_05000.pt'])
    parser.add_argument('--resume-root', default='results/glt_galph_20260920/p2/c1/pretrain')
    parser.add_argument('--sidecar', default='results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream')
    parser.add_argument('--const-profile',
                        default='results/glt_galph_ph_retention_20260920/p0/ph_sidecar_downstream/p_train_mean_profile.npy')
    args = parser.parse_args()
    payload = {
        'deployments': {Path(path).parent.parent.name: deployment_stats(path)
                        for path in args.deployments},
        'adam_decay_check': adam_decay_check(),
        'collapse_timeline_C1': timeline(args.resume_root),
        'frozen_encoder_input_dependence_C1': encoder_input_dependence(
            args.deployments[0], args.sidecar, args.const_profile),
        'interpretation': ('the frozen PH encoder is a constant function, so F_CONST and '
                           'F_REAL are bit-identical and the pre-registered contrast cannot '
                           'be measured with this checkpoint'),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
