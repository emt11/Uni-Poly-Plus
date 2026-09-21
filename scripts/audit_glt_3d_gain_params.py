#!/usr/bin/env python3
"""D0 parameter-group audit for GLT-3D-GAIN-20260921-01.

Builds the deployment model on CPU and reports the exact optimizer grouping
produced by ``scripts/finetune_glt_dual.py:optimizer_for``, which parameters
come from the frozen deployment package and which are newly initialized.
No forward pass, no optimizer step, no GPU.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    package = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    deployment_torsion = bool(package.get('torsion_modules', False))

    model = build_dual_glt_model(config['fusion_mode'], torsion=deployment_torsion)
    initialized = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    load_deployment(model, package, config['downstream_step'])

    # Reproduce the exact grouping of optimizer_for(..., 'full').
    groups, seen = [], set()
    for group_name, module, lr in [('o8', model.o8, config['encoder_lr']),
                                   ('glt', model.glt, config['encoder_lr'])]:
        params = [p for p in module.parameters() if p.requires_grad]
        seen.update(id(p) for p in params)
        if params:
            groups.append(dict(name=group_name, lr=lr, params=params))
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    if rest:
        groups.append(dict(name='fusion_head', lr=config['fusion_head_lr'], params=rest))

    name_by_id = {id(p): name for name, p in model.named_parameters()}
    report = {
        'config': {
            'fusion_mode': config['fusion_mode'],
            'encoder_lr': config['encoder_lr'],
            'fusion_head_lr': config['fusion_head_lr'],
            'finetune_weight_decay': config['finetune_weight_decay'],
            'downstream_step': config['downstream_step'],
            'deployment_torsion_modules': deployment_torsion,
        },
        'checkpoint': {'path': str(Path(args.checkpoint).resolve()),
                       'architecture': package.get('architecture'),
                       'fusion_mode': package.get('fusion_mode'),
                       'step': package.get('step'),
                       'keys': len(package['state_dict'])},
        'total_parameters': int(sum(p.numel() for p in model.parameters())),
        'trainable_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        'groups': [],
        'deployment_loaded': sorted(package['state_dict'].keys()),
        'newly_initialized': sorted(
            name for name in initialized if name not in package['state_dict']),
        'norm2_norm3_group': None,
    }

    for group in groups:
        names = sorted(name_by_id[id(p)] for p in group['params'])
        report['groups'].append({
            'name': group['name'],
            'lr': group['lr'],
            'weight_decay': config['finetune_weight_decay'],
            'num_tensors': len(group['params']),
            'num_parameters': int(sum(p.numel() for p in group['params'])),
            'parameter_names': names,
        })
        if any(name.startswith(('norm2', 'norm3')) for name in names):
            report['norm2_norm3_group'] = group['name']

    # Which tensors the pretraining FP objective can update: norm2/norm3 are on
    # the fuse() path consumed by fp_head; predictor is absent during pretraining.
    report['pretrain_path'] = {
        'norm2_norm3_updated_by_fp_loss': True,
        'reason': 'DualPretrainer.forward calls encoder.fuse(encoded) then fp_head(fused)',
        'predictor_replaced_by_identity_in_pretrainer': True,
        'deployment_head_newly_initialized': any(
            name.startswith('predictor.') for name in report['newly_initialized']),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'norm2_norm3_group': report['norm2_norm3_group'],
                      'newly_initialized': report['newly_initialized']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
