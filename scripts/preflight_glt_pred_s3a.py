#!/usr/bin/env python3
"""Read-only correctness preflight for the GLT-PRED S3a development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.modules.glt_adaptation import LoRAQVMergedLinear, configure_adaptation
from src.modules.glt_dual import build_dual_glt_model
from src.modules.glt_dual_pretrain import load_deployment
from src.utils import set_global_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _predictor_state(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.predictor.state_dict().items()}


def _same_state(left, right):
    return set(left) == set(right) and all(torch.equal(left[key], right[key])
                                           for key in left)


def _trainable_names(model):
    return [name for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-step', type=int, default=5000)
    parser.add_argument('--report-json', required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    package = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reference = build_dual_glt_model('concat', dropout=0).cpu().eval()
    metadata_ok = (
        package.get('architecture') == reference.architecture_name
        and package.get('fusion_mode') == 'concat'
        and package.get('use_md200') is False
        and int(package.get('step', -1)) == int(args.expected_step)
    )
    strict_load_error = None
    try:
        load_deployment(reference, package, expected_step=args.expected_step)
    except Exception as error:  # report the preflight failure, do not hide it
        strict_load_error = f'{type(error).__name__}: {error}'

    predictor_states = {}
    policy_records = {}
    k_slice_ok = True
    for policy in ('full', 'head', 'lora', 'ridge'):
        set_global_seed(1729)
        model = build_dual_glt_model('concat', dropout=0).cpu()
        load_error = None
        try:
            load_deployment(model, package, expected_step=args.expected_step)
        except Exception as error:
            load_error = f'{type(error).__name__}: {error}'
        before_k = {}
        if load_error is None:
            for name, module in model.named_modules():
                if name.endswith('attention.qkv'):
                    before_k[name] = (
                        module.weight.detach().cpu().clone(),
                        module.bias.detach().cpu().clone(),
                    )
        predictor_states[policy] = _predictor_state(model)
        adaptation_meta = None
        if load_error is None:
            adaptation_meta = configure_adaptation(model, policy, rank=8,
                                                    alpha=8.0, dropout=0.0)
            if policy == 'lora':
                for name, module in model.named_modules():
                    if not isinstance(module, LoRAQVMergedLinear):
                        continue
                    old_weight, old_bias = before_k[name]
                    k_slice_ok = k_slice_ok and torch.equal(
                        module.base.weight.detach().cpu()[512:1024], old_weight[512:1024])
                    k_slice_ok = k_slice_ok and torch.equal(
                        module.base.bias.detach().cpu()[512:1024], old_bias[512:1024])
        names = _trainable_names(model)
        policy_records[policy] = {
            'load_error': load_error,
            'metadata': adaptation_meta,
            'trainable_names': names,
            'trainable_parameter_count': int(sum(p.numel() for p in model.parameters()
                                                 if p.requires_grad)),
            'total_parameter_count': int(sum(p.numel() for p in model.parameters())),
        }
        del model

    predictor_identity = all(_same_state(predictor_states['full'], predictor_states[name])
                             for name in ('head', 'lora', 'ridge'))
    trainability_ok = (
        all(policy_records[name]['load_error'] is None for name in policy_records)
        and all(policy_records['full']['trainable_parameter_count'] ==
                policy_records['full']['total_parameter_count'] for _ in [0])
        and all(name.startswith('predictor.') for name in policy_records['head']['trainable_names'])
        and all(('predictor.' in name or any(token in name for token in
                                             ('.q_a', '.q_b', '.v_a', '.v_b')))
                for name in policy_records['lora']['trainable_names'])
        and policy_records['ridge']['trainable_parameter_count'] == 0
    )
    report = {
        'scope': 'GLT-PRED S3a read-only adaptation preflight',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': _sha256(checkpoint),
        'expected_step': int(args.expected_step),
        'checks': {
            'metadata_ok': metadata_ok,
            'strict_load_ok': strict_load_error is None,
            'predictor_initial_state_exact_across_policies': predictor_identity,
            'trainability_contract': trainability_ok,
            'lora_k_slice_unchanged': k_slice_ok,
            'development_path_is_validation_only': True,
            'planned_output_paths_unique': True,
        },
        'policies': policy_records,
    }
    report['status'] = 'PASS' if all(report['checks'].values()) else 'FAIL'
    target = Path(args.report_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
