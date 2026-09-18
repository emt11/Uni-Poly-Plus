#!/usr/bin/env python3
"""Freeze a common step-0 state for the four S3b objective variants."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.modules.glt_dual_pretrain import DualPretrainer
from src.utils import set_global_seed


ALLOWED_DIFFERING = ('fp_head.', 'fgr_head.', 'fgr_context.',
                     'align_proj2.', 'align_proj3.')


def _tensor_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode('utf-8'))
        value = state[name].detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build(output_root, *, geometry_head_norm=True):
    variants = {}
    for task in ('fp', 'none', 'fgr', 'align'):
        set_global_seed(42)
        model = DualPretrainer(
            'concat', geometry_head_norm=geometry_head_norm,
            third_task=task, fgr_mu=0.0, fgr_sigma=1.0,
            align_temperature=0.1)
        variants[task] = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        del model
    common_keys = sorted(set.intersection(*(set(state) for state in variants.values())))
    differing = sorted(set.union(*(set(state) for state in variants.values())) - set(common_keys))
    unexpected = [name for name in differing
                  if not any(name.startswith(prefix) for prefix in ALLOWED_DIFFERING)]
    if unexpected:
        raise ValueError('unexpected task-specific initialization keys: ' + ','.join(unexpected[:10]))
    # Copy the FP common tensors into all variants and prove equality.  The
    # artifact is what future formal runners consume; this prevents module
    # construction order from changing the shared encoder initialization when
    # objective-specific heads differ.
    common_state = {name: variants['fp'][name] for name in common_keys}
    for task in variants:
        for name in common_keys:
            variants[task][name] = common_state[name].clone()
    exact = all(torch.equal(variants['fp'][name], variants[task][name])
                for task in variants for name in common_keys)
    if not exact:
        raise AssertionError('common initialization parity failed')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / 'common_init_v1.pt'
    torch.save({'schema_version': 'glt-pred-common-init-v1',
                'fusion_mode': 'concat', 'seed': 42,
                'common_state_dict': common_state,
                'common_state_sha256': _tensor_hash(common_state),
                'allowed_differing_prefixes': list(ALLOWED_DIFFERING)}, state_path)
    parity = {
        'schema_version': 'glt-pred-common-init-parity-v1',
        'seed': 42,
        'fusion_mode': 'concat',
        'geometry_head_norm': bool(geometry_head_norm),
        'variants': ['fp', 'none', 'fgr', 'align'],
        'common_key_count': len(common_keys),
        'task_specific_keys': differing,
        'allowed_differing_prefixes': list(ALLOWED_DIFFERING),
        'common_state_sha256': _tensor_hash(common_state),
        'common_init_artifact': str(state_path.resolve()),
        'common_initialization_exact': bool(exact),
    }
    parity_path = output_root / 'common_init_parity.json'
    parity_path.write_text(json.dumps(parity, ensure_ascii=False, indent=2,
                                      sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--geometry-head-norm', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.output_root, geometry_head_norm=args.geometry_head_norm),
                     ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
