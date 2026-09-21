#!/usr/bin/env python3
"""r3 phase-2 prerequisite: the shared v2 initial state of the three arms.

Construction only -- no forward, no backward, no optimizer step, no data.  The
r3 order asks the execution record to show that the three arms start from the
fixed v2 common initialization, that no arm resumes the rejected v1 artifact or
a mis-initialised r1 checkpoint, and that the router / gate keep their own
declared initialization.  This script produces that evidence by building each
arm's model, applying the production ``apply_shared_init`` path and comparing
the resulting states tensor by tensor.

It is deliberately *not* a training run and not a replacement for the
pretraining smoke: it never calls the model, so it consumes zero of the round's
forward / backward / update budget.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from scripts.pretrain_mcl_ph import apply_shared_init, shared_new_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R3_DIR = ROOT / 'results/mcl_ph_20260921/p1/pretrain_r3b'
V2_ARTIFACT = R3_DIR / 'shared_new_init.pt'
V1_ARTIFACT = ROOT / 'results/mcl_ph_20260921/p1/pretrain/shared_new_init.pt'
ARMS = ('cat', 'gate', 'xattn')
CONFIG = {arm: ROOT / 'configs' / 'mts' / f'mcl_ph_{arm}.json' for arm in ARMS}
CAT_RECORD = R3_DIR / 'cat' / 'step_0000.json'
FUSION_PREFIX = 'encoder.fusion.'
ROUTER_LAYER = 'encoder.branch.router.net.0.weight'
ROUTER_BIAS = 'encoder.branch.router.net.0.bias'
GATE_WEIGHT = 'encoder.fusion.gate.weight'
DECLARED = {'router_weight_std': 0.02, 'gate_weight_std': 0.001}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config_diff():
    """The three arm configs must differ in the controlled change only."""
    configs = {arm: json.loads(CONFIG[arm].read_text(encoding='utf-8')) for arm in ARMS}
    keys = sorted(set().union(*[set(value) for value in configs.values()]))
    differing = {key: {arm: configs[arm].get(key) for arm in ARMS}
                 for key in keys if len({json.dumps(configs[arm].get(key))
                                         for arm in ARMS}) > 1}
    return configs, differing


def _arm_state(arm, config):
    """Build the arm exactly as the runner does and apply the shared artifact."""
    from src.modules.mcl_ph_pretrain import MCLPHPretrainer
    model = MCLPHPretrainer(arm, dropout=float(config.get('dropout', 0.1)),
                            cutoffs=tuple(config['cutoffs']),
                            router_dense_updates=int(config['router_dense_updates']),
                            collect_diagnostics=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    applied = apply_shared_init(model, V2_ARTIFACT)
    after = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return model, before, after, applied


def main():
    problems = []
    configs, differing = _config_diff()
    if set(differing) != {'fusion_mode'}:
        problems.append('the three arm configs differ in more than the fusion mode: '
                        f'{sorted(differing)}')
    config = configs[ARMS[0]]

    payload = torch.load(V2_ARTIFACT, map_location='cpu', weights_only=False)
    state = payload['state_dict']
    report = {'artifact': {
        'path': str(V2_ARTIFACT.relative_to(ROOT)), 'sha256': _sha256(V2_ARTIFACT),
        'schema': payload.get('schema'), 'seed': payload.get('seed'),
        'source': payload.get('source'), 'tensor_count': len(state),
        'fusion_tensors_in_artifact':
            sorted(name for name in state if name.startswith(FUSION_PREFIX)),
        'router_tensors_in_artifact':
            sorted(name for name in state if name.startswith('encoder.branch.router.')),
    }, 'config_diff': {'differing_keys': differing,
                       'identical_keys': len(configs[ARMS[0]]) - len(differing)},
       'declared': DECLARED, 'arms': {}}

    if report['artifact']['schema'] != 'mcl-ph-shared-new-init-v2':
        problems.append(f"the artifact schema is {report['artifact']['schema']!r}")
    if report['artifact']['fusion_tensors_in_artifact']:
        problems.append('the artifact carries fusion parameters: they must stay arm-specific')
    if ROUTER_LAYER not in state:
        problems.append(f'{ROUTER_LAYER} is not part of the shared artifact')
    else:
        router_std = float(state[ROUTER_LAYER].float().std())
        router_bias = float(state[ROUTER_BIAS].float().abs().max())
        report['artifact']['router_weight_std'] = router_std
        report['artifact']['router_bias_max_abs'] = router_bias
        if abs(router_std - DECLARED['router_weight_std']) > 0.002:
            problems.append(f'the shared router weight is not the declared Normal(0, 0.02): '
                            f'std={router_std:.6f}')

    states = {}
    for arm in ARMS:
        model, before, after, applied = _arm_state(arm, configs[arm])
        entry = {'shared_initialization': applied,
                 'tensor_count': len(after),
                 'common_tensors': sum(1 for name in after
                                       if name.startswith(('encoder.branch.', 'atom_head.',
                                                           'local_decoder.',
                                                           'nonbond_decoder.'))),
                 'fusion_tensors': sorted(name for name in after
                                          if name.startswith(FUSION_PREFIX)),
                 'router_weight_std_after': float(after[ROUTER_LAYER].float().std()),
                 'router_bias_max_abs_after': float(after[ROUTER_BIAS].float().abs().max()),
                 'fusion_tensors_changed_by_shared_init': sorted(
                     name for name in after if name.startswith(FUSION_PREFIX)
                     and not torch.equal(before[name], after[name])),
                 'shared_tensors_applied_verbatim': sum(
                     1 for name, value in state.items() if torch.equal(after[name], value)),
                 'expected_shared_tensor_count': len(shared_new_state(model)),
                 }
        for name in (GATE_WEIGHT,):
            if name in after:
                entry['gate_weight_std_after'] = float(after[name].float().std())
        if entry['shared_tensors_applied_verbatim'] != len(state):
            problems.append(f'{arm}: the artifact was not applied verbatim to every tensor')
        if entry['expected_shared_tensor_count'] != len(state):
            problems.append(f'{arm}: the common block set is {entry["expected_shared_tensor_count"]} '
                            f'tensors, the artifact carries {len(state)}')
        if entry['fusion_tensors_changed_by_shared_init']:
            problems.append(f'{arm}: the shared initialization overwrote fusion parameters')
        if 'gate_weight_std_after' in entry and \
                abs(entry['gate_weight_std_after'] - DECLARED['gate_weight_std']) > 0.0002:
            problems.append(f'{arm}: the gate weight lost its declared Normal(0, 0.001): '
                            f'std={entry["gate_weight_std_after"]:.6f}')
        if abs(entry['router_weight_std_after'] - DECLARED['router_weight_std']) > 0.002:
            problems.append(f'{arm}: the router weight after applying the artifact has '
                            f'std={entry["router_weight_std_after"]:.6f}')
        report['arms'][arm] = entry
        states[arm] = after

    # Cross-arm: every shared tensor must be identical in all three arms.
    mismatch = []
    for name in state:
        values = [states[arm][name] for arm in ARMS]
        if not all(torch.equal(values[0], value) for value in values[1:]):
            mismatch.append(name)
    report['cross_arm'] = {'compared_tensors': len(state), 'mismatching_tensors': mismatch,
                           'fusion_tensor_sets_differ': len(
                               {tuple(report['arms'][arm]['fusion_tensors']) for arm in ARMS}) > 1}
    if mismatch:
        problems.append(f'{len(mismatch)} shared tensors differ across the three arms')
    if not report['cross_arm']['fusion_tensor_sets_differ']:
        problems.append('the three arms expose the same fusion parameter set')

    # The rejected v1 artifact must stay unusable, and the recorded hash must be
    # the one the cat arm actually ran with.
    if V1_ARTIFACT.is_file():
        v1 = torch.load(V1_ARTIFACT, map_location='cpu', weights_only=False)
        report['v1_artifact'] = {'path': str(V1_ARTIFACT.relative_to(ROOT)),
                                 'sha256': _sha256(V1_ARTIFACT), 'schema': v1.get('schema'),
                                 'distinct_from_v2': _sha256(V1_ARTIFACT) != _sha256(V2_ARTIFACT)}
        # The schema guard raises before any tensor is copied, so a rejected
        # artifact cannot have touched the model built above.
        try:
            apply_shared_init(model, V1_ARTIFACT)
            problems.append('the v1 artifact was accepted: the schema guard is not effective')
        except ValueError as error:
            report['v1_artifact']['rejection'] = str(error)[:200]
        report['v1_artifact']['router_weight_std'] = float(v1['state_dict'][ROUTER_LAYER].float().std())
    else:
        report['v1_artifact'] = {'path': str(V1_ARTIFACT.relative_to(ROOT)), 'present': False}

    if CAT_RECORD.is_file():
        recorded = json.loads(CAT_RECORD.read_text(encoding='utf-8'))['shared_new_initialization']
        report['cat_arm_record'] = {key: recorded[key] for key in
                                    ('sha256', 'schema', 'seed', 'tensor_count')}
        if recorded['sha256'] != report['artifact']['sha256']:
            problems.append('the artifact on disk is not the one the cat arm recorded')
        if recorded['tensor_count'] != len(state):
            problems.append('the recorded shared tensor count differs from the artifact')
    else:
        problems.append(f'{CAT_RECORD.relative_to(ROOT)} is missing: the cat arm record cannot '
                        'tie the artifact to the run')

    payload_out = dict(status='PASS' if not problems else 'FAILED', problems=problems)
    payload_out.update(report)
    print(json.dumps(payload_out, indent=2, sort_keys=True))
    return 0 if not problems else 4


if __name__ == '__main__':
    raise SystemExit(main())
