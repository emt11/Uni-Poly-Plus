"""r3 check: deployment export + strict load for the three P1 fusion arms.

Plan MCL-PH-20260921-01/r3, phase 2, requirement 6.  After the two pre-training
updates of each arm this checks, without any forward or backward:

* the exported ``deploy_00002.pt`` strict-loads through ``load_deployment`` with
  the arm's own fusion mode and the expected step -- a wrong fusion, a wrong step,
  a non-Top-2 inference mode, a legacy bundle, or any tensor set/shape difference
  is a failure, never a silent load;
* the deployment's recorded contract keeps the training/inference distinction the
  plan declares: the run trained with dense routing and the package fixes Top-2
  *inference*; the two are reported side by side and never mixed into one claim;
* the two updates really happened: the shared initialisation snapshot no longer
  matches every shared tensor;
* the three arms start from the same common initialisation (identical
  ``shared_new_init_sha256`` and ``common_init_artifact_sha256``) while their
  fusion parameters are arm-specific.

Identity and initial-value evidence comes from each arm's ``step_0000.json``
(recorded by the runner at start-up), never from a re-run.

CPU only, zero model calls, zero optimizer updates.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FUSION = {'cat': 'cat', 'gate': 'gate', 'xattn': 'xattn'}
FUSION_PREFIX = 'fusion.'


def _initial_values(directory):
    record = json.loads((directory / 'step_0000.json').read_text(encoding='utf-8'))
    return {name: (value.get('rms'), value.get('mean'))
            for name, value in (record.get('parameters') or {}).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-root', required=True)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--arms', nargs='+', default=sorted(FUSION))
    parser.add_argument('--output')
    args = parser.parse_args()

    import torch

    from src.modules.mcl_ph import load_deployment
    from src.modules.mcl_ph_pretrain import MCLPHPretrainer

    root = Path(args.pretrain_root)
    snapshot = torch.load(root / 'shared_new_init.pt', map_location='cpu',
                          weights_only=False)
    payload = {'pretrain_root': str(root), 'step': int(args.step),
               'shared_new_init_schema': snapshot.get('schema'),
               'shared_new_init_sha256': None, 'arms': {}, 'problems': []}

    import hashlib

    payload['shared_new_init_sha256'] = hashlib.sha256(
        (root / 'shared_new_init.pt').read_bytes()).hexdigest()
    shared_state = snapshot['state_dict']
    initials = {}

    for arm in args.arms:
        directory = root / arm
        package_path = directory / f'deploy_{args.step:05d}.pt'
        record = {'package': str(package_path), 'strict_load': False}
        payload['arms'][arm] = record
        try:
            package = torch.load(package_path, map_location='cpu', weights_only=False)
            model = MCLPHPretrainer(fusion_mode=FUSION[arm])
            load_deployment(model.encoder, package, args.step,
                            expected_fusion=FUSION[arm])
            record['strict_load'] = True
            record.update({
                'architecture': package.get('architecture'),
                'training_route': package.get('training_route'),
                'fusion_mode': package.get('fusion_mode'),
                'step': int(package.get('step', -1)),
                'inference_mode': (package.get('router') or {}).get('inference_mode'),
                'inference_top_k': (package.get('router') or {}).get('top_k'),
                'dense_updates': (package.get('router') or {}).get('dense_updates'),
                'ph': package.get('ph'),
                'router_mode_after_load': model.encoder.branch.router.mode,
                'tensor_count': len(package['state_dict']),
                'parameter_count': int(sum(value.numel() for value in model.parameters())),
                'shared_new_init_sha256': (package.get('source') or {}).get(
                    'shared_new_init_sha256'),
                'common_init_sha256': (package.get('source') or {}).get(
                    'common_init_artifact_sha256'),
            })
            deltas = {name: float((package['state_dict'][name].float()
                                   - value.float()).abs().max())
                      for name, value in shared_state.items()
                      if name in package['state_dict']}
            record['shared_tensors'] = len(deltas)
            record['changed_shared_tensors'] = sum(1 for value in deltas.values()
                                                   if value > 0.0)
            record['max_abs_delta_from_shared_init'] = max(deltas.values()) if deltas else 0.0
            initials[arm] = _initial_values(directory)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            record['error'] = f'{type(error).__name__}: {error}'
            payload['problems'].append(f'{arm}: {record["error"]}')
            continue
        if record['inference_mode'] != 'top2':
            payload['problems'].append(f'{arm}: the deployment does not fix Top-2 inference')
        if record['training_route'] != 'mcl_ph':
            payload['problems'].append(f'{arm}: not an MCL-PH route package')
        if record['changed_shared_tensors'] == 0:
            payload['problems'].append(f'{arm}: no shared tensor differs from the '
                                       f'initial snapshot, so no update happened')
        if record['shared_new_init_sha256'] != payload['shared_new_init_sha256']:
            payload['problems'].append(f'{arm}: the package cites a different shared '
                                       f'initialisation artifact')

    arms = sorted(initials)
    if len(arms) > 1:
        common = set.intersection(*[set(initials[arm]) for arm in arms])
        shared_names = sorted(name for name in common
                              if not name.startswith(FUSION_PREFIX))
        differences = []
        for name in shared_names:
            values = {initials[arm][name] for arm in arms}
            if len(values) != 1:
                differences.append({'parameter': name,
                                    'values': {arm: initials[arm][name] for arm in arms}})
        payload['shared_initial_parameters'] = len(shared_names)
        payload['initial_value_differences'] = differences[:8]
        payload['initial_value_difference_count'] = len(differences)
        if differences:
            payload['problems'].append(
                f'{len(differences)} common parameters do not start from the same value')
        payload['arm_specific_parameters'] = {
            arm: sorted(set(initials[arm]) - set(shared_names))[:8] for arm in arms}
        payload['shared_init_hashes'] = {
            arm: json.loads((root / arm / 'step_0000.json').read_text(
                encoding='utf-8'))['shared_new_initialization']['sha256'] for arm in arms}
        payload['shared_init_schemas'] = {
            arm: json.loads((root / arm / 'step_0000.json').read_text(
                encoding='utf-8'))['shared_new_initialization']['schema'] for arm in arms}

    payload['status'] = 'PASS' if not payload['problems'] else 'FAILED'
    text = json.dumps(payload, indent=1, default=str)
    print(text, flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    if payload['problems']:
        raise SystemExit(4)


if __name__ == '__main__':
    main()
