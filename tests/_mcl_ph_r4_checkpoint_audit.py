#!/usr/bin/env python3
"""r4 read-only audit of the r3 cat-arm resume checkpoint.

The r3 run left ``resume_00002.pt`` behind when it hung in the export stage.  A
file that exists is not yet a file that loads, so this audit states exactly what
the artifact contains and what it cannot show: every tensor is inspected for
shape and finiteness, the identity block is compared against the run's own
``run.json``, and the runner's resume preconditions are re-checked on CPU.

It performs no forward, no backward and no optimizer step, and it never writes
next to the frozen r3 artifacts.  Passing this audit does **not** establish that
an exact resume would succeed: the sampler position can only be fully checked
against the frozen data source, which this audit deliberately does not open.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAT_DIR = ROOT / 'results/mcl_ph_20260921/p1/pretrain_r3b/cat'
CHECKPOINT = CAT_DIR / 'resume_00002.pt'
RUN_JSON = CAT_DIR / 'run.json'
EXPECTED_STEP = 2
EXPECTED_BATCH_SIZE = 1008
OUTPUT = ROOT / 'logs' / 'mcl_ph_20260921' / 'r4_checkpoint_audit.json'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(tensor):
    return bool(torch.isfinite(tensor.detach().float()).all())


def main():
    problems, report = [], {}
    blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    report['checkpoint'] = {'path': str(CHECKPOINT.relative_to(ROOT)),
                            'bytes': CHECKPOINT.stat().st_size,
                            'sha256': _sha256(CHECKPOINT),
                            'top_level_keys': sorted(blob)}
    for key in ('identity', 'ordered_keys', 'step', 'next_position', 'model',
                'optimizer', 'rng', 'scheduler'):
        if key not in blob:
            problems.append(f'the checkpoint has no {key!r} entry')
    if problems:
        print(json.dumps(dict(status='FAILED', problems=problems), indent=2))
        return 4

    step, position = int(blob['step']), int(blob['next_position'])
    scheduler = blob['scheduler']
    report['position'] = {
        'step': step, 'next_position': position, 'scheduler': scheduler,
        'expected_next_position': EXPECTED_STEP * EXPECTED_BATCH_SIZE,
        'ordered_keys': len(blob['ordered_keys']),
        'ordered_keys_sha256': hashlib.sha256(
            ''.join(blob['ordered_keys']).encode('utf-8')).hexdigest(),
    }
    if step != EXPECTED_STEP:
        problems.append(f"the checkpoint records step {step}, expected {EXPECTED_STEP}")
    if position != step * EXPECTED_BATCH_SIZE:
        problems.append(f'next_position {position} != step * global_batch '
                        f'{step * EXPECTED_BATCH_SIZE}')
    if int(scheduler.get('step', -1)) != step:
        problems.append('the scheduler step disagrees with the checkpoint step')

    # The runner's own resume guard compares the stored identity with the freshly
    # built one; that is checked here against the run record the same process wrote.
    recorded = json.loads(RUN_JSON.read_text(encoding='utf-8'))
    identity = blob['identity']
    report['identity'] = {key: identity[key] for key in
                          ('fusion_mode', 'world_size', 'sample_count', 'cutoffs',
                           'router_dense_updates', 'router_top_k', 'route', 'third_task')
                          if key in identity}
    report['identity']['statistics_sha256'] = identity.get('statistics_sha256')
    report['identity']['shared_new_init_sha256'] = identity.get('shared_new_init_sha256')
    if identity != recorded['identity']:
        problems.append('the checkpoint identity differs from the run record')

    statistics_path = ROOT / 'results' / 'mcl_ph_20260921' / 'p0' / 'statistics.npz'
    report['identity']['statistics_file_sha256'] = _sha256(statistics_path)
    if identity.get('statistics_sha256') != report['identity']['statistics_file_sha256']:
        problems.append('the recorded statistics hash is not the statistics file on disk')
    shared_path = ROOT / 'results' / 'mcl_ph_20260921' / 'p1' / 'pretrain_r3b' / \
        'shared_new_init.pt'
    report['identity']['shared_new_init_file_sha256'] = _sha256(shared_path)
    if identity.get('shared_new_init_sha256') != \
            report['identity']['shared_new_init_file_sha256']:
        problems.append('the recorded shared-initialization hash is not the v2 artifact')

    from src.modules.mcl_ph_pretrain import MCLPHPretrainer
    reference = MCLPHPretrainer(identity['fusion_mode'], dropout=0.1,
                                cutoffs=tuple(identity['cutoffs']),
                                router_dense_updates=int(identity['router_dense_updates']))
    expected = reference.state_dict()
    stored = blob['model']
    report['model'] = {'tensors': len(stored), 'parameters': sum(
        int(value.numel()) for value in stored.values())}
    missing = sorted(set(expected) - set(stored))
    extra = sorted(set(stored) - set(expected))
    if missing or extra:
        problems.append(f'model tensor set mismatch: missing={missing[:3]} extra={extra[:3]}')
    shape_mismatch = [name for name in stored
                      if name in expected and tuple(stored[name].shape) !=
                      tuple(expected[name].shape)]
    if shape_mismatch:
        problems.append(f'model shape mismatch on {shape_mismatch[:3]}')
    non_finite = [name for name, value in stored.items()
                  if value.is_floating_point() and not _finite(value)]
    report['model']['non_finite_tensors'] = non_finite
    if non_finite:
        problems.append(f'{len(non_finite)} model tensors are not finite')

    optimizer = blob['optimizer']
    groups = optimizer['param_groups']
    report['optimizer'] = {'groups': len(groups),
                           'lr': [float(group['lr']) for group in groups],
                           'weight_decay': [float(group['weight_decay']) for group in groups],
                           'state_entries': len(optimizer['state']),
                           'steps': sorted({int(state['step'])
                                            for state in optimizer['state'].values()})}
    if len(groups) != 1:
        problems.append(f"expected one optimizer group, found {len(groups)}")
    non_finite_state = [key for key, state in optimizer['state'].items()
                        for name, value in state.items()
                        if torch.is_tensor(value) and value.is_floating_point()
                        and not _finite(value)]
    report['optimizer']['non_finite_state_entries'] = sorted(set(non_finite_state))
    if non_finite_state:
        problems.append('optimizer state holds non-finite tensors')

    rng = blob['rng']
    report['rng'] = {'entries': len(rng), 'keys_per_entry': sorted(rng[0]),
                     'cuda_devices_per_entry': [len(entry['cuda'] or [])
                                                for entry in rng],
                     'torch_tensor_bytes': int(rng[0]['torch'].numel())}
    if len(rng) != int(identity['world_size']):
        problems.append(f"the RNG block holds {len(rng)} entries for world size "
                        f"{identity['world_size']}")

    payload = dict(status='PASS' if not problems else 'FAILED', problems=problems,
                   note='read-only audit: a loadable checkpoint is not an exact resume',
                   not_verified=['sampler order against the frozen source',
                                 'that a resumed run reaches the same state'],
                   **report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not problems else 4


if __name__ == '__main__':
    raise SystemExit(main())
