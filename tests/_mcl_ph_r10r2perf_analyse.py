"""Analyse the r10R2-perf paired benchmark (6 real 30-update starts).

For each arm it pairs <arm>_online with <arm>_cached, checks that the two runs are
numerically the same at steps 1 and 2 (losses, update denominators, gradient norm,
router mode) and that the sample order is identical, then reports steady-state
(steps 6-30) and full-run (steps 1-30) timing.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'results/mcl_ph_20260921/p2r2_perf_bench'
ARMS = ('cat', 'gate', 'xattn')
STEPS = tuple(range(1, 31))
STEADY = tuple(range(6, 31))
EQUIV_STEPS = (1, 2)


def records(arm, mode):
    """Every rank-0 training record, by step.

    Four ranks write to one stdout pipe, so a line can hold one object followed by
    another with the separator lost; ``raw_decode`` walks the objects in the line and
    a torn fragment simply yields nothing.
    """
    path = BENCH / f'{arm}_{mode}.stdout.log'
    decoder = json.JSONDecoder()
    found = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        index = line.find('{')
        while index != -1:
            try:
                row, end = decoder.raw_decode(line, index)
            except json.JSONDecodeError:
                break
            if isinstance(row, dict) and 'step' in row and int(row.get('rank', -1)) == 0:
                step = int(row['step'])
                assert step not in found or found[step] == row, f'{arm}/{mode} step {step}'
                found[step] = row
            index = line.find('{', end)
    return found


def load_ordered_keys(arm, mode):
    import torch

    path = BENCH / f'{arm}_{mode}' / 'resume_00030.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload


def stats(values):
    values = sorted(float(value) for value in values)
    def quantile(fraction):
        index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
        return values[index]
    return {'median': statistics.median(values), 'p90': quantile(0.90),
            'p99': quantile(0.99), 'max': values[-1], 'min': values[0],
            'mean': statistics.fmean(values), 'n': len(values)}


report = {'arms': {}, 'equivalence': {}}

for arm in ARMS:
    pair = {mode: records(arm, mode) for mode in ('online', 'cached')}
    assert set(pair['online']) == set(STEPS), f'{arm}: online steps {sorted(pair["online"])}'
    assert set(pair['cached']) == set(STEPS), f'{arm}: cached steps {sorted(pair["cached"])}'

    # 1. numerical equivalence at the first two steps.
    equivalence = {}
    for step in EQUIV_STEPS:
        left, right = pair['online'][step], pair['cached'][step]
        equivalence[step] = {
            'losses_equal': left['losses'] == right['losses'],
            'losses': left['losses'],
            'cached_losses': right['losses'],
            'update_denominators_equal': left['update_denominators'] == right['update_denominators'],
            'update_denominators': left['update_denominators'],
            'grad_total_preclip_equal': left['grad_total_preclip'] == right['grad_total_preclip'],
            'grad_total_preclip': left['grad_total_preclip'],
            'cached_grad_total_preclip': right['grad_total_preclip'],
            'router_mode': left['router_mode'],
            'cached_router_mode': right['router_mode'],
            'valid_graphs_equal': left['valid_graphs'] == right['valid_graphs'],
            'target_counts_equal': left['target_counts'] == right['target_counts'],
        }
    orders = {mode: load_ordered_keys(arm, mode) for mode in ('online', 'cached')}
    equivalence['sample_order_equal'] = (orders['online']['ordered_keys']
                                        == orders['cached']['ordered_keys'])
    equivalence['next_position_equal'] = (orders['online']['next_position']
                                          == orders['cached']['next_position'])
    equivalence['sample_count'] = len(orders['online']['ordered_keys'])
    equivalence['cached_identity_mode'] = orders['cached']['identity'].get('trajectory_cache')
    equivalence['online_identity_mode'] = orders['online']['identity'].get('trajectory_cache')
    report['equivalence'][arm] = equivalence

    # 2. timing.
    timing = {}
    for mode in ('online', 'cached'):
        rows = pair[mode]
        step_seconds = [rows[step]['step_seconds'] for step in STEPS]
        timing[mode] = {
            'step_seconds_steady': stats([rows[step]['step_seconds'] for step in STEADY]),
            'step_seconds_full': stats(step_seconds),
            'preparation_seconds_steady': stats([rows[step]['preparation_seconds']
                                                 for step in STEADY]),
            'forward_backward_seconds_steady': stats([rows[step]['forward_backward_seconds']
                                                      for step in STEADY]),
            'training_seconds_full': float(sum(step_seconds)),
            'training_seconds_steady': float(sum(rows[step]['step_seconds'] for step in STEADY)),
            'preparation_seconds_full': float(sum(rows[step]['preparation_seconds']
                                                  for step in STEPS)),
            'forward_backward_seconds_full': float(sum(rows[step]['forward_backward_seconds']
                                                       for step in STEPS)),
        }
    timing['speedup'] = {
        'steady_median_step_seconds': timing['online']['step_seconds_steady']['median']
        / timing['cached']['step_seconds_steady']['median'],
        'steady_median_preparation_seconds': timing['online']['preparation_seconds_steady']['median']
        / timing['cached']['preparation_seconds_steady']['median'],
        'median_forward_backward_seconds': timing['online']['forward_backward_seconds_steady']['median']
        / timing['cached']['forward_backward_seconds_steady']['median'],
        'training_seconds_full': timing['online']['training_seconds_full']
        / timing['cached']['training_seconds_full'],
    }
    timing['tail'] = {
        'step_seconds_p90_ratio': timing['cached']['step_seconds_steady']['p90']
        / timing['online']['step_seconds_steady']['p90'],
        'step_seconds_p99_online': timing['online']['step_seconds_steady']['p99'],
        'step_seconds_p99_cached': timing['cached']['step_seconds_steady']['p99'],
        'step_seconds_max_online': timing['online']['step_seconds_steady']['max'],
        'step_seconds_max_cached': timing['cached']['step_seconds_steady']['max'],
    }
    timing['cached_faster_median_step'] = (timing['cached']['step_seconds_steady']['median']
                                           < timing['online']['step_seconds_steady']['median'])
    report['arms'][arm] = timing

report['all_three_cached_faster'] = all(
    report['arms'][arm]['cached_faster_median_step'] for arm in ARMS)
report['all_equivalent'] = all(
    all(report['equivalence'][arm][step]['losses_equal']
        and report['equivalence'][arm][step]['update_denominators_equal']
        and report['equivalence'][arm][step]['grad_total_preclip_equal']
        for step in EQUIV_STEPS)
    and report['equivalence'][arm]['sample_order_equal'] for arm in ARMS)
print(json.dumps(report, indent=2, default=str))
