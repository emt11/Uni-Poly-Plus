"""Cross-over analysis of the two paired benchmark rounds (r10R2-perf).

Round A (`p2r2_perf_bench`) ran online first, round B (`p2r2_perf_bench_rev`)
ran cached first.  Each arm therefore has one measurement per order, which lets
the cache effect be separated from run order and page-cache warmth.

Reported per arm and order: wall, training_seconds_full (steps 1-30),
training_seconds_steady (steps 6-30), median/p90/p99/max step_seconds and the
preparation_seconds distribution, plus the steps 1/2 equivalence check and the
sample order.

Revised acceptance (replaces the earlier median-step gate):
  cached training_seconds_full   < online training_seconds_full
  cached training_seconds_steady < online training_seconds_steady
  cached steady p90 step_seconds <= online steady p90 step_seconds
  numerical equivalence PASS
`cached_pretraining_path = QUALIFIED` only when all three arms satisfy it in
both orders.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('cat', 'gate', 'xattn')
STEPS = tuple(range(1, 31))
STEADY = tuple(range(6, 31))
EQUIV_STEPS = (1, 2)
DECODER = json.JSONDecoder()


def records(bench, arm, mode):
    """Rank-0 training records by step; a line can hold several JSON objects."""
    path = bench / f'{arm}_{mode}.stdout.log'
    found = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        index = line.find('{')
        while index != -1:
            try:
                row, end = DECODER.raw_decode(line, index)
            except json.JSONDecodeError:
                break
            if isinstance(row, dict) and 'step' in row and int(row.get('rank', -1)) == 0:
                step = int(row['step'])
                assert step not in found or found[step] == row, f'{arm}/{mode} step {step}'
                found[step] = row
            index = line.find('{', end)
    return found


def walls(log_path):
    """arm/mode -> wall seconds, from the driver log."""
    pattern = re.compile(r'RUN arm=(\w+) mode=(\w+) EXIT=(\d+) wall_seconds=([\d.]+)')
    out = {}
    for match in pattern.finditer(log_path.read_text(encoding='utf-8')):
        out[(match.group(1), match.group(2))] = (
            int(match.group(3)), float(match.group(4)))
    return out


def quantile(values, fraction):
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def describe(values):
    return {'median': statistics.median(values), 'p90': quantile(values, 0.90),
            'p99': quantile(values, 0.99), 'max': max(values), 'min': min(values),
            'mean': statistics.fmean(values), 'sum': float(sum(values)), 'n': len(values)}


def order_report(bench, log, mode_of=True):
    """Everything measurable for one round, keyed arm -> mode."""
    wall = walls(log)
    report = {}
    for arm in ARMS:
        pair = {mode: records(bench, arm, mode) for mode in ('online', 'cached')}
        for mode, rows in pair.items():
            assert set(rows) == set(STEPS), f'{bench.name}/{arm}/{mode}: {sorted(rows)}'
        equivalence = {}
        for step in EQUIV_STEPS:
            left, right = pair['online'][step], pair['cached'][step]
            equivalence[step] = {
                'losses_equal': left['losses'] == right['losses'],
                'losses': left['losses'],
                'update_denominators_equal': left['update_denominators']
                == right['update_denominators'],
                'grad_total_preclip_equal': left['grad_total_preclip']
                == right['grad_total_preclip'],
                'grad_total_preclip': left['grad_total_preclip'],
                'router_mode': left['router_mode'],
                'router_mode_equal': left['router_mode'] == right['router_mode'],
                'valid_graphs_equal': left['valid_graphs'] == right['valid_graphs'],
                'target_counts_equal': left['target_counts'] == right['target_counts'],
            }
        equivalence['pass'] = all(
            equivalence[step]['losses_equal']
            and equivalence[step]['update_denominators_equal']
            and equivalence[step]['grad_total_preclip_equal']
            and equivalence[step]['router_mode_equal']
            and equivalence[step]['valid_graphs_equal']
            and equivalence[step]['target_counts_equal'] for step in EQUIV_STEPS)
        timing = {}
        for mode, rows in pair.items():
            timing[mode] = {
                'wall_seconds': wall[(arm, mode)][1],
                'exit_code': wall[(arm, mode)][0],
                'training_seconds_full': float(sum(rows[s]['step_seconds'] for s in STEPS)),
                'training_seconds_steady': float(sum(rows[s]['step_seconds'] for s in STEADY)),
                'step_seconds_full': describe([rows[s]['step_seconds'] for s in STEPS]),
                'step_seconds_steady': describe([rows[s]['step_seconds'] for s in STEADY]),
                'preparation_seconds_full': describe([rows[s]['preparation_seconds']
                                                      for s in STEPS]),
                'preparation_seconds_steady': describe([rows[s]['preparation_seconds']
                                                        for s in STEADY]),
                'forward_backward_seconds_steady': describe(
                    [rows[s]['forward_backward_seconds'] for s in STEADY]),
                'other_seconds_steady': float(sum(
                    rows[s]['step_seconds'] - rows[s]['preparation_seconds']
                    - rows[s]['forward_backward_seconds'] for s in STEADY)),
            }
        report[arm] = {'timing': timing, 'equivalence': equivalence,
                       'faster': {
                           'training_seconds_full': (timing['cached']['training_seconds_full']
                                                     < timing['online']['training_seconds_full']),
                           'training_seconds_steady': (timing['cached']['training_seconds_steady']
                                                       < timing['online']['training_seconds_steady']),
                           'steady_p90_step_seconds': (
                               timing['cached']['step_seconds_steady']['p90']
                               <= timing['online']['step_seconds_steady']['p90']),
                           'wall_seconds': (timing['cached']['wall_seconds']
                                            < timing['online']['wall_seconds']),
                           'median_step_seconds': (
                               timing['cached']['step_seconds_steady']['median']
                               < timing['online']['step_seconds_steady']['median']),
                       }}
    return report


def sample_order(bench, arm):
    import torch

    keys = {}
    for mode in ('online', 'cached'):
        payload = torch.load(bench / f'{arm}_{mode}' / 'resume_00030.pt',
                             map_location='cpu', weights_only=False)
        keys[mode] = (payload['ordered_keys'], payload['next_position'],
                      payload['identity'].get('trajectory_cache', {}).get('mode'))
    return {'equal': keys['online'][0] == keys['cached'][0],
            'next_position_equal': keys['online'][1] == keys['cached'][1],
            'count': len(keys['online'][0]),
            'modes': {'online': keys['online'][2], 'cached': keys['cached'][2]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward', default=str(ROOT / 'results/mcl_ph_20260921/'
                                                 'p2r2_perf_bench'))
    parser.add_argument('--reverse', default=str(ROOT / 'results/mcl_ph_20260921/'
                                                 'p2r2_perf_bench_rev'))
    parser.add_argument('--forward-log',
                        default=str(ROOT / 'logs/mcl_ph_20260921/p2r2perf_bench.log'))
    parser.add_argument('--reverse-log',
                        default=str(ROOT / 'logs/mcl_ph_20260921/p2r2perf_bench_rev.log'))
    args = parser.parse_args()

    report = {'orders': {
        'online_first': order_report(Path(args.forward), Path(args.forward_log)),
        'cached_first': order_report(Path(args.reverse), Path(args.reverse_log)),
    }}
    report['sample_order'] = {
        'online_first': {arm: sample_order(Path(args.forward), arm) for arm in ARMS},
        'cached_first': {arm: sample_order(Path(args.reverse), arm) for arm in ARMS},
    }

    # Cross-order verdict: both orders must support the same direction.
    gate = ('training_seconds_full', 'training_seconds_steady', 'steady_p90_step_seconds')
    verdict = {'gate': list(gate), 'arms': {}, 'qualified': True}
    for arm in ARMS:
        per_order = {name: report['orders'][name][arm]['faster'] for name in report['orders']}
        equivalent = all(report['orders'][name][arm]['equivalence']['pass']
                         for name in report['orders'])
        consistent = {name: all(per_order[name][key] for key in gate) for name in per_order}
        verdict['arms'][arm] = {
            'conditions_per_order': per_order, 'order_holds': consistent,
            'equivalence_pass': equivalent,
            'qualified': bool(consistent['online_first'] and consistent['cached_first']
                              and equivalent),
        }
        verdict['qualified'] = verdict['qualified'] and verdict['arms'][arm]['qualified']
    report['verdict'] = verdict
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__':
    main()
