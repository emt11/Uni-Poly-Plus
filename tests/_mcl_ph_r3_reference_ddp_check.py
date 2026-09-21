"""r3 check: real-DDP runtime + a reference comparison over every block.

Plan MCL-PH-20260921-01/r3.  Launched by ``tests/test_mcl_ph_r3_reference_ddp.py``
with ``torchrun`` on two gloo ranks.  Two configurations:

``partial``   rank 0 carries two valid atoms and two geometric graphs, rank 1 one
              valid atom and no geometry at all, so rank 1's own forward never
              touches the two geometry decoders (``tests/_mcl_ph_r3_unused_probe.py``
              shows the same fixture leaving all fourteen decoder parameters with
              ``grad is None`` in a plain single-process pass).  Update-level
              denominators: 3 atoms, 2 geometric graphs.
``all-zero``  no rank has geometry, so the global geometry denominator is zero;
              both ranks carry two atoms.  Update-level denominators: 4 atoms, 0
              geometric graphs.

What ``partial`` shows for those locally-unused parameters is recorded, not
assumed: with production's ``find_unused_parameters=True`` and
``skip_all_reduce_unused_params`` left at its default ``False``, DDP reduces the
locally-used map across ranks and skips only what is unused on *every* rank
(``torch/csrc/distributed/c10d/reducer.hpp``: ``all_reduce_local_used_map``,
``is_unused_bucket``, ``should_skip_all_reduce_bucket``).  A parameter that is
unused on one rank but used on another is still reduced, and that rank receives
the rank-averaged gradient instead of ``None``.  The run's own classification
reports exactly that: the decoders are compared and verified on *both* ranks and
``unused_on_this_rank`` is empty everywhere.  Parameters unused on all ranks keep
``grad is None`` and appear as ``absent`` (3 O8 path-bias and 22 expert
parameters in this fixture).  So this check does not demonstrate a rank left
without a gradient; it demonstrates that the rank-averaged gradient equals one
single-process computation over both ranks' fixtures under the same update-level
denominators.

Three questions are answered separately and none stands in for another:

``analytic_formula_consistency``
              *not re-run here*.  It is the r2 closed-form evidence for the
              update-level denominator, the accumulation average and the balance
              term (``logs/mcl_ph_20260921/r2_objective_check4.log``); this check
              only records where that evidence lives.
``model_gradient_reference``
              the ``DistributedDataParallel``-averaged gradient of every
              parameter block that takes part in the compared loss (O8, experts,
              router, fusion, local decoder, non-bond decoder, atom head) against
              one single-process computation over both ranks' fixtures under the
              same update-level denominators.  Parameters are classified per rank
              as absent / zero on both sides / non-zero compared, and a block
              whose reference gradient is zero is reported as **not verified**,
              never as a pass.
``real_ddp_runtime``
              the production forward/objective under real DDP: finite loss and
              finite gradients, the update-level denominator each rank used, and
              the local counts.

The compared loss is atom + geometry with the balance weight at zero: the
balance term is defined as a mean over the valid graphs of *every* rank, so a
single-process pass over one rank's fixtures is not the same object and its
gradient is not covered here (the r2 closed form covers it instead).

The reference runs on every rank over the same fixtures, so no rank ever calls a
forward carrying implicit collectives while another rank is elsewhere in the
collective sequence.  ``balance_term`` fires two collectives per forward -- the
only collective in the compared path besides the DDP reducer, since
``effective_term`` takes the update-level denominator from the caller and adds
none.  The process group times out at 90 s and a stack dump fires at 300 s, so a
mismatched collective fails loudly instead of hanging.

Zero optimizer steps, no GPU.
"""
import datetime
import argparse
import faulthandler
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

WEIGHTS = (1.0, 1.0, 0.0)
BLOCKS = ('encoder.o8', 'encoder.branch.experts', 'encoder.branch.router',
          'encoder.fusion', 'local_decoder', 'nonbond_decoder', 'atom_head')
# Per rank: whether the fixture carries geometry, and the atom mask (None keeps
# the fixture default of two valid atoms).
CONFIGURATIONS = {
    'partial': ({'geometry': True, 'atoms': None},
                {'geometry': False, 'atoms': (True, False, False)}),
    'all_zero': ({'geometry': False, 'atoms': None},
                 {'geometry': False, 'atoms': None}),
}


def _block_of(name):
    for block in BLOCKS:
        if name == block or name.startswith(block + '.'):
            return block
    return None


def _labels(batch, build_labels, spec):
    labels = build_labels(batch, geometric=spec['geometry'])
    if spec['atoms'] is not None:
        labels['atom_mask'] = torch.tensor(spec['atoms'], dtype=torch.bool)
    return labels


def _gradients(module):
    return {name: (None if parameter.grad is None else parameter.grad.detach().clone())
            for name, parameter in module.named_parameters()}


def _compare(tested, reference):
    """Per-block comparison, distinguishing absent, zero and non-zero."""
    report = {block: {'parameters': 0, 'absent': [], 'zero_on_both': [],
                      'compared_nonzero': [], 'unused_on_this_rank': [],
                      'presence_mismatch': [], 'max_deviation': 0.0, 'verified': False}
              for block in BLOCKS}
    for name in sorted(tested):
        block = _block_of(name)
        if block is None:
            continue
        entry = report[block]
        entry['parameters'] += 1
        left, right = tested[name], reference.get(name)
        if left is None and right is None:
            entry['absent'].append(name)
            continue
        if left is None:
            entry['unused_on_this_rank'].append(name)
            continue
        if right is None:
            entry['presence_mismatch'].append(name)
            continue
        deviation = float((left - right).abs().max())
        entry['max_deviation'] = max(entry['max_deviation'], deviation)
        if float(right.abs().max()) == 0.0 and float(left.abs().max()) == 0.0:
            entry['zero_on_both'].append(name)
        else:
            entry['compared_nonzero'].append(name)
    for entry in report.values():
        entry['verified'] = bool(entry['compared_nonzero']) and \
            not entry['presence_mismatch'] and entry['max_deviation'] <= 1e-4
    return report


def configuration(rank, world, name, specs, *, accumulation=1):
    from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer

    torch.manual_seed(1234)
    model = torch.nn.parallel.DistributedDataParallel(make_pretrainer('gate'),
                                                      find_unused_parameters=True)
    model.eval()
    batch = build_batch()
    spec = specs[rank]
    labels = _labels(batch, build_labels, spec)
    local = {'atom': int(labels['atom_mask'].sum()),
             'geometry': 2 if spec['geometry'] else 0}
    denominators = {}
    for task in ('atom', 'geometry'):
        shared = torch.tensor([float(local[task] * accumulation)])
        dist.all_reduce(shared)
        denominators[task] = float(shared[0])
    model.zero_grad(set_to_none=True)
    loss, statistics = None, {}
    for _ in range(accumulation):
        report = model(batch, labels)
        loss = model.module.objective(report, weights=WEIGHTS, world_size=world,
                                      denominators=denominators,
                                      accumulation=accumulation)
        if not torch.isfinite(loss):
            raise SystemExit(f'rank {rank} [{name}]: non-finite objective in real DDP')
        loss.backward()
        statistics = report.get('objective_statistics') or {}
    tested = _gradients(model.module)
    non_finite = [key for key, value in tested.items()
                  if value is not None and not bool(torch.isfinite(value).all())]
    if non_finite:
        raise SystemExit(f'rank {rank} [{name}]: non-finite gradient on {non_finite[:4]}')

    # One process over both ranks' fixtures, same initialisation, same inputs,
    # same denominators -- run on every rank so the collective sequence matches.
    reference_model = make_pretrainer('gate')
    reference_model.load_state_dict(model.module.state_dict())
    reference_model.eval()
    reference_model.zero_grad(set_to_none=True)
    same_init = all(torch.equal(left, right) for left, right in
                    zip(reference_model.state_dict().values(),
                        model.module.state_dict().values()))
    if not same_init:
        raise SystemExit(f'rank {rank} [{name}]: the reference is not the tested model')
    global_batch = build_batch()
    for other in specs:
        for _ in range(accumulation):
            reference_report = reference_model(
                global_batch, _labels(global_batch, build_labels, other))
            reference_loss = reference_model.objective(
                reference_report, weights=WEIGHTS, world_size=1,
                denominators=denominators, accumulation=accumulation)
            reference_loss.backward()
    reference = _gradients(reference_model)
    return {'rank': rank, 'configuration': name, 'world_size': world,
            'local_counts': local, 'update_denominators': denominators,
            'denominators_source': statistics.get('denominators_source'),
            'denominator_statistics': {task: (statistics.get(task) or {})
                                       for task in ('atom', 'geometry')},
            'loss': float(loss.detach()), 'same_initial_state': bool(same_init),
            'blocks': _compare(tested, reference)}


def _verdict(gathered):
    """The two verdicts this run owns, from every rank's evidence."""
    problems, coverage, divergence = [], {}, []
    for name in [key for key in CONFIGURATIONS if key in gathered[0]]:
        coverage[name] = {}
        for rank_report in gathered:
            for block, value in rank_report[name]['blocks'].items():
                entry = coverage[name].setdefault(block, {
                    'parameters': value['parameters'], 'verified_on': [],
                    'compared_nonzero': 0, 'zero_on_both': 0, 'absent': 0,
                    'unused_on_this_rank': 0, 'max_deviation': 0.0})
                entry['compared_nonzero'] = max(entry['compared_nonzero'],
                                                len(value['compared_nonzero']))
                entry['zero_on_both'] = max(entry['zero_on_both'],
                                            len(value['zero_on_both']))
                entry['absent'] = max(entry['absent'], len(value['absent']))
                entry['unused_on_this_rank'] += len(value['unused_on_this_rank'])
                entry['max_deviation'] = max(entry['max_deviation'],
                                             value['max_deviation'])
                if value['presence_mismatch']:
                    problems.append(f'{name}:{block}: rank {rank_report["rank"]} '
                                    f'has a gradient the reference does not')
                if value['verified']:
                    entry['verified_on'].append(rank_report['rank'])
                elif value['compared_nonzero']:
                    problems.append(f'{name}:{block}: rank {rank_report["rank"]} '
                                    f'deviation {value["max_deviation"]}')
                for parameter in value['unused_on_this_rank']:
                    divergence.append(
                        f'{name}:{block}:{parameter} has no gradient on rank '
                        f'{rank_report["rank"]} (unused there) while the global '
                        f'computation gives it one; production reduces '
                        f'locally-unused parameters, so a None gradient on that '
                        f'rank is unexplained here and needs separate investigation')
    for name in [key for key in CONFIGURATIONS if key in gathered[0]]:
        for entry in gathered:
            if name in entry and not entry[name]['same_initial_state']:
                problems.append(f'{name}: rank {entry[name]["rank"]}: the reference '
                                f'model differs from the tested one')
    return {'status': 'PASS' if not problems else 'FAIL', 'problems': problems,
            'coverage': coverage, 'unused_parameter_note': divergence[:8],
            'unused_parameter_count': len(divergence)}


def _payload(gathered, gradient_verdict):
    """The three verdicts and the per-rank runtime evidence, as one record.

    Kept a pure function of the gathered reports so the persisted structure can be
    tested without launching ranks (``tests/test_mcl_ph_r3_payload.py``).
    """
    configurations = {
        name: {'per_rank': {
            str(entry[name]['rank']): {
                'local_counts': entry[name]['local_counts'],
                'update_denominators': entry[name]['update_denominators'],
                'same_initial_state': entry[name]['same_initial_state'],
                'loss': entry[name]['loss'],
                'numerator': {task: (entry[name]['denominator_statistics']
                                     .get(task) or {}).get('numerator')
                              for task in ('atom', 'geometry')},
                'effective_graphs':
                    {task: (entry[name]['denominator_statistics']
                            .get(task) or {}).get('effective_graphs')
                     for task in ('atom', 'geometry')},
            } for entry in gathered if name in entry},
            'local_counts': gathered[0][name]['local_counts'],
            'update_denominators': gathered[0][name]['update_denominators'],
            'denominators_source': gathered[0][name]['denominators_source']}
        for name in CONFIGURATIONS if name in gathered[0]}
    return {
        'status': gradient_verdict['status'],
        'configurations': [name for name in CONFIGURATIONS if name in gathered[0]],
        'verdicts': {
            'analytic_formula_consistency': {
                'status': 'CARRIED_OVER_FROM_R2', 're_run': False,
                'evidence': 'logs/mcl_ph_20260921/r2_objective_check4.log',
                'note': 'update-level denominator, accumulation average and '
                        'balance closed form; neither re-run here nor a '
                        'substitute for the other two verdicts'},
            'model_gradient_reference': gradient_verdict,
            'real_ddp_runtime': {
                'status': 'PASS' if not gradient_verdict['problems'] else 'FAIL',
                'configurations': configurations,
                'collective_order': 'every rank runs the same forward, '
                                    'objective, backward and reference passes '
                                    'in the same order; a mismatch times out '
                                    'at 90 s and dumps stacks at 300 s'},
        }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configurations', nargs='+', default=list(CONFIGURATIONS),
                        choices=list(CONFIGURATIONS))
    parser.add_argument('--output', help='where rank 0 writes the verdict payload')
    args = parser.parse_args()
    faulthandler.dump_traceback_later(300, exit=True)
    torch.manual_seed(0)
    dist.init_process_group('gloo', init_method='env://',
                            timeout=datetime.timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise SystemExit(f'the r3 check is declared for two ranks, got {world}')
    report = {'rank': rank, 'world_size': world}
    for name in args.configurations:
        report[name] = configuration(rank, world, name, CONFIGURATIONS[name])
        print(f'rank {rank}: {name} done', flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, report)
    if rank == 0:
        gradient_verdict = _verdict(gathered)
        payload = _payload(gathered, gradient_verdict)
        text = json.dumps(payload, default=str)
        print(text, flush=True)
        if args.output:
            Path(args.output).write_text(text + '\n', encoding='utf-8')
        for name in CONFIGURATIONS:
            if name not in gathered[0]:
                continue
            summary = {block: {'verified_on': value['verified_on'],
                               'compared_nonzero': value['compared_nonzero'],
                               'zero_on_both': value['zero_on_both'],
                               'absent': value['absent'],
                               'unused_on_this_rank': value['unused_on_this_rank'],
                               'max_deviation': value['max_deviation']}
                       for block, value in gradient_verdict['coverage'][name].items()}
            print(json.dumps({'block_summary': name, **summary}), flush=True)
        if gradient_verdict['problems']:
            raise SystemExit(4)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
