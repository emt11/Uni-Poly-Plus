"""CPU objective-consistency check for plan MCL-PH-20260921-01/r2.

Launched by ``tests/test_mcl_ph_objective_math.py`` with ``torchrun``.  No GPU,
no optimizer step and no training trajectory: the whole run is a verification of
the declared objective arithmetic under distribution.

Three stages:

``synthetic``  the production ``MCLPHPretrainer.objective`` is driven with
               analytic per-rank/microstep reports whose global value and
               gradient are known in closed form.  The effective counts differ
               per rank *and* per microstep and the accumulation is 3, so the
               update-level denominator, the ``world / accumulation`` scale of
               the balance term and the empty-denominator zero are each checked
               against the hand-computed number.
``balance``    the production ``balance_term`` (one collective) is compared with
               a closed-form gradient over the gathered global probability set,
               microstep by microstep, including the accumulation average.
``model``      the real pre-trainer on a tiny fixture with two ranks whose
               effective counts differ (one rank has no geometry, and a smaller
               atom mask), one optimizer update of two microsteps; the
               DDP-averaged gradient must equal the gradient of the same
               objective evaluated as a single global computation.

Prints one JSON verdict line on rank 0.
"""
import datetime
import faulthandler
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

BALANCE_WEIGHT = 1e-3
WEIGHTS = (1.0, 1.0, BALANCE_WEIGHT)
ACCUMULATION = 3

# Per rank, per microstep.  Rank 1 never carries geometry; rank 3's last
# microstep carries no atom target at all.  Every row has a different total,
# which is exactly the case a local (per-rank or per-microstep) denominator gets
# wrong.
ATOM_BASE = ((1.0, 2.0, 3.0), (0.5, 0.25, 0.75), (2.0, 1.0, 1.0), (1.5, 0.0, 0.0))
ATOM_COUNT = ((3, 2, 5), (1, 4, 2), (6, 1, 1), (2, 3, 0))
GEO_BASE = ((0.2, 0.4, 0.1), (0.0, 0.0, 0.0), (0.6, 0.3, 0.3), (0.1, 0.2, 0.0))
GEO_COUNT = ((2, 3, 1), (0, 0, 0), (4, 2, 3), (1, 1, 0))
BALANCE_BASE = ((0.5, -0.25, 1.0), (-0.5, 0.5, 0.25), (0.75, 0.0, -0.75), (0.1, 0.2, 0.3))


def _total(table):
    return float(sum(sum(row) for row in table))


def synthetic_stage(rank, world, *, geometry_enabled):
    """Model-free: the production objective against a closed-form gradient.

    The emulation of DDP is explicit: the local loss is built with the
    production ``objective`` and the update-level denominators, its gradient is
    summed over ranks and divided by the world size, which is exactly what
    ``DistributedDataParallel`` does to the gradients it averages.
    """
    from src.modules.mcl_ph import global_sum
    from src.modules.mcl_ph_pretrain import MCLPHPretrainer

    world = int(world)
    model = MCLPHPretrainer('gate')
    atom_leaf = torch.tensor(1.0, requires_grad=True)
    geometry_leaf = torch.tensor(2.0, requires_grad=True)
    balance_leaf = torch.tensor(-1.5, requires_grad=True)
    atom_counts = list(ATOM_COUNT[rank]) if rank < len(ATOM_COUNT) else [0, 0, 0]
    geometry_counts = (list(GEO_COUNT[rank]) if rank < len(GEO_COUNT) else [0, 0, 0]) \
        if geometry_enabled else [0, 0, 0]
    update_atom = sum(atom_counts)
    update_geometry = sum(geometry_counts)
    denominators = {
        'atom': float(global_sum(torch.tensor([float(update_atom)])).detach()),
        'geometry': float(global_sum(torch.tensor([float(update_geometry)])).detach()),
    }
    global_atom = _total(ATOM_COUNT)
    global_geometry = _total(GEO_COUNT) if geometry_enabled else 0.0
    if denominators['atom'] != global_atom or denominators['geometry'] != global_geometry:
        raise SystemExit(f'rank {rank}: update-level denominator is not the global count')

    statistics, values = [], []
    for step in range(ACCUMULATION):
        report = {
            # The numerator of an empty selection is a zero that keeps its
            # grad_fn, exactly as ``report['geo_sum']`` is on a rank whose valid
            # set is empty -- a zero *value* with a live backward path.
            'atom_sum': atom_leaf * (float(ATOM_BASE[rank][step]) if atom_counts[step] else 0.0),
            'atom_count': torch.tensor(float(atom_counts[step])),
            'geo_sum': geometry_leaf * (float(GEO_BASE[rank][step])
                                        if geometry_counts[step] else 0.0),
            'geo_count': torch.tensor(float(geometry_counts[step])),
            'balance': balance_leaf * float(BALANCE_BASE[rank][step]),
            'balance_empty_valid': bool(sum(geometry_counts) == 0),
        }
        loss = model.objective(report, weights=WEIGHTS, world_size=world,
                               denominators=denominators, accumulation=ACCUMULATION)
        if not torch.isfinite(loss):
            raise SystemExit(f'rank {rank}: non-finite objective at microstep {step}')
        loss.backward()
        statistics.append(report['objective_statistics'])
        values.append(float(loss.detach()))
    del model

    gradient = torch.tensor([float(atom_leaf.grad), float(geometry_leaf.grad),
                             float(balance_leaf.grad)])
    dist.all_reduce(gradient)
    gradient = gradient / float(world)
    expected = [
        _total(ATOM_BASE) / global_atom,
        (_total(GEO_BASE) / global_geometry) if global_geometry else 0.0,
        BALANCE_WEIGHT * _total(BALANCE_BASE) / float(ACCUMULATION),
    ]
    # What the pre-r2 formula would produce for this rank: its own count and an
    # un-averaged balance sum over the accumulation.
    legacy = sum(ATOM_BASE[rank][step] if atom_counts[step] else 0.0
                 for step in range(ACCUMULATION)) / max(1, sum(atom_counts))
    legacy_balance = BALANCE_WEIGHT * sum(BALANCE_BASE[rank])
    deviation = max(abs(float(gradient[index]) - expected[index]) for index in range(3))
    if deviation > 1e-5:
        raise SystemExit(f'rank {rank}: the objective gradient deviates by {deviation}')
    if abs(legacy - expected[0]) < 1e-6:
        raise SystemExit(f'rank {rank}: the local denominator is not distinguishable here')
    return {
        'rank': rank, 'world_size': world, 'denominators': denominators,
        'global_update_counts': {'atom': global_atom, 'geometry': global_geometry},
        'loss_values': values, 'gradient': [float(value) for value in gradient],
        'expected': expected, 'max_deviation': deviation,
        'legacy_local_denominator_gradient': legacy,
        'legacy_balance_gradient': legacy_balance,
        'statistics': statistics,
    }


def balance_stage(rank, world, *, accumulation=3):
    """The collective balance term against a closed-form global gradient.

    ``L_bal = 3 sum_k (mean_k)^2 - 1`` with ``mean_k`` over the globally valid
    set has ``dL_bal/dp_bk = 6 mean_k / N`` on the valid entries, so the emulated
    DDP average of the production local gradients has a closed form to match --
    the collective is not compared against itself.
    """
    from src.modules.mcl_ph import balance_term

    world = int(world)
    generator = torch.Generator().manual_seed(600 + rank)
    rows = 2 + rank
    # The leaf is the probability row itself, so the closed form is a plain
    # derivative of a linear function and no softmax Jacobian is involved.
    probabilities = [torch.softmax(torch.randn((rows, 3), generator=generator), dim=-1)
                     .requires_grad_(True) for _ in range(accumulation)]
    valid = []
    for step in range(accumulation):
        mask = torch.zeros(rows, dtype=torch.bool)
        mask[:1 + (rank + step) % rows] = True
        valid.append(mask)
    values, local = [], []
    for step in range(accumulation):
        term, empty = balance_term(probabilities[step], valid[step])
        if empty:
            raise SystemExit(f'rank {rank}: an unexpected empty valid set at step {step}')
        (term * (float(world) / float(accumulation))).backward()
        values.append(float(term.detach()))
        local.append(probabilities[step].grad.clone())
    # ``world`` is the factor the production objective applies so that DDP's
    # gradient averaging yields the global gradient of a *shared* parameter; for
    # this rank-local leaf the same statement is ``local / world == dL/dp_rank``.

    gathered = [None] * world
    dist.all_gather_object(
        gathered, [(probabilities[step].detach(), valid[step])
                   for step in range(accumulation)])
    deviation = 0.0
    for step in range(accumulation):
        pieces = [gathered[source][step][0] for source in range(world)]
        masks = [gathered[source][step][1] for source in range(world)]
        probability = torch.cat(pieces, dim=0)
        mask = torch.cat(masks, dim=0)
        mean = (probability * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp_min(1)
        row = ((6.0 * mean / mask.sum().clamp_min(1)) / float(accumulation)).unsqueeze(0)
        # A masked-out row is not part of the valid set: its gradient is exactly
        # zero, which is what the mask multiplication in the term produces.
        start = sum(piece.shape[0] for piece in pieces[:rank])
        own = mask[start:start + probabilities[step].shape[0]]
        expected = torch.where(own.unsqueeze(-1), row.expand(own.numel(), 3),
                               torch.zeros(1, 3))
        deviation = max(deviation, float(
            (local[step] / float(world) - expected).abs().max().detach()))
    if deviation > 1e-6:
        raise SystemExit(f'rank {rank}: balance gradient deviates by {deviation}')
    return {'rank': rank, 'accumulation': accumulation, 'term_values': values,
            'max_deviation': deviation}


def model_stage(rank, world, accumulation=1):
    """The real pre-trainer: the rank-averaged gradient versus one global pass.

    Two ranks, two microsteps and one update: rank 0 carries two valid atoms and
    two geometric graphs per microstep, rank 1 one valid atom and no geometry at
    all.  No rank's local count equals the update-level global count (3 atoms, 2
    geometries), so a per-rank or per-microstep denominator cannot reproduce the
    reference.  The accumulation is one here to keep the round's CPU budget: the
    accumulation arithmetic itself is covered by the ``synthetic`` stage, which
    drives the production objective at accumulation three.

    ``DistributedDataParallel`` averages the gradient of a shared parameter over
    the ranks, and that average is emulated here with one explicit
    ``all_reduce``: real DDP with a rank-dependent parameter-use order is not a
    legal configuration, because the collective order has to match on every
    rank.

    The balance weight is zero in this stage on purpose.  The balance term is a
    mean over the *globally* valid graphs, so a single-process pass over one
    rank's graphs is not the same mathematical object; the term has its own
    stage (``balance``) against the closed-form global gradient.
    """
    from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer

    world = int(world)
    weights = (WEIGHTS[0], WEIGHTS[1], 0.0)
    model = make_pretrainer('gate')
    # The bond-path encoder carries dropout, so two forwards of the same batch
    # differ in training mode.  The comparison is of arithmetic, not of noise.
    model.eval()
    batch = build_batch()
    labels = build_labels(batch)
    local = {'atom': 2, 'geometry': 2}
    if rank == 1:
        labels = reduced_labels(batch, build_labels)
        local = {'atom': 1, 'geometry': 0}
    elif rank >= 2:
        # The idle ranks carry no target at all.  They still run the same
        # forward (every rank fires the balance collectives) but contribute a
        # zero numerator, so the reference over the two label sets is exact.
        labels = empty_labels(batch, build_labels)
        local = {'atom': 0, 'geometry': 0}
    # One collective per task in a fixed order: the shared value is the exact
    # update-level global count over every rank and microstep.
    denominators = {}
    for name in ('atom', 'geometry'):
        shared = torch.tensor([float(local[name] * accumulation)])
        dist.all_reduce(shared)
        denominators[name] = float(shared[0])
    declared = {'atom': 3.0 * accumulation, 'geometry': 2.0 * accumulation}
    for name, value in declared.items():
        if abs(denominators[name] - value) > 1e-6:
            raise SystemExit(f'rank {rank}: the {name} denominator is '
                             f'{denominators[name]}, not the declared {value}')
    losses, numerators = [], []
    for _ in range(accumulation):
        report = model(batch, labels)
        loss = model.objective(report, weights=weights, world_size=world,
                               denominators=denominators, accumulation=accumulation)
        loss.backward()
        losses.append(float(loss.detach()))
        numerators.append({name: float(report[name].detach())
                           for name in ('atom_sum', 'atom_count', 'geo_sum', 'geo_count')})
    gradient = torch.cat([parameter.grad.reshape(-1) for parameter in
                          model.atom_head.parameters()]).clone()
    dist.all_reduce(gradient)
    gradient = gradient / float(world)

    reference = None
    if rank == 0:
        # One single process computes the same update over the global batch: the
        # four rank-microsteps, under the shared update-level denominators.
        reference_model = make_pretrainer('gate')
        reference_model.load_state_dict(model.state_dict())
        reference_model.eval()
        reference_model.zero_grad(set_to_none=True)
        global_batch = build_batch()
        for pass_labels in (build_labels(global_batch),
                            reduced_labels(global_batch, build_labels)):
            for _ in range(accumulation):
                report = reference_model(global_batch, pass_labels)
                reference_loss = reference_model.objective(
                    report, weights=weights, world_size=1, denominators=denominators,
                    accumulation=accumulation)
                reference_loss.backward()
        reference = torch.cat([parameter.grad.reshape(-1) for parameter in
                               reference_model.atom_head.parameters()]).clone()
    else:
        # Every forward fires the two ``balance_term`` collectives, so the ranks
        # that are not computing the reference still have to take part or the
        # reference deadlocks.  They contribute zeros, which leaves rank 0's
        # balance value exactly the single-process one.
        _shadow_balance_collectives(2 * accumulation)
    deviation = (float((gradient - reference).abs().max())
                 if reference is not None else None)
    if reference is not None:
        scale = max(1.0, float(reference.norm()))
        if deviation > 1e-4 * scale:
            raise SystemExit(f'rank 0: the rank-averaged gradient deviates by {deviation}')
    return {'rank': rank, 'local_counts': local, 'update_denominators': denominators,
            'expected_update_counts': declared, 'accumulation': accumulation,
            'model_forwards': accumulation + (2 * accumulation if rank == 0 else 0),
            'model_backwards': accumulation + (2 * accumulation if rank == 0 else 0),
            'loss_values': losses, 'numerators': numerators,
            'gradient_norm': float(gradient.norm()),
            'reference_norm': float(reference.norm()) if reference is not None else None,
            'max_deviation': deviation}


def reduced_labels(batch, build_labels):
    """The fixture of rank 1: no geometry and a single masked canonical atom."""
    labels = build_labels(batch, geometric=False)
    labels['atom_mask'] = torch.tensor([True, False, False], dtype=torch.bool)
    return labels


def empty_labels(batch, build_labels):
    """The fixture of the idle ranks: no atom target and no geometry at all."""
    labels = build_labels(batch, geometric=False)
    labels['atom_mask'] = torch.tensor([False, False, False], dtype=torch.bool)
    return labels


def _shadow_balance_collectives(passes):
    """Mirror the two collectives one production forward fires in ``balance_term``.

    The single-process reference still runs the real model, whose balance term
    all-reduces the local probability sum ([3]) and the valid count ([1]) once
    per forward.  Every other rank joins with zeros so the reference sees the
    arithmetic of one process, and no rank is left waiting in a collective.
    """
    for _ in range(int(passes)):
        dist.all_reduce(torch.zeros(3))
        dist.all_reduce(torch.zeros(1))


def main():
    # A mismatched collective must fail loudly instead of burning the round's
    # wall clock: the process-group timeout covers the collectives and this dump
    # covers everything else.
    faulthandler.dump_traceback_later(240, exit=True)
    torch.manual_seed(0)
    dist.init_process_group('gloo', init_method='env://',
                            timeout=datetime.timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    report = {'rank': rank, 'world_size': world}
    report['synthetic_partial'] = synthetic_stage(rank, world, geometry_enabled=True)
    print(f'rank {rank}: synthetic_partial done', flush=True)
    report['synthetic_empty_geometry'] = synthetic_stage(rank, world, geometry_enabled=False)
    print(f'rank {rank}: synthetic_empty_geometry done', flush=True)
    report['balance'] = balance_stage(rank, world)
    print(f'rank {rank}: balance done', flush=True)
    report['model_partial'] = model_stage(rank, world)
    print(f'rank {rank}: model_partial done', flush=True)
    gathered = [None] * world
    dist.all_gather_object(gathered, report)
    if rank == 0:
        print(json.dumps({'status': 'PASS', 'ranks': gathered}, default=str), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
