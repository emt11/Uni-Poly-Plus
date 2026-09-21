"""4-rank gloo DDP check for the MCL-PH objective (0 optimizer updates).

Launched by ``tests/test_mcl_ph_pretrain.py``.  Two distributed forwards and
backwards are performed: one where only some ranks have valid geometry
(``partial-zero``) and one where no rank has any (``all-zero``).  Prints one
JSON verdict line.
"""
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    torch.manual_seed(0)
    dist.init_process_group('gloo', init_method='env://')
    rank, world = dist.get_rank(), dist.get_world_size()
    from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer
    from src.modules.mcl_ph_pretrain import BALANCE_WEIGHT

    report = {'rank': rank, 'world_size': world, 'stages': {}}

    # ---- partial-zero: only even ranks carry a valid geometry graph --------
    valid = (rank % 2 == 0)
    model = torch.nn.parallel.DistributedDataParallel(
        make_pretrainer('gate'), find_unused_parameters=True)
    # The contract fixes this check at update 501, the first Top-2 update: a
    # rejected expert loses its atom-CE gradient but must keep the geometry one.
    assert model.module.encoder.branch.router.routing_mode_for_step(501) == 'top2'
    model.module.set_router_mode('top2', step=501)
    batch = build_batch(readout_valid=(valid, valid), geometry_valid=(valid, valid))
    labels = build_labels(batch, geometric=valid)
    result = model(batch, labels)
    objective = model.module.objective(result, weights=(1.0, 1.0, BALANCE_WEIGHT),
                                       world_size=world)
    if not torch.isfinite(objective):
        raise SystemExit(f'rank {rank}: non-finite objective with partial geometry')
    objective.backward()
    local_atom = float(result['atom_count'].detach())
    local_geo = float(result['geo_count'].detach())
    global_atom = torch.tensor([local_atom])
    global_geo = torch.tensor([local_geo])
    dist.all_reduce(global_atom)
    dist.all_reduce(global_geo)
    balance_gradient = model.module.encoder.branch.router.net[0].weight.grad
    top_k = result['router_top_k']
    graph = {
        'local_atom_count': local_atom,
        'local_geometry_count': local_geo,
        'global_atom_count': float(global_atom),
        'global_geometry_count': float(global_geo),
        'objective_is_finite': bool(torch.isfinite(objective)),
        'balance_empty': bool(result['balance_empty_valid']),
        'router_gradient_norm': (0.0 if balance_gradient is None
                                 else float(balance_gradient.abs().sum())),
        'router_mode': model.module.encoder.branch.router.mode,
        'alpha': [[float(value) for value in row] for row in result['alpha'].detach()],
        'selected_slots': (int(top_k.size(1)) if top_k is not None else 0),
        'expert_gradient_norms': [
            float(sum(float(parameter.grad.abs().sum())
                      for parameter in expert.parameters()
                      if parameter.grad is not None))
            for expert in model.module.encoder.branch.experts],
    }
    if local_atom > 0:
        gradient = model.module.encoder.o8.layers[0].attention.qkv.weight.grad
        graph['o8_gradient_norm'] = 0.0 if gradient is None else float(gradient.abs().sum())
    report['stages']['partial_zero'] = graph

    # ---- all-zero: no rank has a valid geometry graph ----------------------
    del model
    model = torch.nn.parallel.DistributedDataParallel(
        make_pretrainer('gate'), find_unused_parameters=True)
    batch = build_batch(readout_valid=(False, False), geometry_valid=(False, False))
    labels = build_labels(batch, geometric=False)
    result = model(batch, labels)
    objective = model.module.objective(result, weights=(1.0, 1.0, BALANCE_WEIGHT),
                                       world_size=world)
    if not torch.isfinite(objective):
        raise SystemExit(f'rank {rank}: non-finite objective with no geometry at all')
    objective.backward()
    balance_gradient = model.module.encoder.branch.router.net[0].weight.grad
    report['stages']['all_zero'] = {
        'balance_empty': bool(result['balance_empty_valid']),
        'balance_value': float(result['balance'].detach()),
        'objective_is_finite': bool(torch.isfinite(objective)),
        'router_gradient_norm': (0.0 if balance_gradient is None
                                 else float(balance_gradient.abs().sum())),
        'roster_gradient_norm': float(
            model.module.encoder.branch.experts[0].element.weight.grad.abs().sum())
        if model.module.encoder.branch.experts[0].element.weight.grad is not None else None,
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, report)
    if rank == 0:
        print(json.dumps({'status': 'PASS', 'ranks': gathered}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
