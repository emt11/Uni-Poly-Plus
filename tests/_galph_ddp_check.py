"""2-rank gloo DDP check for the GLT-GALPH contrastive path (0 optimizer updates).

Launched by tests/test_glt_galformer_ph.py; prints one JSON verdict line.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    torch.manual_seed(0)
    dist.init_process_group('gloo', init_method='env://')
    rank, world = dist.get_rank(), dist.get_world_size()
    from test_complete_trimer_glt import _toy_pair
    from src.dataset.canonical_periodic import build_canonical_periodic_topology
    from src.dataset.glt_dual_static import build_dual_static
    from src.dataset.glt_galformer_ph import galformer_collate, prepare_galformer_sample
    from src.modules.glt_galformer_ph_pretrain import (GalformerPretrainer,
                                                       galformer_objective)
    smiles = '*CCOCC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    static = build_dual_static(topology, trimer, smiles)
    records = [prepare_galformer_sample(topology, trimer, smiles, static=static,
                                        seed=42, key=f'fixture{rank}', position=rank)]
    batch, labels = galformer_collate(records)
    trainer = GalformerPretrainer('mean', None)
    trainer = torch.nn.parallel.DistributedDataParallel(trainer,
                                                        find_unused_parameters=True)
    result = trainer.module(batch, labels, world_size=world)
    objective = galformer_objective(result, world_size=world)
    objective.backward()
    finite = bool(torch.isfinite(objective).all())
    grads = [value for value in
             (parameter.grad for parameter in trainer.parameters())
             if value is not None]
    grads_finite = all(bool(torch.isfinite(value).all()) for value in grads)
    cl_grad = any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                  for name, parameter in trainer.named_parameters()
                  if name.startswith('module.model.cl_proj'))
    dist.barrier()
    if rank == 0:
        print(json.dumps({'status': 'PASS' if (finite and grads_finite and cl_grad)
                          else 'FAIL', 'loss_finite': finite,
                          'grads_finite': grads_finite, 'cl_grad': cl_grad,
                          'cl_count': int(result['counts'][2]),
                          'optimizer_updates': 0}))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
