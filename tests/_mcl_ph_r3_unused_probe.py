"""r3 probe: does the geometry-free fixture really leave the decoders unused?

Plan MCL-PH-20260921-01/r3, phase 1.  The r3 real-DDP check classifies every
parameter block as absent / unused on a rank / zero on both sides / non-zero
compared.  Its ``partial`` configuration gives one rank a fixture with no
geometry at all, so the two geometry decoders are expected to receive no
gradient there -- the case ``find_unused_parameters`` exists for.

This probe answers that one question in a single process, with one forward and
one backward on that exact fixture, before any further distributed run is
spent: it prints, per parameter block, how many parameters have ``None``, an
exactly zero, or a non-zero gradient.  ``GlobalSum`` short-circuits at world
size one, so no process group is needed.

CPU only, zero optimizer steps, one forward and one backward.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

BLOCKS = ('encoder.o8', 'encoder.branch.experts', 'encoder.branch.router',
          'encoder.fusion', 'local_decoder', 'nonbond_decoder', 'atom_head')


def _block_of(name):
    for block in BLOCKS:
        if name == block or name.startswith(block + '.'):
            return block
    return None


def main():
    from test_mcl_ph_modules import build_batch, build_labels, make_pretrainer

    torch.manual_seed(1234)
    model = make_pretrainer('gate')
    model.eval()
    batch = build_batch()
    # The exact fixture of the ``partial`` configuration's second rank.
    labels = build_labels(batch, geometric=False)
    labels['atom_mask'] = torch.tensor([True, False, False], dtype=torch.bool)
    model.zero_grad(set_to_none=True)
    report = model(batch, labels)
    loss = model.objective(report, weights=(1.0, 1.0, 0.0), world_size=1,
                           denominators={'atom': 1.0, 'geometry': 0.0},
                           accumulation=1)
    loss.backward()
    counts = {key: int(report[key]) for key in
              ('atom_count', 'geo_count', 'local_count', 'nonbond_count')}
    blocks = {block: {'parameters': 0, 'none': [], 'zero': 0, 'nonzero': 0}
              for block in BLOCKS}
    for name, parameter in model.named_parameters():
        block = _block_of(name)
        if block is None:
            continue
        entry = blocks[block]
        entry['parameters'] += 1
        if parameter.grad is None:
            entry['none'].append(name)
        elif float(parameter.grad.abs().max()) == 0.0:
            entry['zero'] += 1
        else:
            entry['nonzero'] += 1
    print(json.dumps({
        'checked_out': str(ROOT / 'tests' / '_mcl_ph_r3_unused_probe.py'),
        'forward': 1, 'backward': 1, 'optimizer_steps': 0,
        'fixture': 'geometry=False, atom_mask=(True, False, False)',
        'loss': float(loss.detach()), 'report_counts': counts,
        'blocks': {block: {'parameters': value['parameters'],
                           'none': len(value['none']), 'zero': value['zero'],
                           'nonzero': value['nonzero']}
                   for block, value in blocks.items()},
        'none_examples': {block: value['none'][:3]
                          for block, value in blocks.items() if value['none']},
    }, indent=1), flush=True)


if __name__ == '__main__':
    main()
