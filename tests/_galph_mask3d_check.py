"""Deterministic 3D masking-contract check, run in a clean subprocess.

Verifies on a real fixture that the MASK line node carries the learned mask
embedding and that PathAngleBias conditioning sees MASK_TYPE, while REPLACE
takes donor inputs and keeps the real graph.  Prints one JSON verdict line.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import torch

from test_complete_trimer_glt import _toy_pair
from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.glt_galformer_ph import galformer_collate, prepare_galformer_sample
from src.modules.glt_galformer_ph import MASK_TYPE, GLTGalPH


def main():
    smiles = '*CCOCC*'
    topology = build_canonical_periodic_topology(smiles)
    _, trimer = _toy_pair(smiles)
    static = build_dual_static(topology, trimer, smiles)
    data, labels = prepare_galformer_sample(topology, trimer, smiles, static=static,
                                           seed=42, key='mask3d', position=0)
    bonds = int(data.bond_distance.numel())
    rows = torch.arange(bonds, dtype=torch.long)
    policy = torch.zeros(bonds, dtype=torch.long)
    policy[0], policy[1] = 1, 2
    data.mask3d_rows, data.mask3d_policy = rows, policy
    data.mask3d_donor_atoms = torch.full((1,), 2, dtype=torch.long)
    batch, labels = galformer_collate([(data, labels)])
    model = GLTGalPH('mean')
    captured = {}
    handle = model.glt.angle_bias.register_forward_hook(
        lambda m, inputs, output: captured.setdefault('types', inputs[0].detach()))
    out = model(batch)
    handle.remove()
    mask_rows = rows[policy == 1]
    replace_rows = rows[policy == 2]
    checked = {
        'mask_embedding_exact': bool(torch.allclose(
            out['bond_states'][mask_rows],
            model.mask_3d_embedding.expand(mask_rows.numel(), 512))),
        'mask_type_in_bias': bool((captured['types'][mask_rows] == MASK_TYPE).all()),
        'replace_not_mask_type': bool((captured['types'][replace_rows] != MASK_TYPE).all()),
        'donor_differs': bool((labels['label_3d'][replace_rows]
                               != labels['label_3d'][data.mask3d_donor_atoms]).any()),
        'line_topology_untouched': int(batch.line_path.size(0)) == int(data.line_path.size(0)),
        'states_finite': bool(torch.isfinite(out['bond_states']).all()),
    }
    print(json.dumps({'status': 'PASS' if all(checked.values()) else 'FAIL',
                      'checks': checked}))


if __name__ == '__main__':
    main()
