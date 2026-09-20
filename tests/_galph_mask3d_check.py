"""Deterministic 3D masking-contract check, run in a clean subprocess.

Verifies on a real fixture that
  * MASK rows carry the learned mask embedding as their line-node input and are
    conditioned on MASK_TYPE by PathAngleBias,
  * REPLACE rows take the donor bond's inputs (a different real line node) while
    keeping the real graph topology,
  * KEEP rows are byte-identical to an unmasked forward,
printed as one JSON verdict line per policy combination.

Every capture uses a named forward hook that returns ``None``: a forward hook
returning a value replaces the module output.
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
from src.modules.glt_dual import triplet_type
from src.modules.glt_galformer_ph import MASK_TYPE, GLTGalPH

KEEP, MASK, REPLACE = 0, 1, 2
DONOR_BOND = 2
SMILES = '*CCOCC*'


def _run(policy_spec):
    """One eval-mode forward with per-row policies; returns captures."""
    topology = build_canonical_periodic_topology(SMILES)
    _, trimer = _toy_pair(SMILES)
    static = build_dual_static(topology, trimer, SMILES)
    data, labels = prepare_galformer_sample(topology, trimer, SMILES, static=static,
                                           seed=42, key='mask3d', position=0)
    bonds = int(data.bond_distance.numel())
    rows = torch.arange(bonds, dtype=torch.long)
    policy = torch.zeros(bonds, dtype=torch.long)
    for index, value in policy_spec.items():
        policy[index] = value
    data.mask3d_rows, data.mask3d_policy = rows, policy
    data.mask3d_donor_atoms = torch.full((1,), DONOR_BOND, dtype=torch.long)
    batch, _ = galformer_collate([(data, labels)])

    torch.manual_seed(0)                    # identical weights across runs
    model = GLTGalPH('mean')
    model.eval()
    captured = {}

    def capture_types(module, inputs, output):
        captured['types'] = inputs[0].detach().clone()
        return None

    def capture_layer_input(module, inputs, output):
        captured['layer_input'] = inputs[0].detach().clone()
        return None

    def capture_distance(module, inputs, output):
        captured['distance'] = inputs[0].detach().clone()
        return None

    handles = [
        model.glt.angle_bias.register_forward_hook(capture_types),
        model.glt.layers[0].register_forward_hook(capture_layer_input),
        model.glt.distance_basis.register_forward_hook(capture_distance),
    ]
    with torch.no_grad():
        out = model(batch)
    for handle in handles:
        handle.remove()
    return {'data': data, 'batch': batch, 'model': model, 'out': out,
            'captured': captured, 'policy': policy}


def _case(name, policy_spec):
    reference = _run({})
    run = _run(policy_spec)
    data, batch, model = run['data'], run['batch'], run['model']
    out, captured, ref = run['out'], run['captured'], reference['captured']
    policy = run['policy']
    mask_rows = torch.where(policy == MASK)[0]
    replace_rows = torch.where(policy == REPLACE)[0]
    keep_rows = torch.where(policy == KEEP)[0]
    real_type = triplet_type(data.bond_z_a.long(), data.bond_z_b.long(),
                            data.bond_type.long())
    checks = {
        'forward_finite': bool(torch.isfinite(out['bond_states']).all()),
        'types_one_per_bond': tuple(captured['types'].shape) == (data.bond_distance.numel(),),
        # The real line graph is never masked: PathAngleBias sees it unchanged.
        'line_topology_untouched': (int(batch.line_path.size(0))
                                    == int(data.line_path.size(0))
                                    and int(batch.line_source.numel())
                                    == int(data.line_source.numel())),
        'states_one_per_bond': int(out['bond_states'].size(0)) == int(data.bond_distance.numel()),
    }
    if mask_rows.numel():
        checks['mask_embedding_exact'] = bool(torch.equal(
            captured['layer_input'][mask_rows],
            model.mask_3d_embedding.expand(mask_rows.numel(), 512)))
        checks['mask_type_in_bias'] = bool(
            (captured['types'][mask_rows] == MASK_TYPE).all())
        checks['mask_input_differs_from_clean'] = bool(not torch.allclose(
            captured['layer_input'][mask_rows],
            ref['layer_input'][mask_rows]))
    if replace_rows.numel():
        checks['replace_takes_donor_input'] = bool(torch.allclose(
            captured['layer_input'][replace_rows],
            ref['layer_input'][DONOR_BOND].expand(replace_rows.numel(), 512),
            atol=1e-6, rtol=1e-6))
        checks['replace_distance_is_donor'] = bool(torch.allclose(
            captured['distance'][replace_rows],
            ref['distance'][DONOR_BOND].expand(replace_rows.numel()),
            atol=1e-6, rtol=1e-6))
        checks['replace_type_is_donor'] = bool(
            (captured['types'][replace_rows] != MASK_TYPE).all()
            and (captured['types'][replace_rows] == real_type[DONOR_BOND]).all())
    if keep_rows.numel():
        checks['keep_input_is_clean'] = bool(torch.equal(
            captured['layer_input'][keep_rows], ref['layer_input'][keep_rows]))
        checks['keep_type_is_real'] = bool(
            (captured['types'][keep_rows] == real_type[keep_rows]).all())
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks}


def main():
    cases = {
        'MASK_only': {0: MASK},
        'REPLACE_only': {0: REPLACE},
        'KEEP_only': {},
        'MASK_REPLACE_KEEP': {0: MASK, 1: REPLACE, 2: KEEP},
    }
    results = {name: _case(name, spec) for name, spec in cases.items()}
    status = 'PASS' if all(item['status'] == 'PASS' for item in results.values()) else 'FAIL'
    print(json.dumps({'status': status, 'cases': results}))


if __name__ == '__main__':
    main()
