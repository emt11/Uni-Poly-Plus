"""Final geometry failures retain chemistry without fabricated coordinates."""
import pytest
import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.glt_dual import build_dual_sample
from src.dataset.glt_dual_static import build_dual_static
from src.dataset.trimer_mcl import (
    attach_unavailable_trimer_identity, get_etkdg_call_count,
    reset_etkdg_call_counter,
)


@pytest.mark.parametrize('smiles', ['*CC*', '*C/C=C/C*', '*C([2H])C*'])
def test_failed_geometry_retains_valid_atom_identity(smiles):
    reset_etkdg_call_counter()
    topology = build_canonical_periodic_topology(smiles)
    trimer = attach_unavailable_trimer_identity(topology.clone(), smiles, 'TIMEOUT')
    assert trimer.trimer_pos.shape == (0, 3)
    assert not trimer.trimer_geometry_valid
    assert trimer.trimer_atomic_number.numel() > 0
    assert torch.equal(trimer.trimer_atomic_number[trimer.mips_to_trimer_central_index],
                       topology.z)
    static = build_dual_static(topology, trimer, smiles)
    assert not static['geometry_valid']
    sample = build_dual_sample(topology, trimer, smiles, static=static)
    assert not sample.geometry_valid
    assert sample.bond_distance.numel() == 0
    assert torch.equal(sample.mips_x, topology.mips_x)
    assert get_etkdg_call_count() == 0


def test_unexpected_failure_is_rejected():
    topology = build_canonical_periodic_topology('*CC*')
    with pytest.raises(ValueError, match='unsupported final'):
        attach_unavailable_trimer_identity(topology, '*CC*', 'GENERATOR_ERROR')
