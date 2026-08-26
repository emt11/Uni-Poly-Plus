import numpy as np
import torch

from src.dataset.periodic_line_glt import derive_relation_source_distance_observations
from src.modules.periodic_line_glt_v2 import JointRadialAngularRelationBiasV2


def test_joint_sbf_control_and_radial_contract():
    ac = JointRadialAngularRelationBiasV2(mode="control")
    ra = JointRadialAngularRelationBiasV2(mode="radial")
    distance = torch.tensor([1.1, 1.7])
    angle = torch.tensor([0.7, 0.7])
    assert torch.equal(
        ac.observation_features(distance, angle),
        ac.observation_features(distance + 0.2, angle),
    )
    assert not torch.equal(
        ra.observation_features(distance, angle),
        ra.observation_features(distance + 0.2, angle),
    )
    assert sum(p.numel() for p in ac.parameters()) == 672
    assert sum(p.numel() for p in ra.parameters()) == 672
    assert torch.count_nonzero(ac.mean_projection.weight) == 0
    assert torch.count_nonzero(ac.variance_projection.weight) == 0


def test_occurrence_pairing_reconstructs_source_slots():
    tokens = {
        "token_atom_a": np.array([0, 1]),
        "token_atom_b": np.array([1, 2]),
        "token_shift": np.array([0, 0]),
        "token_observation_distances": np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], np.float32),
        "token_observation_count": np.array([3, 3]),
        "token_valid": np.array([True, True]),
    }
    relations = {
        "relation_source": np.array([1, 0]),
        "relation_target": np.array([0, 1]),
        "relation_center_atom": np.array([1, 1]),
        "relation_multiplicity": np.array([3, 3]),
        "relation_observation_count": np.array([3, 3]),
        "relation_valid": np.array([True, True]),
        "relation_is_fallback": np.array([False, False]),
    }
    paired = derive_relation_source_distance_observations(tokens, relations)
    np.testing.assert_array_equal(paired["source_distances"][0], tokens["token_observation_distances"][1])
    np.testing.assert_array_equal(paired["source_distances"][1], tokens["token_observation_distances"][0])
    assert paired["observation_valid"].all()
