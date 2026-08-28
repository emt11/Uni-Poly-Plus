"""Focused unit coverage for the retained MIPS O8 graph branch."""

import torch

from src.dataset.canonical_periodic import build_canonical_periodic_topology
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_glt import build_periodic_line_sample
from src.dataset.lmdb_cache import sample_key_from_smiles
from src.dataset.trimer_mcl import attach_finite_trimer_mcl
from src.modules.mips_local_graph import MIPSLocalGraphEncoder


def _sample(smiles="*CCO*"):
    data = build_canonical_periodic_topology(smiles)
    attach_finite_trimer_mcl(data, smiles, num_candidates=1)
    record = build_periodic_line_sample(sample_key_from_smiles(smiles), data, data)
    data.glt_geometry_valid = bool(record["geometry_valid"])
    for name, values in {
        "token_atom_a": [item["atom_a"] for item in record["tokens"]],
        "token_atom_b": [item["atom_b"] for item in record["tokens"]],
        "token_shift": [item["shift"] for item in record["tokens"]],
        "token_endpoint_z_a": [item["z_a"] for item in record["tokens"]],
        "token_endpoint_z_b": [item["z_b"] for item in record["tokens"]],
        "token_bond_type": [item["bond_type"] for item in record["tokens"]],
        "token_label": [item["label"] for item in record["tokens"]],
        "token_observation_count": [item["observation_count"] for item in record["tokens"]],
    }.items():
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=torch.long))
    data.glt_token_valid = torch.tensor(
        [item["valid"] for item in record["tokens"]], dtype=torch.bool
    )
    data.glt_token_observation_distances = torch.tensor(
        [item["distances"] for item in record["tokens"]], dtype=torch.float32
    )
    for name, values in {
        "relation_source": [item["source"] for item in record["relations"]],
        "relation_target": [item["target"] for item in record["relations"]],
        "relation_center_atom": [item["center_atom"] for item in record["relations"]],
        "relation_multiplicity": [item["multiplicity"] for item in record["relations"]],
        "relation_observation_count": [item["observation_count"] for item in record["relations"]],
    }.items():
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=torch.long))
    for name, values in {
        "relation_valid": [item["valid"] for item in record["relations"]],
        "relation_is_fallback": [item["fallback"] for item in record["relations"]],
    }.items():
        setattr(data, f"glt_{name}", torch.tensor(values, dtype=torch.bool))
    data.glt_relation_observation_angles = torch.tensor(
        [item["angles"] for item in record["relations"]], dtype=torch.float32
    )
    data.mips_md = torch.zeros(200)
    data.mips_md_valid = torch.tensor(False)
    data.y = torch.zeros(1)
    return data


def test_fixed_o8_architecture_and_canonical_pooling():
    batch = mips_trimer_collate([_sample()])
    encoder = MIPSLocalGraphEncoder(use_star_rbf=False).eval()
    graph, nodes = encoder(batch)
    assert encoder.architecture_name == "MIPS-Trimer-SCAGE"
    assert encoder.max_hops == 2
    assert len(encoder.layers) == 6
    assert encoder.use_star_rbf is False
    assert graph.shape == (1, 512)
    assert nodes.shape == (batch.num_nodes, 512)
    assert torch.isfinite(graph).all() and torch.isfinite(nodes).all()


def test_o8_invalid_geometry_keeps_finite_exact_path():
    batch = mips_trimer_collate([_sample()])
    encoder = MIPSLocalGraphEncoder(use_star_rbf=False).eval()
    batch.graph_available.zero_()
    with torch.no_grad():
        graph, nodes = encoder(batch)
    assert torch.equal(graph, torch.zeros_like(graph))
    assert torch.equal(nodes, torch.zeros_like(nodes))
