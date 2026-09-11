"""Focused checks for the independent complete-Trimer GLT input contract."""

import copy
import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.dataset.graph_data import build_periodic_multimer_mol
from src.dataset.dataloader import mips_trimer_collate
from src.dataset.periodic_line_glt_complete import (
    BOND_FEATURE_DIM,
    CompleteTrimerGLTSidecar,
    SIDECAR_SCHEMA,
    STEREO_UNKNOWN,
    bond_feature_vector,
    bond_stereo_index,
    build_complete_trimer_glt_sample,
    write_complete_trimer_sidecar,
)
from src.modules.periodic_line_glt_v3 import (
    CompleteTrimerGLTEncoder,
    CompleteTrimerGLTFusionRegressor,
)


def _toy_pair(smiles="*CC*"):
    molecule, metadata = build_periodic_multimer_mol(
        smiles, num_repeat_units=3, close_periodic=False
    )
    base_count = int(metadata["base_atom_count"])
    positions = torch.zeros((molecule.GetNumAtoms(), 3), dtype=torch.float32)
    for unit, atom_indices in enumerate(metadata["unit_atoms"]):
        for base_id, local in enumerate(atom_indices):
            positions[int(local)] = torch.tensor(
                [unit * 3.0 + base_id * 1.4, base_id * 0.2, 0.1 * unit]
            )
    edge_sources, edge_targets, edge_codes = [], [], []
    code = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    for bond in molecule.GetBonds():
        left, right = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        value = code.get(bond.GetBondType(), 5)
        edge_sources.extend([left, right])
        edge_targets.extend([right, left])
        edge_codes.extend([value, value])
    topology = Data(
        canonical_to_trimer_base_atom_id=torch.arange(base_count),
        atomic_numbers=torch.tensor([
            molecule.GetAtomWithIdx(int(i)).GetAtomicNum()
            for i in metadata["unit_atoms"][0]
        ]),
    )
    trimer = Data(
        trimer_pos=positions,
        trimer_atomic_number=torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long
        ),
        trimer_edge_index=torch.tensor([edge_sources, edge_targets], dtype=torch.long),
        trimer_bond_type=torch.tensor(edge_codes, dtype=torch.long),
        trimer_base_ru_atom_id=torch.arange(base_count).repeat(3),
        trimer_ru_offset=torch.repeat_interleave(
            torch.tensor([-1, 0, 1], dtype=torch.long), base_count
        ),
        trimer_geometry_valid=True,
        trimer_geometry_is_3d=True,
        trimer_2d_fallback=False,
    )
    trimer.mips_to_trimer_central_index = torch.tensor(
        metadata["unit_atoms"][1], dtype=torch.long
    )
    return topology, trimer


def _attach_complete(item, row):
    item = copy.copy(item)
    item.glt3_geometry_valid = bool(row["geometry_valid"])
    for name, value in row["tokens"].items():
        dtype = (
            torch.float32 if name in {"token_distance", "token_bond_features"}
            else torch.bool if name in {"token_valid", "token_center_internal"}
            else torch.long
        )
        setattr(item, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
    for name, value in row["relations"].items():
        dtype = torch.float32 if name == "relation_angle" else (
            torch.bool if name == "relation_valid" else torch.long
        )
        setattr(item, f"glt3_{name}", torch.as_tensor(np.array(value, copy=True), dtype=dtype))
    return item


def test_complete_row_contains_all_physical_bonds_and_features():
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    assert row["schema"] == SIDECAR_SCHEMA
    assert row["geometry_valid"]
    assert len(row["tokens"]["token_atom_a"]) == 5  # 3N+2 for N=1
    assert int(row["tokens"]["token_center_internal"].sum()) == 1
    assert row["tokens"]["token_bond_features"].shape == (5, BOND_FEATURE_DIM)
    assert np.isfinite(row["tokens"]["token_distance"]).all()
    assert row["relations"]["relation_source"].size > 0


def test_bond_feature_order_ring_and_unknown_stereo_are_distinct():
    ring = Chem.MolFromSmiles("C1CC1")
    ring_bond = ring.GetBondWithIdx(0)
    vector = bond_feature_vector(ring_bond)
    assert vector.shape == (14,)
    assert vector[0] == 1.0 and vector[6] == 1.0 and vector[7] == 1.0

    class UnknownBond:
        def GetBondType(self):
            return Chem.rdchem.BondType.SINGLE

        def GetStereo(self):
            return object()

        def GetIsConjugated(self):
            return False

        def IsInRing(self):
            return False

    unknown = UnknownBond()
    assert bond_stereo_index(unknown) == STEREO_UNKNOWN
    assert bond_feature_vector(unknown)[7 + STEREO_UNKNOWN] == 1.0


def test_complete_encoder_endpoint_exchange_mask_and_gradients():
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    item = _attach_complete(Data(), row)
    item.glt3_token_batch = torch.zeros(len(row["tokens"]["token_atom_a"]), dtype=torch.long)
    item.glt3_geometry_valid = torch.tensor([True])
    model = CompleteTrimerGLTEncoder(dropout=0.0).eval()
    first = model(item)
    exchanged = copy.copy(item)
    exchanged.glt3_token_endpoint_z_a = item.glt3_token_endpoint_z_b.clone()
    exchanged.glt3_token_endpoint_z_b = item.glt3_token_endpoint_z_a.clone()
    exchanged.glt3_token_atom_a = item.glt3_token_atom_b.clone()
    exchanged.glt3_token_atom_b = item.glt3_token_atom_a.clone()
    second = model(exchanged)
    torch.testing.assert_close(first["line_states"], second["line_states"])
    torch.testing.assert_close(first["graph_geometry"], second["graph_geometry"])

    masked = torch.zeros(first["line_states"].size(0), dtype=torch.bool)
    masked[0] = True
    changed = copy.copy(item)
    changed.glt3_token_endpoint_z_a = item.glt3_token_endpoint_z_a.clone()
    changed.glt3_token_bond_features = item.glt3_token_bond_features.clone()
    changed.glt3_token_distance = item.glt3_token_distance.clone()
    changed.glt3_token_endpoint_z_a[0] = 8
    changed.glt3_token_bond_features[0] = 0
    changed.glt3_token_distance[0] = 2.9
    state_a = model._line_inputs(item, masked)
    state_b = model._line_inputs(changed, masked)
    torch.testing.assert_close(state_a[0], state_b[0])
    weighting = torch.linspace(
        0.5, 1.5, state_b[1:].numel(), dtype=state_b.dtype
    ).reshape_as(state_b[1:])
    (state_b[1:] * weighting).sum().backward()
    assert model.endpoint_projection.weight.grad is not None
    assert model.bond_feature_projection.weight.grad is not None
    assert model.distance_projection.weight.grad is not None
    assert model.distance_basis.centers.grad is not None
    assert model.distance_basis.centers.grad.abs().sum() > 0
    assert model.distance_basis.pair_affine.weight.grad is not None
    assert model.distance_basis.pair_affine.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_complete_encoder_is_rigid_motion_invariant():
    topology, trimer = _toy_pair()
    original = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    transformed_trimer = copy.copy(trimer)
    angle = math.pi / 3.0
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed_trimer.trimer_pos = trimer.trimer_pos @ rotation.T + torch.tensor([4.0, -2.0, 0.7])
    transformed = build_complete_trimer_glt_sample(topology, transformed_trimer, "*CC*")
    first = _attach_complete(Data(), original)
    second = _attach_complete(Data(), transformed)
    for item in (first, second):
        item.glt3_token_batch = torch.zeros(item.glt3_token_atom_a.numel(), dtype=torch.long)
        item.glt3_geometry_valid = torch.tensor([True])
    model = CompleteTrimerGLTEncoder(dropout=0.0).eval()
    output_a = model(first)
    output_b = model(second)
    torch.testing.assert_close(output_a["line_states"], output_b["line_states"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output_a["graph_geometry"], output_b["graph_geometry"], atol=1e-5, rtol=1e-5)


def test_complete_collate_and_fusion_shape():
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    base = Data(
        x=torch.zeros((2, 1)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        lga_edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        lga_spd=torch.tensor([0]),
        lga_path_index=torch.tensor([[0, -1, -1], [1, -1, -1]]),
        lga_path_mask=torch.tensor([[True, False, False], [True, False, False]]),
        lga_path_shift=torch.zeros((2, 3), dtype=torch.long),
        lga_path_bond_hist=torch.zeros((2, 2, 6)),
        lga_star_edge_mask=torch.zeros(1, dtype=torch.bool),
        topology_representation="canonical_lifted",
        mts_canonical_periodic=True,
        mips_local_lga_schema_version=2,
        canonical_ru_atom_index=torch.arange(2),
        canonical_to_trimer_base_atom_id=torch.arange(2),
        mips_x=torch.zeros((2, 137)),
        mips_backbone_mask=torch.zeros(2, dtype=torch.long),
        y=torch.zeros(1),
        graph_available=True,
        mips_condition_valid=True,
        polymer_link_mask=torch.zeros(1, dtype=torch.bool),
        lga_source_image_shift=torch.zeros(1, dtype=torch.long),
    )
    first = _attach_complete(base, row)
    second = _attach_complete(base, row)
    batch = mips_trimer_collate([first, second])
    assert batch.glt3_token_bond_features.shape == (10, 14)
    assert batch.glt3_token_ring.shape == (10,)
    assert int(batch.glt3_relation_source.max()) >= 5
    assert int(batch.glt3_relation_target.max()) >= 5
    assert torch.equal(batch.glt3_relation_source, batch.glt3_relation_source.long())
    fusion = CompleteTrimerGLTFusionRegressor(dropout=0.0)
    o8 = torch.randn(1, 512, requires_grad=True)
    glt = torch.randn(1, 512, requires_grad=True)
    output = fusion(o8, glt, glt_valid=torch.tensor([True]))
    assert output.shape == (1, 1)
    output.sum().backward()
    assert fusion.predictor[0].weight.grad is not None
    assert fusion.predictor[-1].weight.grad is not None
    assert o8.grad is not None and glt.grad is not None


def test_mixed_legacy_and_complete_rows_are_rejected():
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    complete = _attach_complete(Data(), row)
    legacy = copy.copy(complete)
    del legacy.glt3_token_bond_features
    with pytest.raises(ValueError, match="cannot mix complete-Trimer"):
        mips_trimer_collate([complete, legacy])


def test_complete_collate_rejects_missing_ring_field():
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    item = _attach_complete(Data(), row)
    del item.glt3_token_ring
    with pytest.raises(ValueError, match="require the Ring field"):
        mips_trimer_collate([item])


def test_complete_sidecar_writer_reader_roundtrip(tmp_path):
    topology, trimer = _toy_pair()
    row = build_complete_trimer_glt_sample(topology, trimer, "*CC*")
    root = tmp_path / "complete"
    write_complete_trimer_sidecar(root, [b"a" * 32], [row])
    reader = CompleteTrimerGLTSidecar(root)
    loaded = reader.model_row(0)
    assert reader.metadata["schema"] == SIDECAR_SCHEMA
    assert loaded["geometry_valid"]
    assert loaded["tokens"]["token_bond_features"].shape == (5, BOND_FEATURE_DIM)
    np.testing.assert_array_equal(
        loaded["tokens"]["token_center_internal"],
        row["tokens"]["token_center_internal"],
    )
