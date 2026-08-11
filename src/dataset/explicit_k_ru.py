"""Corrected explicit ``k``-RU topology for the MTS comparison route.

Unlike the canonical production representation, this module materialises the
smallest finite repeated chain whose open end-to-end graph distance is greater
than five, then adds one virtual terminal polymer link.  It deliberately does
not provide coordinates: finite-Trimer geometry remains an independent cache
layer keyed by canonical RU atom identity.
"""

from __future__ import annotations

from rdkit import Chem
import torch
from torch_geometric.data import Data

from .canonical_periodic import build_canonical_periodic_topology
from .graph_data import (
    MIPSLocalConfig,
    _bond_path_codes,
    attach_mips_local_lga,
    build_mips_local_structure,
)
from .mips_trimer_contract import (
    EXPLICIT_FEATURE_SCHEMA,
    EXPLICIT_LGA_SCHEMA_VERSION,
    EXPLICIT_TOPOLOGY_LMDB_SCHEMA,
    TOPOLOGY_EXPLICIT,
)


def _physical_bond_tensors(molecule: Chem.Mol):
    sources, targets, attributes = [], [], []
    for bond in molecule.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        sources.extend((begin, end))
        targets.extend((end, begin))
        # ``edge_attr`` is diagnostic in MTS, but retain the real bond type so
        # the explicit cache is self-contained and auditable.
        attributes.extend((_bond_path_codes(bond, False)[0],) * 2)
    return (
        torch.tensor([sources, targets], dtype=torch.long),
        torch.tensor(attributes, dtype=torch.long).reshape(-1, 1),
    )


def build_explicit_k_ru_topology(
    smiles_or_mol,
    *,
    max_hops: int = 2,
    boundary_threshold: int = 5,
    max_repeat_units: int = 16,
    max_model_atoms: int = 384,
) -> Data:
    """Build the corrected, runnable explicit MTS comparison topology.

    The minimum legal ``k`` is selected from the *open* repeated chain.  The
    terminal Star edge is added only to the sparse attention adjacency, never
    to the RDKit molecule, so it cannot be mistaken for a physical bond or a
    Trimer distance edge.
    """

    max_hops = int(max_hops)
    boundary_threshold = int(boundary_threshold)
    if max_hops != 2:
        raise ValueError("MTS explicit comparison requires max_hops=2")
    if boundary_threshold != 5:
        raise ValueError("MTS explicit comparison requires boundary >5")

    config = MIPSLocalConfig(
        max_hops=max_hops,
        max_repeat_units=int(max_repeat_units),
        max_model_atoms=int(max_model_atoms),
    )
    if int(config.required_boundary_distance) != boundary_threshold:
        raise ValueError("explicit boundary threshold disagrees with max_hops")
    structure = build_mips_local_structure(smiles_or_mol, config=config)

    # Use the canonical builder as the one feature/identity source.  It does
    # not supply explicit relations; those are independently built below from
    # the finite molecule and virtual Star edge.
    canonical = build_canonical_periodic_topology(
        smiles_or_mol, max_hops=max_hops
    )
    molecule = structure["structure_mol"]
    metadata = structure.get("repeat_metadata") or {}
    unit_atoms = metadata.get("unit_atoms") or []
    repeat_units = int(structure.get("mips_repeat_units", len(unit_atoms) or 1))
    canonical_count = int(canonical.mips_x.size(0))
    if len(unit_atoms) != repeat_units:
        raise ValueError("explicit topology has an incomplete RU identity table")
    if any(len(unit) != canonical_count for unit in unit_atoms):
        raise ValueError("explicit RU copies do not match canonical atom count")

    data = Data()
    data.x = canonical.mips_x.repeat(repeat_units, 1).clone()
    data.mips_x = data.x.clone()
    data.mips_backbone_mask = canonical.mips_backbone_mask.repeat(
        repeat_units
    ).clone()
    data.edge_index, data.edge_attr = _physical_bond_tensors(molecule)
    data.atomic_numbers = torch.tensor(
        [int(atom.GetAtomicNum()) for atom in molecule.GetAtoms()],
        dtype=torch.long,
    )
    data.atomic_number = data.atomic_numbers
    data.z = data.atomic_numbers
    data.num_nodes = int(molecule.GetNumAtoms())

    attach_mips_local_lga(data, structure, config=config)
    terminal_star = data.lga_star_edge_mask.bool().clone()
    source, target = data.lga_edge_index.long()
    source_copy = data.ru_copy_index[source]
    target_copy = data.ru_copy_index[target]
    # Every direct inter-RU connection represents the same homopolymer bond,
    # including the virtual last->first Star edge.  All receive the symmetric
    # d_star bias; internal one-hop bonds do not.
    polymer_link = terminal_star | (
        (data.lga_spd.long() == 1) & (source_copy != target_copy)
    )
    data.polymer_link_mask = polymer_link
    data.lga_polymer_link_mask = polymer_link
    data.lga_star_edge_mask = polymer_link

    data.canonical_atom_count = canonical_count
    data.canonical_to_trimer_base_atom_id = torch.arange(
        canonical_count, dtype=torch.long
    )
    data.canonical_to_trimer_base_atom_index = (
        data.canonical_to_trimer_base_atom_id
    )
    # Trimer is merged later.  Keep one placeholder per explicit node so a
    # canonical mapping can be expanded without guessing copy/offset identity.
    data.mips_to_trimer_central_index = torch.full(
        (data.num_nodes,), -1, dtype=torch.long
    )
    data.mips_repeat_factor = repeat_units
    data.mips_repeat_units = repeat_units
    data.mips_boundary_distance = int(structure["mips_boundary_distance"])
    data.mips_distance_threshold = max_hops + 1
    data.mips_condition_valid = bool(structure["mips_condition_valid"])
    data.graph_available = bool(structure["graph_available"])
    data.mips_alias_free = bool(structure["graph_available"])
    data.topology_failure_code = str(structure.get("topology_failure_code", ""))
    data.feature_schema = EXPLICIT_FEATURE_SCHEMA
    data.explicit_k_ru_topology_schema = EXPLICIT_TOPOLOGY_LMDB_SCHEMA
    data.mips_local_lga_schema_version = EXPLICIT_LGA_SCHEMA_VERSION
    data.mts_canonical_periodic = False
    data.mts_topology_representation = TOPOLOGY_EXPLICIT
    data.topology_representation = TOPOLOGY_EXPLICIT
    data.explicit_k_ru_builder_version = 1
    normalized = getattr(canonical, "normalized_canonical_smiles", None)
    if normalized is None:
        source = (
            Chem.Mol(smiles_or_mol)
            if isinstance(smiles_or_mol, Chem.Mol)
            else Chem.MolFromSmiles(str(smiles_or_mol))
        )
        normalized = Chem.MolToSmiles(source, canonical=True) if source else ""
    data.normalized_canonical_smiles = str(normalized)
    data.smiles = (
        str(smiles_or_mol)
        if not isinstance(smiles_or_mol, Chem.Mol)
        else str(normalized)
    )
    return data


__all__ = ["build_explicit_k_ru_topology"]
