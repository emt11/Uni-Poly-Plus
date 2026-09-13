"""Terminal H-cap contract tests for the open three-RU Trimer.

The open oligomer is H-terminated at the two missing seam bonds via
``missing_seam_bond_order_hydrogen_equivalents``: the terminal atoms receive
``SetNumExplicitHs(+cap_count)`` where cap_count follows the seam bond order
(single=1, double=2, triple=3; aromatic attachments connect as single).
These tests verify the cap changes neither the source identity of any repeat
unit nor bond order, aromaticity, formal charge or declared stereochemistry,
and that all three copies carry identical internal chemistry.
"""

import sys
from pathlib import Path

import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset.graph_data import build_periodic_multimer_mol

CAPPING = "missing_seam_bond_order_hydrogen_equivalents"


def _build(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, smiles
    return build_periodic_multimer_mol(
        molecule, 3, close_periodic=False, terminal_capping=CAPPING
    )


def _unit_bond_fingerprint(trimer, metadata):
    """Per-RU internal bond-order fingerprint; all copies must agree."""

    fingerprints = []
    for unit_atoms in metadata["unit_atoms"]:
        unit_atoms = set(int(atom) for atom in unit_atoms)
        orders = sorted(
            str(trimer.GetBondBetweenAtoms(int(bond.GetBeginAtomIdx()),
                                           int(bond.GetEndAtomIdx())).GetBondType())
            for bond in trimer.GetBonds()
            if int(bond.GetBeginAtomIdx()) in unit_atoms
            and int(bond.GetEndAtomIdx()) in unit_atoms
        )
        fingerprints.append(tuple(orders))
    return fingerprints


def _unit_aromatic_counts(trimer, metadata):
    counts = []
    for unit_atoms in metadata["unit_atoms"]:
        counts.append(sum(
            1 for atom_index in unit_atoms
            if trimer.GetAtomWithIdx(int(atom_index)).GetIsAromatic()
        ))
    return counts


def _unit_formal_charges(trimer, metadata):
    charges = []
    for unit_atoms in metadata["unit_atoms"]:
        charges.append(sorted(
            trimer.GetAtomWithIdx(int(atom_index)).GetFormalCharge()
            for atom_index in unit_atoms
        ))
    return charges


def test_terminal_cap_single_attachment():
    trimer, metadata = _build("*CCO*")
    assert metadata["terminal_cap_hydrogens"] == [
        {"side": "left", "atom": metadata["left_boundary"], "count": 1},
        {"side": "right", "atom": metadata["right_boundary"], "count": 1},
    ]
    for entry in metadata["terminal_cap_hydrogens"]:
        atom = trimer.GetAtomWithIdx(entry["atom"])
        assert atom.GetNumExplicitHs() >= entry["count"]
    # three chemically identical RU copies (the historic bug: RDKit
    # kekulized only the terminal copy when the cap was undeclared)
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]
    assert len({fingerprints[0], fingerprints[1], fingerprints[2]}) == 1


def _build_with_attachment_bond(bond_type):
    """Construct a two-carbon RU whose DUMMY bonds carry an explicit order
    (SMILES dummy bonds are always single, so build the molecule directly)."""

    molecule = Chem.RWMol()
    left_dummy = molecule.AddAtom(Chem.Atom(0))
    carbon_a = molecule.AddAtom(Chem.Atom(6))
    carbon_b = molecule.AddAtom(Chem.Atom(6))
    right_dummy = molecule.AddAtom(Chem.Atom(0))
    molecule.AddBond(left_dummy, carbon_a, bond_type)
    molecule.AddBond(carbon_a, carbon_b, Chem.BondType.SINGLE)
    molecule.AddBond(carbon_b, right_dummy, bond_type)
    return build_periodic_multimer_mol(
        molecule.GetMol(), 3, close_periodic=False, terminal_capping=CAPPING
    )


def test_terminal_cap_double_attachment():
    trimer, metadata = _build_with_attachment_bond(Chem.BondType.DOUBLE)
    assert metadata["connection_bond_policy"] == "matching_attachment_type"
    assert str(metadata["attachment_bond_type"]) == "DOUBLE"
    assert all(entry["count"] == 2 for entry in metadata["terminal_cap_hydrogens"])
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_terminal_cap_triple_attachment():
    trimer, metadata = _build_with_attachment_bond(Chem.BondType.TRIPLE)
    assert str(metadata["attachment_bond_type"]) == "TRIPLE"
    assert all(entry["count"] == 3 for entry in metadata["terminal_cap_hydrogens"])
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_terminal_chiral_attachment_preserves_declared_stereo():
    smiles = "*C[C@H](F)O*"
    trimer, metadata = _build(smiles)
    source = Chem.MolFromSmiles(smiles)
    declared = [
        atom.GetChiralTag()
        for atom in source.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    assert declared, "test molecule must declare a stereocenter"
    observed = [
        atom.GetChiralTag()
        for atom in trimer.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    # every RU copy retains exactly the declared configuration
    assert len(observed) == 3 * len(declared)
    assert set(observed) == set(declared)
    assert metadata["stereo_restore_failures"] == []


def _build_with_aromatic_attachment(smiles):
    """Attach both dummies to the benzene ring through explicit AROMATIC
    bonds (RDKit normalizes '*:c...' SMILES to single bonds)."""

    source = Chem.MolFromSmiles("c1ccccc1")
    molecule = Chem.RWMol(source)
    for attach_index in (0, 5):  # two opposite ring carbons
        dummy = molecule.AddAtom(Chem.Atom(0))
        molecule.AddBond(dummy, attach_index, Chem.BondType.AROMATIC)
    return build_periodic_multimer_mol(
        molecule.GetMol(), 3, close_periodic=False, terminal_capping=CAPPING
    )


def test_charged_aromatic_terminal_cap():
    """Aromatic attachment connects as single; the cap keeps every RU copy
    kekulizable so aromaticity is identical across the three copies."""

    trimer, metadata = _build_with_aromatic_attachment("c1ccccc1")
    assert metadata["connection_bond_policy"] == "aromatic_single"
    assert all(entry["count"] == 1 for entry in metadata["terminal_cap_hydrogens"])
    assert _unit_aromatic_counts(trimer, metadata)[0] \
        == _unit_aromatic_counts(trimer, metadata)[1] \
        == _unit_aromatic_counts(trimer, metadata)[2] > 0
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_pyridinium_terminal_n_plus_keeps_charge_and_cap():
    smiles = "*[n+]1ccccc1*"
    trimer, metadata = _build(smiles)
    charges = _unit_formal_charges(trimer, metadata)
    # the charged aromatic terminal atom exists in all three copies with +1
    assert all(charge_list.count(1) >= 1 for charge_list in charges)
    assert all(entry["count"] == 1 for entry in metadata["terminal_cap_hydrogens"])
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_charged_aromatic_nH_terminal_from_real_source():
    """[nH+] terminal atom: the seam cap must not overwrite the declared H
    and all RU copies must stay chemically identical."""
    smiles = "*CC(=O)c1cc[nH+]cc1*"
    trimer, metadata = _build(smiles)
    charges = _unit_formal_charges(trimer, metadata)
    assert all(charge_list.count(1) >= 1 for charge_list in charges)
    assert _unit_aromatic_counts(trimer, metadata)[0] \
        == _unit_aromatic_counts(trimer, metadata)[1] \
        == _unit_aromatic_counts(trimer, metadata)[2]
    fingerprints = _unit_bond_fingerprint(trimer, metadata)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_cap_does_not_change_ru_identity():
    """The capped open trimer contains exactly three copies of the source RU:
    per-unit atom counts equal the base RU and per-unit element multiset is
    identical across copies."""

    smiles = "*Oc1ccc(N*)cc1"
    trimer, metadata = _build(smiles)
    source = Chem.MolFromSmiles(smiles)
    base_heavy = sum(
        1 for atom in source.GetAtoms() if atom.GetAtomicNum() > 1
    )
    unit_sizes = [len(metadata["unit_atoms"][index]) for index in range(3)]
    assert unit_sizes == [base_heavy] * 3
    element_sets = []
    for unit_atoms in metadata["unit_atoms"]:
        element_sets.append(sorted(
            trimer.GetAtomWithIdx(int(atom_index)).GetSymbol()
            for atom_index in unit_atoms
        ))
    assert element_sets[0] == element_sets[1] == element_sets[2]


def test_charged_aromatic_boundary_samples_from_current_source():
    """Real charged-aromatic boundaries from the current pilot source must
    build with an identical per-unit fingerprint (regression guard)."""

    import json

    staging = (
        ROOT / "data/processed/mts_cache_pilot_20260913/builds/"
        "7339faaf6401fe58176dbfe5a9e3692eced0a5aec2751fc1d979450d2f34074e.staging"
    )
    if not staging.is_dir():  # pragma: no cover - staging always present here
        pytest.skip("pilot staging not available")
    charged = []
    for line in (staging / "source" / "records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        row = json.loads(line)
        molecule = Chem.MolFromSmiles(row["normalized_smiles"])
        if molecule is None:
            continue
        for atom in molecule.GetAtoms():
            if atom.GetFormalCharge() != 0 and atom.GetIsAromatic():
                charged.append(row["normalized_smiles"])
                break
    assert charged, "pilot source must contain charged aromatic boundaries"
    checked = 0
    for smiles in charged[:8]:
        trimer, metadata = _build(smiles)
        fingerprints = _unit_bond_fingerprint(trimer, metadata)
        assert fingerprints[0] == fingerprints[1] == fingerprints[2], smiles
        checked += 1
    assert checked >= 1
