#!/usr/bin/env python3
"""Read-only audit of an accepted Trimer LMDB point-in-time snapshot.

This tool deliberately does not call ETKDG, MMFF, the Trimer generator, or
any cache writer.  It reads the source rows copied into a results snapshot and
the accepted RU/Topology/Trimer LMDB payloads through read-only transactions.
All output is written below the supplied results directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.cache_lifecycle import deserialize_record  # noqa: E402
from src.dataset.cache_spec import RECORD_FIELDS  # noqa: E402
from src.dataset.canonical_periodic import resolve_normalized_identity  # noqa: E402


DEFINED_STEREO = {
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
}


class AuditRecordError(RuntimeError):
    pass


def _bond_code(bond: Chem.Bond) -> int:
    value = bond.GetBondType()
    if value == Chem.BondType.SINGLE:
        return 1
    if value == Chem.BondType.DOUBLE:
        return 2
    if value == Chem.BondType.TRIPLE:
        return 3
    if value == Chem.BondType.AROMATIC:
        return 4
    return int(round(float(bond.GetBondTypeAsDouble())))


def _bond_order(code: int, aromatic: bool = False) -> float:
    return 1.5 if bool(aromatic) or int(code) == 4 else float(code)


def _as_array(data, name, *, dtype=None):
    if not hasattr(data, name):
        raise AuditRecordError(f"missing field: {name}")
    value = torch.as_tensor(getattr(data, name), dtype=dtype).detach().cpu()
    return value.numpy()


def _scalar(data, name):
    if not hasattr(data, name):
        raise AuditRecordError(f"missing field: {name}")
    value = torch.as_tensor(getattr(data, name)).reshape(-1)
    if value.numel() != 1:
        raise AuditRecordError(f"scalar field has invalid shape: {name}")
    return value.item()


def _bool_scalar(data, name):
    return bool(_scalar(data, name))


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _source_info(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise AuditRecordError("normalized source cannot be parsed")
    canonical = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    base_indices = [
        int(atom.GetIdx()) for atom in molecule.GetAtoms()
        if int(atom.GetAtomicNum()) != 0
    ]
    base_rank = {atom: rank for rank, atom in enumerate(base_indices)}
    dummy = []
    for atom in molecule.GetAtoms():
        if int(atom.GetAtomicNum()) != 0:
            continue
        neighbours = list(atom.GetNeighbors())
        if len(neighbours) != 1 or neighbours[0].GetIdx() not in base_rank:
            raise AuditRecordError("source dummy attachment is not unary")
        neighbour = int(neighbours[0].GetIdx())
        bond = molecule.GetBondBetweenAtoms(int(atom.GetIdx()), neighbour)
        dummy.append({
            "index": int(atom.GetIdx()),
            "neighbor_source": neighbour,
            "neighbor_base": int(base_rank[neighbour]),
            "code": _bond_code(bond),
            "aromatic": bool(bond.GetIsAromatic()),
        })
    if len(dummy) != 2:
        raise AuditRecordError("source does not contain exactly two dummies")

    bonds = {}
    for bond in molecule.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        if left not in base_rank or right not in base_rank:
            continue
        pair = (min(base_rank[left], base_rank[right]),
                max(base_rank[left], base_rank[right]))
        bonds[pair] = {
            "code": _bond_code(bond),
            "aromatic": bool(bond.GetIsAromatic()),
        }
    atoms = []
    for index in base_indices:
        atom = molecule.GetAtomWithIdx(index)
        try:
            total_h = int(atom.GetTotalNumHs())
            total_valence = float(atom.GetTotalValence())
        except (RuntimeError, ValueError):
            total_h = -1
            total_valence = float("nan")
        atoms.append({
            "z": int(atom.GetAtomicNum()),
            "isotope": int(atom.GetIsotope()),
            "formal_charge": int(atom.GetFormalCharge()),
            "aromatic": bool(atom.GetIsAromatic()),
            "total_h": total_h,
            "total_valence": total_valence,
            "chiral_tag": int(atom.GetChiralTag()),
        })
    stereo = []
    for bond in molecule.GetBonds():
        if bond.GetStereo() not in DEFINED_STEREO:
            continue
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin not in base_rank or end not in base_rank:
            continue
        stereo.append({
            "begin_base": int(base_rank[begin]),
            "end_base": int(base_rank[end]),
            "stereo": str(bond.GetStereo()),
            "stereo_enum": bond.GetStereo(),
            "refs": tuple(int(value) for value in bond.GetStereoAtoms()),
            "ref_base": tuple(
                int(base_rank[value]) if int(value) in base_rank else None
                for value in bond.GetStereoAtoms()
            ),
        })
    return {
        "molecule": molecule,
        "canonical": canonical,
        "base_indices": base_indices,
        "base_rank": base_rank,
        "dummy": dummy,
        "bonds": bonds,
        "atoms": atoms,
        "stereo": stereo,
    }


def _fragment_kekule_smiles(source_info, payload_bonds, payload_atoms=None):
    """Normalize one RU's internal graph independently of the finite cap.

    Dummy atoms and their original attachment bond orders are restored only as
    valence anchors.  No coordinates or generator output is used here.
    """
    source = source_info["molecule"]
    base_rank = source_info["base_rank"]
    rw = Chem.RWMol()
    old_to_new = {}
    for atom in source.GetAtoms():
        copy = Chem.Atom(atom)
        copy.SetIsAromatic(False)
        old_to_new[int(atom.GetIdx())] = rw.AddAtom(copy)
    for bond in source.GetBonds():
        left = int(bond.GetBeginAtomIdx())
        right = int(bond.GetEndAtomIdx())
        if left in base_rank and right in base_rank:
            pair = (min(base_rank[left], base_rank[right]),
                    max(base_rank[left], base_rank[right]))
            attr = payload_bonds.get(pair)
            if attr is None:
                raise AuditRecordError("fragment bond is missing")
            code = int(attr["code"])
            aromatic = bool(attr["aromatic"]) or code == 4
            bond_type = Chem.BondType.AROMATIC if aromatic else {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }.get(code)
            if bond_type is None:
                raise AuditRecordError("fragment bond type is unsupported")
        else:
            bond_type = bond.GetBondType()
            aromatic = bool(bond.GetIsAromatic())
        rw.AddBond(old_to_new[left], old_to_new[right], bond_type)
        if aromatic:
            rw.GetAtomWithIdx(old_to_new[left]).SetIsAromatic(True)
            rw.GetAtomWithIdx(old_to_new[right]).SetIsAromatic(True)
    fragment = rw.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except Exception:
        try:
            Chem.Kekulize(fragment, clearAromaticFlags=True)
        except Exception as exc:
            raise AuditRecordError("fragment normalization failed") from exc
    try:
        return Chem.MolToSmiles(
            fragment, canonical=True, isomericSmiles=True, kekuleSmiles=True
        )
    except Exception as exc:
        raise AuditRecordError("fragment canonicalization failed") from exc


def _reconstruct_payload_mol(arrays, pair_attr):
    """Rebuild the saved all-atom graph for independent RDKit semantics.

    The accepted payload intentionally stores a directed edge table rather
    than a serialized ``Mol``.  Rebuilding a throw-away molecule here lets
    the audit ask RDKit for total valence and hybridization instead of
    approximating aromatic valence by summing ``1.5`` bond orders.
    """

    rw = Chem.RWMol()
    for index, atomic_number in enumerate(arrays["atomic"].tolist()):
        atom = Chem.Atom(int(atomic_number))
        atom.SetIsotope(int(arrays["isotope"][index]))
        atom.SetFormalCharge(int(arrays["charge"][index]))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(arrays["aromatic_atom"][index]))
        rw.AddAtom(atom)
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    for (left, right), (code, aromatic) in pair_attr.items():
        bond_type = bond_types.get(int(code))
        if bond_type is None:
            raise AuditRecordError("payload bond type is unsupported")
        rw.AddBond(int(left), int(right), bond_type)
        bond = rw.GetBondBetweenAtoms(int(left), int(right))
        if bond is None:
            raise AuditRecordError("payload bond disappeared during rebuild")
        bond.SetIsAromatic(bool(aromatic) or int(code) == 4)
        if bool(aromatic) or int(code) == 4:
            rw.GetAtomWithIdx(int(left)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(right)).SetIsAromatic(True)
    molecule = rw.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise AuditRecordError("saved payload graph cannot be sanitized") from exc
    return molecule


def _resolve_source_mapping(topology, row, source_info):
    normalized = str(row["normalized_smiles"])
    raw = str(row["source_smiles"])
    raw_mol = Chem.MolFromSmiles(raw)
    if raw_mol is None:
        raise AuditRecordError("raw source cannot be parsed")
    if raw == normalized:
        atom_map = _as_array(topology, "source_to_normalized_atom_id", dtype=torch.long).reshape(-1).tolist()
        base_map = _as_array(topology, "source_to_normalized_canonical_atom_id", dtype=torch.long).reshape(-1).tolist()
        if atom_map != list(range(raw_mol.GetNumAtoms())):
            raise AuditRecordError("same-spelling source atom mapping is not identity")
        if base_map != list(range(len(source_info["base_indices"]))):
            raise AuditRecordError("same-spelling source base mapping is not identity")
        expected_attachment = []
        for item in source_info["dummy"]:
            expected_attachment.append({
                "source_dummy": item["index"],
                "source_neighbor": item["neighbor_source"],
                "normalized_dummy": item["index"],
                "normalized_neighbor": item["neighbor_source"],
            })
        observed_attachment = getattr(topology, "source_to_normalized_attachment_map", None)
        if observed_attachment is None or list(observed_attachment) != expected_attachment:
            raise AuditRecordError("same-spelling attachment mapping mismatch")
        return True
    identity = resolve_normalized_identity(
        topology, raw, require_fields=True
    )
    if str(identity["normalized_smiles"]) != normalized:
        raise AuditRecordError("raw/normalized mapping returned another identity")
    return True


def _record_arrays(trimer):
    fields = set(trimer.keys())
    missing = sorted(set(RECORD_FIELDS["trimer"]) - fields)
    if missing:
        raise AuditRecordError("missing Trimer fields: " + ",".join(missing))
    return {
        "pos": _as_array(trimer, "trimer_pos", dtype=torch.float64),
        "atomic": _as_array(trimer, "trimer_atomic_number", dtype=torch.long).reshape(-1),
        "isotope": _as_array(trimer, "trimer_isotope", dtype=torch.long).reshape(-1),
        "source_count": int(_scalar(trimer, "trimer_source_atom_count")),
        "is_source": _as_array(trimer, "is_source_atom", dtype=torch.bool).reshape(-1),
        "is_source_h": _as_array(trimer, "is_source_explicit_h", dtype=torch.bool).reshape(-1),
        "is_added_h": _as_array(trimer, "is_added_h", dtype=torch.bool).reshape(-1),
        "parents": _as_array(trimer, "h_parent_heavy_index", dtype=torch.long).reshape(-1),
        "atom_id": _as_array(trimer, "trimer_atom_id", dtype=torch.long).reshape(-1),
        "edge": _as_array(trimer, "trimer_edge_index", dtype=torch.long),
        "bond": _as_array(trimer, "trimer_bond_type", dtype=torch.long).reshape(-1),
        "aromatic_bond": _as_array(trimer, "trimer_bond_aromatic", dtype=torch.bool).reshape(-1),
        "charge": _as_array(trimer, "trimer_formal_charge", dtype=torch.long).reshape(-1),
        "aromatic_atom": _as_array(trimer, "trimer_is_aromatic", dtype=torch.bool).reshape(-1),
        "chiral": _as_array(trimer, "trimer_chiral_tag", dtype=torch.long).reshape(-1),
        "role": _as_array(trimer, "trimer_attachment_role", dtype=torch.long).reshape(-1),
        "internal_degree": _as_array(trimer, "trimer_internal_degree", dtype=torch.long).reshape(-1),
        "base": _as_array(trimer, "trimer_base_ru_atom_id", dtype=torch.long).reshape(-1),
        "offset": _as_array(trimer, "trimer_ru_offset", dtype=torch.long).reshape(-1),
        "central_mask": _as_array(trimer, "trimer_central_ru_mask", dtype=torch.bool).reshape(-1),
        "central_index": _as_array(trimer, "trimer_central_atom_index", dtype=torch.long).reshape(-1),
        "o8_map": _as_array(trimer, "mips_to_trimer_central_index", dtype=torch.long).reshape(-1),
        "o8_heavy_mask": _as_array(trimer, "o8_heavy_mask", dtype=torch.bool).reshape(-1),
        "o8_heavy_indices": _as_array(trimer, "o8_heavy_indices", dtype=torch.long).reshape(-1),
        "heavy_mask": _as_array(trimer, "trimer_heavy_mask", dtype=torch.bool).reshape(-1),
        "heavy_indices": _as_array(trimer, "trimer_heavy_indices", dtype=torch.long).reshape(-1),
    }


def _topology_check(topology, source_info):
    fields = set(topology.keys())
    missing = sorted(set(RECORD_FIELDS["topology"]) - fields)
    if missing:
        raise AuditRecordError("missing Topology fields: " + ",".join(missing))
    count = len(source_info["base_indices"])
    z = _as_array(topology, "z", dtype=torch.long).reshape(-1)
    if int(_scalar(topology, "num_nodes")) != count or z.tolist() != [a["z"] for a in source_info["atoms"]]:
        raise AuditRecordError("Topology canonical atom identity mismatch")
    for name in ("canonical_ru_atom_index", "canonical_to_trimer_base_atom_id"):
        value = _as_array(topology, name, dtype=torch.long).reshape(-1).tolist()
        if value != list(range(count)):
            raise AuditRecordError(f"Topology {name} is not identity")
    x = _as_array(topology, "mips_x", dtype=torch.float64)
    if x.shape != (count, 137) or not bool(np.isfinite(x).all()):
        raise AuditRecordError("Topology mips_x shape/finite contract failed")
    if not bool(_scalar(topology, "graph_available")):
        raise AuditRecordError("Topology graph is unavailable for accepted Trimer")
    normalized = str(getattr(topology, "normalized_canonical_smiles", ""))
    if normalized != source_info["canonical"]:
        raise AuditRecordError("Topology normalized canonical SMILES mismatch")
    return True


def _stereo_cosine(positions, i, j, ref0, ref1, stereo):
    axis = positions[j] - positions[i]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None, "STEREO_UNDETERMINED"
    axis = axis / norm
    u = positions[ref0] - positions[i]
    v = positions[ref1] - positions[j]
    u = u - float(np.dot(u, axis)) * axis
    v = v - float(np.dot(v, axis)) * axis
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return None, "STEREO_UNDETERMINED"
    cosine = float(np.dot(u, v) / denominator)
    if not math.isfinite(cosine) or abs(cosine) <= 1e-12:
        return None, "STEREO_UNDETERMINED"
    expected = -1.0 if stereo in {
        Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS
    } else 1.0
    if expected * cosine < 0.0:
        return cosine, "STEREO_MISMATCH"
    return cosine, "PASS"


def _audit_stereo(source_info, arrays, base_global, dummy_indices):
    declared = len(source_info["stereo"])
    if not declared:
        return {"status": "N/A", "declared_bonds": 0, "checked": 0,
                "terminal_unresolved": 0, "failures": []}
    positions = arrays["pos"]
    checked = 0
    terminal_unresolved = 0
    failures = []
    # Physical seam lookup is independent of any bond-stereo field in the
    # payload.  It maps a center-RU dummy reference to the real adjacent RU.
    dummy_base = [item["neighbor_base"] for item in source_info["dummy"]]
    # ``dummy_indices`` are source-molecule atom ids; ``dummy_base`` are the
    # corresponding normalized base ranks.  They are intentionally distinct
    # (a dummy can have an index outside the non-dummy base range).
    seam_external = {
        dummy_indices[0]: base_global[-1][dummy_base[1]],
        dummy_indices[1]: base_global[1][dummy_base[0]],
    }
    for descriptor in source_info["stereo"]:
        for ru in (-1, 0, 1):
            endpoints = [
                base_global[ru][descriptor["begin_base"]],
                base_global[ru][descriptor["end_base"]],
            ]
            refs = []
            unresolved = False
            for ref, ref_base in zip(descriptor["refs"], descriptor["ref_base"]):
                if ref_base is not None:
                    refs.append(base_global[ru][ref_base])
                elif ru == 0 and ref in seam_external:
                    refs.append(seam_external[ref])
                else:
                    unresolved = True
                    break
            if unresolved:
                terminal_unresolved += 1
                continue
            checked += 1
            cosine, result = _stereo_cosine(
                positions, endpoints[0], endpoints[1], refs[0], refs[1],
                descriptor["stereo_enum"],
            )
            if result != "PASS":
                failures.append({
                    "bond": [descriptor["begin_base"], descriptor["end_base"]],
                    "ru_offset": ru,
                    "result": result,
                    "cosine": _finite_float(cosine),
                })
    if failures:
        status = "FAIL"
    elif terminal_unresolved:
        status = "UNRESOLVED"
    else:
        status = "PASS"
    return {
        "status": status,
        "declared_bonds": declared,
        "checked": checked,
        "terminal_unresolved": terminal_unresolved,
        "failures": failures[:8],
    }


def _audit_record(key_hex, row, topology, trimer, runtime_row, position):
    flags = {
        "record_complete": False,
        "source_identity_ok": False,
        "heavy_connectivity_ok": False,
        "center_ru_chemistry_ok": False,
        "terminal_cap_ok": False,
        "aromatic_equivalence_ok": False,
        "formal_charge_ok": False,
        "valence_ok": False,
        "isotope_ok": False,
        "bond_table_ok": False,
        "o8_mapping_ok": False,
        "stereo_ok": None,
        "finite_geometry_ok": False,
    }
    failures = []
    unresolved = []
    diagnostics = {}
    source_info = None
    arrays = None
    try:
        source_info = _source_info(row["normalized_smiles"])
        dummy_codes = [item["code"] for item in source_info["dummy"]]
        seam_code = (
            1 if dummy_codes[0] != dummy_codes[1] or 4 in dummy_codes
            else dummy_codes[0]
        )
        diagnostics["high_risk_groups"] = {
            "double_seam": seam_code == 2,
            "single_seam": seam_code == 1,
            "aromatic": any(a["aromatic"] for a in source_info["atoms"]),
            "fused_aromatic": (
                source_info["molecule"].GetRingInfo().NumRings() > 1
                and any(a["aromatic"] for a in source_info["atoms"])
            ),
            "hetero_aromatic": any(
                a["aromatic"] and a["z"] not in (1, 6)
                for a in source_info["atoms"]
            ),
            "charged_aromatic": any(
                a["aromatic"] and a["formal_charge"] != 0
                for a in source_info["atoms"]
            ),
        }
        canonical = source_info["canonical"]
        if canonical != str(row["normalized_smiles"]):
            failures.append("FAIL_SOURCE_IDENTITY")
        _resolve_source_mapping(topology, row, source_info)
        _topology_check(topology, source_info)
        flags["source_identity_ok"] = not failures
        arrays = _record_arrays(trimer)
        n_all = int(arrays["atomic"].size)
        base_count = len(source_info["base_indices"])
        runtime_bytes = int(runtime_row.get("record_bytes", 0))
        payload_bytes = runtime_row.get("_payload_bytes")
        if runtime_bytes <= 0 or payload_bytes is None or runtime_bytes != int(payload_bytes):
            failures.append("FAIL_RECORD_BYTES")
        if len(key_hex) != 64:
            failures.append("FAIL_SAMPLE_KEY")
        if arrays["pos"].shape != (n_all, 3) or not bool(np.isfinite(arrays["pos"]).all()):
            failures.append("FAIL_NONFINITE_GEOMETRY")
        else:
            flags["finite_geometry_ok"] = True
        if arrays["source_count"] != 3 * base_count:
            failures.append("FAIL_SOURCE_COUNT")
        if arrays["atomic"].size != n_all:
            failures.append("FAIL_ATOM_COUNT")
        if arrays["atom_id"].tolist() != list(range(n_all)):
            failures.append("FAIL_ATOM_ID")
        if not (_bool_scalar(trimer, "trimer_geometry_valid")
                and _bool_scalar(trimer, "trimer_geometry_is_3d")
                and not _bool_scalar(trimer, "trimer_2d_fallback")):
            failures.append("FAIL_GEOMETRY_FLAGS")
        if str(getattr(trimer, "trimer_geometry_source", "")) == "":
            failures.append("FAIL_GEOMETRY_SOURCE")
        source_region = arrays["is_source"]
        if source_region.size != n_all or not np.array_equal(
            arrays["is_added_h"], ~source_region
        ) or not np.array_equal(
            arrays["is_source_h"], source_region & (arrays["atomic"] == 1)
        ):
            failures.append("FAIL_SOURCE_H_FLAGS")
        if arrays["isotope"].size != n_all:
            failures.append("FAIL_ISOTOPE_SHAPE")
        if np.any(arrays["isotope"][arrays["is_added_h"]] != 0):
            failures.append("FAIL_ISOTOPE_H")
        if np.any(arrays["atomic"][arrays["is_added_h"]] != 1):
            failures.append("FAIL_ADDED_H_ATOMIC_NUMBER")
        if not np.array_equal(arrays["heavy_mask"], arrays["atomic"] > 1):
            failures.append("FAIL_HEAVY_MASK")
        expected_heavy = np.flatnonzero(arrays["atomic"] > 1)
        if not np.array_equal(arrays["heavy_indices"], expected_heavy):
            failures.append("FAIL_HEAVY_INDICES")
        if arrays["base"].size != n_all or arrays["offset"].size != n_all:
            failures.append("FAIL_IDENTITY_SHAPE")
        if not np.all(arrays["base"][arrays["is_source"]] >= 0):
            failures.append("FAIL_SOURCE_BASE_ID")
        if np.any(arrays["base"][arrays["is_added_h"]] >= 0):
            failures.append("FAIL_ADDED_H_BASE_ID")
        if not np.array_equal(arrays["central_mask"], arrays["offset"] == 0):
            failures.append("FAIL_CENTRAL_MASK")

        base_global = {}
        for ru in (-1, 0, 1):
            base_global[ru] = {}
            for base_id in range(base_count):
                found = np.flatnonzero(
                    source_region
                    & (arrays["offset"] == ru)
                    & (arrays["base"] == base_id)
                )
                if found.size != 1:
                    failures.append("FAIL_RU_BASE_MAPPING")
                else:
                    base_global[ru][base_id] = int(found[0])
        if all(len(base_global[ru]) == base_count for ru in (-1, 0, 1)):
            expected_z = np.asarray([a["z"] for a in source_info["atoms"]])
            expected_iso = np.asarray([a["isotope"] for a in source_info["atoms"]])
            expected_charge = np.asarray([a["formal_charge"] for a in source_info["atoms"]])
            for ru in (-1, 0, 1):
                indices = np.asarray([base_global[ru][i] for i in range(base_count)])
                if not np.array_equal(arrays["atomic"][indices], expected_z):
                    failures.append("FAIL_SOURCE_IDENTITY")
                if not np.array_equal(arrays["isotope"][indices], expected_iso):
                    failures.append("FAIL_ISOTOPE")
                if not np.array_equal(arrays["charge"][indices], expected_charge):
                    failures.append("FAIL_FORMAL_CHARGE")
            flags["isotope_ok"] = "FAIL_ISOTOPE" not in failures and "FAIL_ISOTOPE_H" not in failures
            flags["formal_charge_ok"] = "FAIL_FORMAL_CHARGE" not in failures

        edge = arrays["edge"]
        if edge.ndim != 2 or edge.shape[0] != 2 or edge.shape[1] != arrays["bond"].size \
                or arrays["aromatic_bond"].size != arrays["bond"].size:
            failures.append("FAIL_BOND_TABLE_SHAPE")
            physical = {}
        else:
            physical = defaultdict(list)
            for column in range(edge.shape[1]):
                left, right = int(edge[0, column]), int(edge[1, column])
                if left < 0 or right < 0 or left >= n_all or right >= n_all or left == right:
                    failures.append("FAIL_BOND_ENDPOINT")
                    continue
                pair = (min(left, right), max(left, right))
                physical[pair].append((left, right, int(arrays["bond"][column]), bool(arrays["aromatic_bond"][column])))
            pair_attr = {}
            for pair, values in physical.items():
                expected_directions = {
                    (int(pair[0]), int(pair[1])),
                    (int(pair[1]), int(pair[0])),
                }
                observed_directions = {
                    (int(v[0]), int(v[1])) for v in values
                }
                if len(values) != 2 or observed_directions != expected_directions:
                    failures.append("FAIL_DIRECTED_BOND_DUPLICATION")
                attrs = {(v[2], v[3]) for v in values}
                if len(attrs) != 1:
                    failures.append("FAIL_DIRECTED_BOND_ATTRIBUTES")
                pair_attr[pair] = values[0][2:]

            expected_internal = set()
            expected_heavy_internal = set()
            for pair in source_info["bonds"]:
                for ru in (-1, 0, 1):
                    if pair[0] in base_global[ru] and pair[1] in base_global[ru]:
                        global_pair = tuple(sorted((base_global[ru][pair[0]], base_global[ru][pair[1]])))
                        expected_internal.add(global_pair)
                        if source_info["atoms"][pair[0]]["z"] > 1 and source_info["atoms"][pair[1]]["z"] > 1:
                            expected_heavy_internal.add(global_pair)
            actual_internal = {
                pair for pair in pair_attr
                if arrays["is_source"][pair[0]] and arrays["is_source"][pair[1]]
                and arrays["offset"][pair[0]] == arrays["offset"][pair[1]]
            }
            actual_heavy_internal = {
                pair for pair in actual_internal
                if arrays["atomic"][pair[0]] > 1 and arrays["atomic"][pair[1]] > 1
            }
            if actual_internal != expected_internal:
                failures.append("FAIL_CONNECTIVITY")
            if actual_heavy_internal != expected_heavy_internal:
                failures.append("FAIL_HEAVY_CONNECTIVITY")
            flags["heavy_connectivity_ok"] = (
                "FAIL_CONNECTIVITY" not in failures
                and "FAIL_HEAVY_CONNECTIVITY" not in failures
            )

            dummy_base = [item["neighbor_base"] for item in source_info["dummy"]]
            dummy_codes = [item["code"] for item in source_info["dummy"]]
            connection_code = 1 if dummy_codes[0] != dummy_codes[1] or 4 in dummy_codes else dummy_codes[0]
            expected_cross = {
                tuple(sorted((base_global[-1][dummy_base[1]], base_global[0][dummy_base[0]]))),
                tuple(sorted((base_global[0][dummy_base[1]], base_global[1][dummy_base[0]]))),
            }
            actual_cross = {
                pair for pair in pair_attr
                if arrays["offset"][pair[0]] != arrays["offset"][pair[1]]
            }
            if actual_cross != expected_cross:
                failures.append("FAIL_CROSS_RU_BOND")
            for pair in actual_cross:
                if arrays["atomic"][pair[0]] <= 1 or arrays["atomic"][pair[1]] <= 1:
                    failures.append("FAIL_CROSS_RU_ENDPOINT")
                if pair_attr.get(pair, (None, None))[0] != connection_code:
                    failures.append("FAIL_CROSS_RU_BOND_TYPE")
            allowed_pairs = expected_internal | expected_cross
            h_pairs = {
                pair for pair in pair_attr
                if arrays["atomic"][pair[0]] == 1 or arrays["atomic"][pair[1]] == 1
            }
            for pair in h_pairs:
                hydrogen = pair[0] if arrays["atomic"][pair[0]] == 1 else pair[1]
                parent = int(arrays["parents"][hydrogen])
                if parent < 0 or tuple(sorted((hydrogen, parent))) != pair:
                    failures.append("FAIL_H_PARENT_BOND")
            if set(pair_attr) - allowed_pairs - h_pairs:
                failures.append("FAIL_EXTRA_BOND")

            raw_equal = []
            normalized_equal = []
            expected_attrs = source_info["bonds"]
            for ru in (-1, 0, 1):
                observed = {}
                for pair in expected_attrs:
                    global_pair = tuple(sorted((base_global[ru][pair[0]], base_global[ru][pair[1]])))
                    if global_pair not in pair_attr:
                        continue
                    code, aromatic = pair_attr[global_pair]
                    observed[pair] = {"code": int(code), "aromatic": bool(aromatic)}
                raw_equal.append(len(observed) == len(expected_attrs) and all(observed[p] == expected_attrs[p] for p in expected_attrs))
                try:
                    expected_smiles = _fragment_kekule_smiles(source_info, expected_attrs)
                    observed_smiles = _fragment_kekule_smiles(source_info, observed)
                    normalized_equal.append(expected_smiles == observed_smiles)
                except AuditRecordError:
                    normalized_equal.append(False)
            diagnostics["raw_bondtype_equal"] = raw_equal
            diagnostics["chemically_normalized_equal"] = normalized_equal
            diagnostics["representation_only_difference"] = [
                bool(not raw and norm) for raw, norm in zip(raw_equal, normalized_equal)
            ]
            flags["aromatic_equivalence_ok"] = bool(all(normalized_equal))
            if not all(normalized_equal):
                failures.append("FAIL_CHEMICAL_BOND_NORMALIZATION")

            # Formal charge, valence and H counts are checked independently of
            # aromatic/Kekulé representation.
            h_counts = Counter()
            for index in np.flatnonzero(arrays["atomic"] == 1):
                parent = int(arrays["parents"][index])
                if parent >= 0:
                    h_counts[parent] += 1
            h_ok = True
            valence_ok = True
            seam_hybridization_changes = []
            # ``Atom.GetTotalValence`` on the source P-SMILES includes the
            # virtual dummy attachment bond.  The finite Trimer instead has
            # either a real cross-RU seam or the declared terminal H-cap;
            # mismatched/aromatic attachments deliberately use the resolved
            # connection policy.  Build the expected finite valence from the
            # source's internal bonds plus those finite seam/cap contributions.
            source_dummy_valence = Counter()
            for item in source_info["dummy"]:
                source_dummy_valence[item["neighbor_base"]] += _bond_order(
                    item["code"], item["aromatic"]
                )
            finite_valence = {}
            for ru in (-1, 0, 1):
                finite_valence[ru] = {}
                for base_id, atom_meta in enumerate(source_info["atoms"]):
                    finite_valence[ru][base_id] = float(atom_meta["total_valence"])
                    finite_valence[ru][base_id] -= float(source_dummy_valence[base_id])
                    # Every source attachment is represented by exactly one
                    # finite seam or terminal cap in each RU copy.  This also
                    # handles a shared boundary atom (two contributions).
                    finite_valence[ru][base_id] += float(
                        connection_code
                        * sum(item["neighbor_base"] == base_id
                              for item in source_info["dummy"])
                    )
            try:
                payload_mol = _reconstruct_payload_mol(arrays, pair_attr)
            except AuditRecordError:
                payload_mol = None
                failures.append("FAIL_VALENCE")
                valence_ok = False
            for ru in (-1, 0, 1):
                for base_id, atom_meta in enumerate(source_info["atoms"]):
                    global_index = base_global[ru][base_id]
                    expected_h = int(atom_meta["total_h"])
                    if ru == -1 and base_id == dummy_base[0]:
                        expected_h += int(connection_code)
                    if ru == 1 and base_id == dummy_base[1]:
                        expected_h += int(connection_code)
                    if int(h_counts[global_index]) != expected_h:
                        h_ok = False
                    expected_valence = float(finite_valence[ru][base_id])
                    if payload_mol is not None:
                        observed_atom = payload_mol.GetAtomWithIdx(global_index)
                        observed_valence = float(observed_atom.GetTotalValence())
                        if (math.isfinite(expected_valence)
                                and abs(observed_valence - expected_valence) > 0.51):
                            valence_ok = False
                        expected_hybridization = source_info["molecule"].GetAtomWithIdx(
                            source_info["base_indices"][base_id]
                        ).GetHybridization()
                        if (expected_hybridization != Chem.HybridizationType.UNSPECIFIED
                                and observed_atom.GetHybridization() != expected_hybridization):
                            change = {
                                "ru_offset": int(ru),
                                "base_atom": int(base_id),
                                "source": str(expected_hybridization),
                                "observed": str(observed_atom.GetHybridization()),
                            }
                            if base_id in dummy_base:
                                # A seam atom can legitimately change local
                                # hybridization when the virtual dummy is
                                # replaced by a real neighbouring RU atom
                                # (for example an ester O).  Keep this visible
                                # as a seam diagnostic; it is not an internal
                                # RU chemistry failure.  Non-seam changes are
                                # still fatal below.
                                seam_hybridization_changes.append(change)
                            else:
                                failures.append("FAIL_HYBRIDIZATION_CHEMISTRY")
                                valence_ok = False
            diagnostics["seam_hybridization_changes"] = seam_hybridization_changes
            if not h_ok:
                failures.append("FAIL_UNEXPECTED_H")
            if not valence_ok:
                failures.append("FAIL_VALENCE")
            flags["valence_ok"] = valence_ok
            # Added-H parents must be heavy and remain in the same RU.
            for index in np.flatnonzero(arrays["is_added_h"]):
                parent = int(arrays["parents"][index])
                if parent < 0 or parent >= n_all or arrays["atomic"][parent] <= 1 or arrays["offset"][parent] != arrays["offset"][index]:
                    failures.append("FAIL_H_PARENT_IDENTITY")
            # Center is the non-terminal copy; terminal chemistry is allowed
            # only through the declared seam cap and otherwise uses the same
            # independently normalized source graph.
            flags["center_ru_chemistry_ok"] = bool(normalized_equal[1] and h_ok and valence_ok)
            flags["terminal_cap_ok"] = bool(all(normalized_equal) and h_ok and valence_ok)
            if not flags["center_ru_chemistry_ok"]:
                failures.append("FAIL_CENTER_RU_CHEMISTRY")
            if not flags["terminal_cap_ok"]:
                failures.append("FAIL_TERMINAL_CAP")

            bond_table_failures = {
                "FAIL_BOND_TABLE_SHAPE", "FAIL_BOND_ENDPOINT",
                "FAIL_DIRECTED_BOND_DUPLICATION", "FAIL_DIRECTED_BOND_ATTRIBUTES",
                "FAIL_CONNECTIVITY", "FAIL_HEAVY_CONNECTIVITY",
                "FAIL_CROSS_RU_BOND", "FAIL_CROSS_RU_ENDPOINT",
                "FAIL_CROSS_RU_BOND_TYPE", "FAIL_H_PARENT_BOND",
                "FAIL_EXTRA_BOND",
            }
            flags["bond_table_ok"] = not any(
                code in bond_table_failures for code in failures
            )

            source_aromatic = np.asarray([a["aromatic"] for a in source_info["atoms"]])
            target_aromatic = arrays["aromatic_atom"]
            for ru in (-1, 0, 1):
                indices = np.asarray([base_global[ru][i] for i in range(base_count)])
                diff = np.flatnonzero(target_aromatic[indices] != source_aromatic)
                if diff.size and not normalized_equal[ru]:
                    failures.append("FAIL_HYBRIDIZATION_CHEMISTRY")
            # Explicit source identity includes isotope/formal charge and all
            # source atom flags; chiral tags are compared where declared.
            for ru in (-1, 0, 1):
                indices = np.asarray([base_global[ru][i] for i in range(base_count)])
                if np.any(arrays["charge"][indices] != expected_charge):
                    failures.append("FAIL_FORMAL_CHARGE")

            mapping = arrays["o8_map"]
            topology_count = int(_scalar(topology, "num_nodes"))
            if mapping.size != topology_count or mapping.size != base_count or np.any(mapping < 0) or np.any(mapping >= n_all):
                failures.append("FAIL_O8_MAPPING")
            else:
                expected_mapping = np.asarray([base_global[0][i] for i in range(base_count)])
                if not np.array_equal(mapping, expected_mapping):
                    failures.append("FAIL_O8_MAPPING")
                expected_o8_mask = arrays["atomic"][mapping] > 1
                if not np.array_equal(arrays["o8_heavy_mask"], expected_o8_mask):
                    failures.append("FAIL_O8_HEAVY_MASK")
                if not np.array_equal(arrays["o8_heavy_indices"], np.flatnonzero(expected_o8_mask)):
                    failures.append("FAIL_O8_HEAVY_INDICES")
            flags["o8_mapping_ok"] = not any(code.startswith("FAIL_O8") for code in failures)

            # Star is diagnostic only; a false value is deliberately not a
            # failure in this audit.
            diagnostics["star_3d_valid"] = bool(_bool_scalar(trimer, "star_3d_valid"))
            diagnostics["star_3d_distance"] = _finite_float(_scalar(trimer, "star_3d_distance"))
            diagnostics["star_3d_asymmetry"] = _finite_float(_scalar(trimer, "star_3d_asymmetry"))
            diagnostics["zero_distance_duplicate_atoms"] = int(
                n_all - np.unique(np.round(arrays["pos"], decimals=6), axis=0).shape[0]
            ) if flags["finite_geometry_ok"] else None
            if flags["finite_geometry_ok"] and pair_attr:
                distances = [
                    float(np.linalg.norm(arrays["pos"][a] - arrays["pos"][b]))
                    for a, b in pair_attr
                ]
                diagnostics["bond_distance_min"] = _finite_float(min(distances))
                diagnostics["bond_distance_max"] = _finite_float(max(distances))
                diagnostics["extreme_bond_count"] = int(sum(d < 0.2 or d > 5.0 for d in distances))

            dummy_indices = [item["index"] for item in source_info["dummy"]]
            stereo_result = _audit_stereo(
                source_info, arrays, base_global, dummy_indices
            )
            diagnostics["stereo"] = stereo_result
            if stereo_result["status"] == "FAIL":
                failures.append("FAIL_STEREO")
                flags["stereo_ok"] = False
            elif stereo_result["status"] == "UNRESOLVED":
                unresolved.append("STEREO_TERMINAL_REFERENCE_UNRESOLVED")
                flags["stereo_ok"] = None
            else:
                flags["stereo_ok"] = stereo_result["status"] == "PASS" or stereo_result["status"] == "N/A"
        flags["record_complete"] = not any(code.startswith("FAIL_") for code in failures)
    except AuditRecordError as exc:
        unresolved.append("AUDIT_UNRESOLVED:" + str(exc))
    except Exception as exc:  # audit records never terminate the full scan
        unresolved.append("AUDIT_SCRIPT_EXCEPTION:" + type(exc).__name__ + ":" + str(exc)[:180])

    if failures:
        status = "AUDIT_FAIL"
    elif unresolved:
        status = "AUDIT_UNRESOLVED"
    else:
        status = "AUDIT_PASS"
    return {
        "position": int(position),
        "sample_key": str(key_hex),
        "source_row": int(row.get("source_row", -1)),
        "status": status,
        "failures": sorted(set(failures)),
        "unresolved": unresolved,
        "flags": flags,
        "diagnostics": diagnostics,
        "runtime_status": runtime_row.get("status"),
    }


def _audit_payload_job(arguments):
    """Process-pool entry point; all inputs are immutable snapshot bytes."""

    key_hex, row, top_payload, tri_payload, runtime_row, position = arguments
    try:
        key = bytes.fromhex(key_hex)
        runtime_row = dict(runtime_row)
        runtime_row["_payload_bytes"] = len(tri_payload)
        topology = deserialize_record(bytes(top_payload), key)
        trimer = deserialize_record(bytes(tri_payload), key)
        return _audit_record(
            key_hex, row, topology, trimer, runtime_row, position
        )
    except Exception as exc:
        return {
            "position": int(position),
            "sample_key": str(key_hex),
            "source_row": int(row.get("source_row", -1)) if row else -1,
            "status": "AUDIT_UNRESOLVED",
            "failures": [],
            "unresolved": [
                "AUDIT_WORKER_EXCEPTION:" + type(exc).__name__ + ":"
                + str(exc)[:180]
            ],
            "flags": {"record_complete": False},
            "diagnostics": {},
            "runtime_status": runtime_row.get("status"),
        }


def _audit_worker_init():
    """Keep each conservative audit worker single-threaded."""

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _build_status():
    try:
        output = subprocess.run(
            ["pgrep", "-af", "build_full_mts_cache|build_mts_cache"],
            capture_output=True, text=True, check=False,
        ).stdout.strip().splitlines()
    except Exception:
        output = []
    return {"active": bool(output), "processes": output[:8]}


def _atomic_json(path: Path, value):
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    snapshot_dir = args.snapshot_dir.resolve()
    output_dir = (args.output_dir or snapshot_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
    keys = []
    for line in (snapshot_dir / "accepted_keys.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            keys.append(json.loads(line))
    rows = {}
    for line in (snapshot_dir / "source_rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sample_key"]] = row
    start_status = _build_status()
    staging = Path(snapshot["staging_path"])
    if not staging.exists():
        candidate = staging.parent / snapshot["staging_bundle_hash"]
        if candidate.exists():
            staging = candidate
    tri_root = staging / "trimer" / "data.lmdb"
    top_root = staging / "topology" / "data.lmdb"
    if not tri_root.is_dir() or not top_root.is_dir():
        report = {
            "schema": "trimer-live-accepted-audit-v1",
            "status": "SCRIPT_ERROR",
            "error": "snapshot LMDB path is unavailable",
            "snapshot": snapshot,
            "build_status_at_start": start_status,
        }
        _atomic_json(output_dir / "summary.json", report)
        return 2

    accepted_runtime = {item["sample_key"]: item for item in keys}
    per_sample_path = output_dir / "per_sample_audit.jsonl"
    failure_path = output_dir / "failures.jsonl"
    per_sample = per_sample_path.open("w", encoding="utf-8")
    failures_file = failure_path.open("w", encoding="utf-8")
    statuses = Counter()
    failure_counts = Counter()
    high_risk = {
        name: Counter() for name in (
            "double_seam", "single_seam", "aromatic", "fused_aromatic",
            "hetero_aromatic", "charged_aromatic",
        )
    }
    star_counts = Counter()
    representation_only = 0
    stereo_declared = stereo_checked = stereo_failed = 0
    audited = 0
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        # The live builder grows the LMDB map while this read-only audit is
        # running.  A generous virtual map avoids stale-map ``MDB_MAP_RESIZED``
        # failures without allocating or writing that address space.
        read_map_size = 1 << 40
        # Four independent CPU workers keep the audit conservative while
        # avoiding a many-hour single-process scan.  The LMDB handles remain
        # in the parent; workers receive only immutable payload bytes.
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=4, initializer=_audit_worker_init
        )
        tri_env = lmdb.open(
            str(tri_root), subdir=True, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=512,
            map_size=read_map_size,
        )
        top_env = lmdb.open(
            str(top_root), subdir=True, readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=512,
            map_size=read_map_size,
        )
        def emit(result):
            nonlocal audited, representation_only
            nonlocal stereo_declared, stereo_checked, stereo_failed
            per_sample.write(
                json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
            )
            if result["status"] != "AUDIT_PASS":
                failures_file.write(
                    json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
                )
            audited += 1
            statuses[result["status"]] += 1
            for code in result.get("failures", []):
                failure_counts[code] += 1
            diagnostics = result.get("diagnostics", {})
            if any(diagnostics.get("representation_only_difference", [])):
                representation_only += 1
            stereo = diagnostics.get("stereo", {})
            stereo_declared += int(stereo.get("declared_bonds", 0) or 0)
            stereo_checked += int(stereo.get("checked", 0) or 0)
            stereo_failed += int(bool(stereo.get("failures")))
            if "star_3d_valid" in diagnostics:
                star_counts[
                    "true" if diagnostics["star_3d_valid"] else "false"
                ] += 1
            groups = diagnostics.get("high_risk_groups", {})
            for name in high_risk:
                if groups.get(name):
                    high_risk[name][result["status"]] += 1
            if audited % 1000 == 0:
                per_sample.flush()
                failures_file.flush()

        def reader_failure(position, key_hex, row, runtime_row, exc):
            return {
                "position": int(position),
                "sample_key": str(key_hex),
                "source_row": int(row.get("source_row", -1)) if row else -1,
                "status": "AUDIT_UNRESOLVED",
                "failures": [],
                "unresolved": [
                    "AUDIT_READER_EXCEPTION:" + type(exc).__name__ + ":"
                    + str(exc)[:180]
                ],
                "flags": {"record_complete": False},
                "diagnostics": {},
                "runtime_status": runtime_row.get("status"),
            }

        pending = {}
        max_inflight = 32

        def drain_one():
            if not pending:
                return
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                position, key_hex, row, runtime_row = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = reader_failure(
                        position, key_hex, row, runtime_row, exc
                    )
                emit(result)

        def emit_batch(batch):
            for position, key_hex, row, runtime_row, top_payload, tri_payload in batch:
                if row is None:
                    emit(reader_failure(
                        position, key_hex, row, runtime_row,
                        AuditRecordError("accepted snapshot source row missing"),
                    ))
                    continue
                while len(pending) >= max_inflight:
                    drain_one()
                future = executor.submit(
                    _audit_payload_job,
                    (
                        key_hex,
                        row,
                        top_payload,
                        tri_payload,
                        dict(runtime_row),
                        position,
                    ),
                )
                pending[future] = (position, key_hex, row, runtime_row)

        try:
            batch_size = 64
            for batch_start in range(0, len(keys), batch_size):
                batch_end = min(batch_start + batch_size, len(keys))
                batch = []
                read_error = None
                # A short transaction prevents a live map resize from
                # invalidating the entire 626k-record scan.  If the writer
                # resizes between batches, reopen the read handles and retry
                # the same fixed key slice before recording an unresolved row.
                for attempt in range(5):
                    try:
                        with tri_env.begin(buffers=False) as tri_txn, top_env.begin(buffers=False) as top_txn:
                            batch = []
                            for position in range(batch_start, batch_end):
                                runtime_row = keys[position]
                                key_hex = runtime_row["sample_key"]
                                key = bytes.fromhex(key_hex)
                                tri_payload = tri_txn.get(key)
                                top_payload = top_txn.get(key)
                                if tri_payload is None or top_payload is None:
                                    raise AuditRecordError(
                                        "accepted snapshot key missing from LMDB"
                                    )
                                batch.append((
                                    position,
                                    key_hex,
                                    rows.get(key_hex),
                                    runtime_row,
                                    bytes(top_payload),
                                    bytes(tri_payload),
                                ))
                        read_error = None
                        break
                    except lmdb.Error as exc:
                        read_error = exc
                        time.sleep(min(1.0, 0.1 * (attempt + 1)))
                        try:
                            tri_env.close()
                            top_env.close()
                        except Exception:
                            pass
                        tri_env = lmdb.open(
                            str(tri_root), subdir=True, readonly=True, lock=False,
                            readahead=False, meminit=False, max_readers=512,
                            map_size=read_map_size,
                        )
                        top_env = lmdb.open(
                            str(top_root), subdir=True, readonly=True, lock=False,
                            readahead=False, meminit=False, max_readers=512,
                            map_size=read_map_size,
                        )
                    except Exception as exc:
                        read_error = exc
                        break
                if read_error is not None:
                    for position in range(batch_start, batch_end):
                        runtime_row = keys[position]
                        key_hex = runtime_row["sample_key"]
                        emit(reader_failure(
                            position, key_hex, rows.get(key_hex), runtime_row,
                            read_error,
                        ))
                else:
                    emit_batch(batch)
            while pending:
                drain_one()
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
            tri_env.close()
            top_env.close()
    except Exception as exc:
        script_error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    else:
        script_error = None
    finally:
        per_sample.close(); failures_file.close()

    known_controls = {
        "12282435adcb0b02dbd3591075b6bf36423d8e36f5c5d11c2922e4b19cd38a13",
        "0d7521a0eae9c5825ad54c6c3135f2771d9684e9b38a3bedf61bba4eef9c5836",
        "87fa818c0670b04a9ae4ae56aa2b9d6dcee4a28ba5e44488a48adb426b3c0b58",
        "50feb61abd4f2263af5234a94c3e41ebf332cbc06300e295c93fe0d443596a97",
    }
    control_rows = []
    rejection_path = staging / "trimer" / "rejections.jsonl"
    if rejection_path.is_file():
        for line in rejection_path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            item = json.loads(line)
            if item.get("sample_key") in known_controls:
                control_rows.append({"sample_key": item["sample_key"], "in_snapshot": item["sample_key"] in accepted_runtime,
                                     "failure_code": item.get("failure_code"), "exception_type": item.get("exception_type"),
                                     "exception_message": item.get("exception_message"), "record_bytes": 0})
    end_status = _build_status()
    summary = {
        "schema": "trimer-live-accepted-audit-v1",
        "status": "SCRIPT_ERROR" if script_error else "COMPLETE",
        "script_error": script_error,
        "snapshot": snapshot,
        "audit_with_build_running": bool(start_status["active"]),
        "build_status_at_start": start_status,
        "build_status_at_end": end_status,
        "resolved_lmdb_root": str(staging.resolve()),
        "snapshot_accepted_count": len(keys),
        "audited_count": audited,
        "record_completeness": {"audited_equals_snapshot": audited == len(keys)},
        "status_counts": dict(statuses),
        "failure_counts": dict(failure_counts),
        "representation_only_difference_count": representation_only,
        "stereo": {"declared_bonds": stereo_declared, "checked": stereo_checked, "samples_with_failure": stereo_failed},
        "star_3d_valid_counts": dict(star_counts),
        "high_risk_subgroups": {name: dict(counter) for name, counter in high_risk.items()},
        "known_four_rejected_controls": control_rows,
        "elapsed_seconds": time.monotonic() - started,
        "AUDIT_PASS": int(statuses["AUDIT_PASS"]),
        "AUDIT_FAIL": int(statuses["AUDIT_FAIL"]),
        "AUDIT_UNRESOLVED": int(statuses["AUDIT_UNRESOLVED"]),
        "CURRENT_BUILD_STOPPED_BY_AUDIT": False,
        "CURRENT_BUILD_MODIFIED_BY_AUDIT": False,
        "training_ready_recommendation": "UNDECIDED_UNTIL_AUDIT_SUMMARY_REVIEW",
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), "audited_count": audited, "status_counts": dict(statuses), "script_error": script_error}, sort_keys=True))
    return 2 if script_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
