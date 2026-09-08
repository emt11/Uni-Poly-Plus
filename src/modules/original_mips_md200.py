"""Original MIPS MD200 descriptor path.

This module is intentionally separate from the current MTS descriptor attach
path.  The scientific definition is the small preprocessing function in the
fixed ``wjxts/MIPS`` commit: substitute the two pSMILES dummy atoms with the
opposite connector atom number, serialize with ``Chem.MolToSmiles`` and pass
that string to ``RDKit2DNormalized``.  No ring closure, repeated-unit or
geometry operation belongs here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Iterable

import numpy as np
from rdkit import Chem

from src.dataset.mips_descriptors.rdNormalizedDescriptors import RDKit2DNormalized


MD_DIM = 200


def psmiles_star_sub(smiles: str) -> str:
    """Return the exact Original-MIPS connector substitution serialization."""

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse pSMILES: {smiles!r}")
    star_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"
    ]
    if len(star_indices) != 2:
        raise ValueError(
            f"Original MIPS MD200 expects exactly two '*' atoms, got {len(star_indices)}"
        )
    connected_atoms = []
    for wildcard in star_indices:
        neighbors = list(mol.GetAtomWithIdx(wildcard).GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError("Original MIPS wildcard must have exactly one neighbor")
        connected_atoms.append(neighbors[0].GetIdx())
    editable = Chem.RWMol(mol)
    editable.GetAtomWithIdx(star_indices[0]).SetAtomicNum(
        editable.GetAtomWithIdx(connected_atoms[1]).GetAtomicNum()
    )
    editable.GetAtomWithIdx(star_indices[1]).SetAtomicNum(
        editable.GetAtomWithIdx(connected_atoms[0]).GetAtomicNum()
    )
    new_mol = editable.GetMol()
    return Chem.MolToSmiles(new_mol)


@dataclass
class MD200Stats:
    md_none_count: int = 0
    md_nan_sample_count: int = 0
    md_nan_value_count: int = 0
    md_zero_fallback_count: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "md_none_count": int(self.md_none_count),
            "md_nan_sample_count": int(self.md_nan_sample_count),
            "md_nan_value_count": int(self.md_nan_value_count),
            "md_zero_fallback_count": int(self.md_zero_fallback_count),
        }


class OriginalMIPSMD200:
    """Compute MD200 while retaining the official failure semantics."""

    protocol = (
        "pSMILES -> psmiles_star_sub -> RDKit2DNormalized.process -> "
        "discard return[0] -> last 200 dimensions"
    )

    def __init__(self, generator: RDKit2DNormalized | None = None) -> None:
        self.generator = generator or RDKit2DNormalized()
        self._lock = Lock()
        self._stats = MD200Stats()

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = MD200Stats()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return self._stats.as_dict()

    def _update(self, **increments: int) -> None:
        with self._lock:
            for name, value in increments.items():
                setattr(self._stats, name, getattr(self._stats, name) + int(value))

    def transform(self, smiles: str) -> str:
        return psmiles_star_sub(smiles)

    def one(self, smiles: str) -> np.ndarray:
        transformed = psmiles_star_sub(smiles)
        raw = self.generator.process(transformed)
        if raw is None:
            # Official ``mol_descriptor`` returns np.zeros(200) on None.  The
            # dataset subsequently applies np.nan_to_num to the stacked array.
            self._update(md_none_count=1, md_zero_fallback_count=1)
            return np.zeros((MD_DIM,), dtype=np.float64)
        values = np.asarray(raw[1:])
        if values.shape != (MD_DIM,):
            raise ValueError(
                f"RDKit2DNormalized returned shape {values.shape}, expected {(MD_DIM,)}"
            )
        nan_mask = np.isnan(values)
        nan_values = int(nan_mask.sum())
        if nan_values:
            self._update(md_nan_sample_count=1, md_nan_value_count=nan_values)
            # This is exactly the official dataset-level operation.  Do not
            # use a tolerance or silently drop the sample.
            values = np.nan_to_num(values, nan=0)
        return np.asarray(values)

    def many(self, smiles_list: Iterable[str]) -> tuple[np.ndarray, dict[str, int]]:
        rows = [self.one(value) for value in smiles_list]
        if not rows:
            return np.empty((0, MD_DIM), dtype=np.float64), self.stats()
        return np.stack(rows, axis=0), self.stats()


__all__ = ["MD_DIM", "MD200Stats", "OriginalMIPSMD200", "psmiles_star_sub"]
