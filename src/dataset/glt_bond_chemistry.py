"""Shared, geometry-independent bond chemistry for O8 and complete Trimer."""

import numpy as np
from rdkit import Chem

BOND_FEATURE_DIM = 14
STEREO_VALUES = (
    Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
)
STEREO_TO_INDEX = {value: index for index, value in enumerate(STEREO_VALUES)}
STEREO_UNKNOWN = len(STEREO_VALUES)
NUM_STEREO_TYPES = STEREO_UNKNOWN + 1


def bond_type_index(bond):
    return {
        Chem.rdchem.BondType.SINGLE: 0, Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2, Chem.rdchem.BondType.AROMATIC: 3,
    }.get(bond.GetBondType(), 4)


def bond_stereo_index(bond):
    return STEREO_TO_INDEX.get(bond.GetStereo(), STEREO_UNKNOWN)


def bond_feature_vector(bond):
    result = np.zeros(BOND_FEATURE_DIM, dtype=np.float32)
    result[bond_type_index(bond)] = 1
    result[5] = float(bond.GetIsConjugated())
    result[6] = float(bond.IsInRing())
    result[7 + bond_stereo_index(bond)] = 1
    return result
