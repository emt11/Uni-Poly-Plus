import re

from .common import stable_hash, stable_id


def canonicalize_repeat_unit(smiles):
    raw = str(smiles or "").strip()
    result = {
        "raw_smiles": raw,
        "canonical_smiles": raw,
        "valid_rdkit_parse": False,
        "formula": "",
        "molecular_weight_M0": "",
        "repeat_unit_representation_type": "attachment_point_smiles" if "*" in raw else "molecular_smiles",
        "polymer_origin_hint": "unknown",
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return result
        result.update(
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            valid_rdkit_parse=True,
            formula=rdMolDescriptors.CalcMolFormula(mol),
            molecular_weight_M0=f"{Descriptors.MolWt(mol):.6f}",
        )
    except (ImportError, RuntimeError, ValueError):
        pass
    result["polymer_origin_hint"] = infer_origin_hint(result["canonical_smiles"])
    return result


def infer_origin_hint(smiles):
    text = str(smiles)
    hetero_linkages = len(re.findall(r"C\(=O\)[ON]|[ON]C\(=O\)|C\(=O\)O|OC\(=O\)", text))
    if hetero_linkages:
        return "possible_condensation_polymer"
    if "*" in text and re.search(r"C=C|\[C@?H?\]", text):
        return "possible_chain_growth_polymer"
    return "unknown"


def repeat_unit_identity(smiles):
    normalized = canonicalize_repeat_unit(smiles)
    identity_text = normalized["canonical_smiles"] or normalized["raw_smiles"]
    normalized["structure_hash"] = stable_hash(identity_text, length=24)
    normalized["repeat_unit_id"] = stable_id("ru", identity_text, length=20)
    return normalized

