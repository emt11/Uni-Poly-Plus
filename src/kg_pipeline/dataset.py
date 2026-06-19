"""Dataset record and repeat-unit preparation for Polymer KG."""

from .io_utils import stable_hash, stable_id


def canonicalize_repeat_unit(smiles):
    raw = str(smiles or "").strip()
    result = {
        "raw_smiles": raw,
        "canonical_smiles": raw,
        "valid_rdkit_parse": False,
        "formula": "",
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return result
        result.update(
            canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
            valid_rdkit_parse=True,
            formula=rdMolDescriptors.CalcMolFormula(mol),
        )
    except (ImportError, RuntimeError, ValueError):
        pass
    return result


def repeat_unit_identity(smiles):
    normalized = canonicalize_repeat_unit(smiles)
    identity_text = normalized["canonical_smiles"] or normalized["raw_smiles"]
    normalized["structure_hash"] = stable_hash(identity_text, length=24)
    normalized["repeat_unit_id"] = stable_id("ru", identity_text, length=20)
    return normalized


from collections import defaultdict

from .io_utils import read_csv, stable_id, write_csv


RECORD_FIELDS = ["record_id", "row_index", "smiles", "prop", "repeat_unit_id"]
REPEAT_UNIT_FIELDS = [
    "repeat_unit_id", "raw_smiles", "canonical_smiles", "structure_hash",
    "valid_rdkit_parse", "formula", "num_records",
    "props",
]


def prepare_records(input_path, output_dir, max_repeat_units=None):
    input_rows = read_csv(input_path)
    required = {"smiles", "prop"}
    if not input_rows or not required.issubset(input_rows[0]):
        raise ValueError(f"{input_path} must contain smiles and prop columns")

    records = []
    units = {}
    props = defaultdict(set)
    counts = defaultdict(int)
    allowed = set()
    for row_index, row in enumerate(input_rows):
        smiles = row.get("smiles", "").strip()
        prop = row.get("prop", "").strip()
        unit = repeat_unit_identity(smiles)
        repeat_unit_id = unit["repeat_unit_id"]
        if max_repeat_units is not None and repeat_unit_id not in allowed:
            if len(allowed) >= max_repeat_units:
                continue
            allowed.add(repeat_unit_id)
        units.setdefault(repeat_unit_id, unit)
        props[repeat_unit_id].add(prop)
        counts[repeat_unit_id] += 1
        records.append({
            "record_id": stable_id("dr", input_path, row_index, smiles, prop, length=20),
            "row_index": row_index,
            "smiles": smiles,
            "prop": prop,
            "repeat_unit_id": repeat_unit_id,
        })

    repeat_units = []
    for repeat_unit_id in sorted(units):
        unit = dict(units[repeat_unit_id])
        unit["valid_rdkit_parse"] = str(bool(unit["valid_rdkit_parse"])).lower()
        unit["num_records"] = counts[repeat_unit_id]
        unit["props"] = ";".join(sorted(props[repeat_unit_id]))
        repeat_units.append(unit)

    write_csv(f"{output_dir}/records.csv", RECORD_FIELDS, records)
    write_csv(f"{output_dir}/repeat_units.csv", REPEAT_UNIT_FIELDS, repeat_units)
    return {"records": len(records), "repeat_units": len(repeat_units)}
