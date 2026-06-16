import os

from .common import normalize_alias, read_csv, stable_id, write_csv, write_jsonl


ALIAS_FIELDS = [
    "alias_id", "alias_text", "normalized_alias", "target_id", "target_type",
    "mapping_method", "confidence", "status",
]


def heuristic_candidate(unit):
    hint = unit.get("polymer_origin_hint", "unknown")
    candidate = {
        "polymer_class_id": "pc_unknown",
        "canonical_name": "unknown",
        "polymer_family": "unknown",
        "aliases": [],
        "composition_type": "unknown",
    }
    if hint == "possible_condensation_polymer":
        candidate.update(
            polymer_class_id="pc_condensation_polymer_unknown",
            canonical_name="unknown condensation polymer",
            polymer_family="condensation_polymer",
            composition_type="condensation_multi_monomer",
        )
    return candidate


def write_mapping_outputs(repeat_units_path, output_dir, mapped_rows, mapping_method, source_type):
    units = {row["repeat_unit_id"]: row for row in read_csv(repeat_units_path)}
    output_rows = []
    aliases = []
    for mapped in mapped_rows:
        unit = units[mapped["repeat_unit_id"]]
        candidates = mapped.get("polymer_class_candidates") or [heuristic_candidate(unit)]
        confidence = min(float(mapped.get("confidence", 0.25)), 0.70 if source_type == "llm_temporary_mapping" else 0.40)
        warnings = list(mapped.get("warnings", []))
        if source_type == "llm_temporary_mapping":
            warnings.append("Temporary LLM mapping for pipeline testing; not a verified registry mapping.")
        output_rows.append({
            "repeat_unit_id": unit["repeat_unit_id"],
            "raw_smiles": unit["raw_smiles"],
            "canonical_smiles": unit["canonical_smiles"],
            "polymer_class_candidates": candidates,
            "mapping_method": mapping_method,
            "source_type": source_type,
            "confidence": confidence,
            "warnings": sorted(set(warnings)),
        })
        for candidate in candidates:
            target_id = candidate.get("polymer_class_id", "pc_unknown")
            names = [candidate.get("canonical_name")] + list(candidate.get("aliases") or [])
            for name in filter(None, names):
                aliases.append({
                    "alias_id": stable_id("alias", name, target_id),
                    "alias_text": name,
                    "normalized_alias": normalize_alias(name),
                    "target_id": target_id,
                    "target_type": "PolymerClass",
                    "mapping_method": mapping_method,
                    "confidence": confidence,
                    "status": "review" if source_type == "llm_temporary_mapping" else "candidate",
                })
    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(f"{output_dir}/polymer_class_candidates.jsonl", output_rows)
    write_csv(f"{output_dir}/entity_aliases.csv", ALIAS_FIELDS, aliases)
    return len(output_rows)


def generate_rule_stub(repeat_units_path, output_dir):
    rows = []
    for unit in read_csv(repeat_units_path):
        rows.append({
            "repeat_unit_id": unit["repeat_unit_id"],
            "polymer_class_candidates": [heuristic_candidate(unit)],
            "confidence": 0.20,
            "warnings": ["Heuristic offline stub; requires registry or manual review."],
        })
    return write_mapping_outputs(repeat_units_path, output_dir, rows, "rule_stub", "heuristic_stub")
