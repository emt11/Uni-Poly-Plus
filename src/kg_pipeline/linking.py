import json
import os

from .common import normalize_alias, read_csv, read_jsonl, stable_id, write_jsonl


STRICT_RELATIONS = {"exact_repeat_unit_match", "canonical_smiles_match", "polymer_class_match"}


def _validated_documents(directory):
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json") and name != "review_queue.json":
            with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                yield json.load(handle)


def build_dataset_links(validated_dir, repeat_units_path, candidates_path, output_path, aliases_path=None):
    units = read_csv(repeat_units_path)
    candidates = {row["repeat_unit_id"]: row for row in read_jsonl(candidates_path)}
    by_canonical = {row.get("canonical_smiles"): row for row in units if row.get("canonical_smiles")}
    by_class = {}
    by_alias = {}
    for repeat_unit_id, row in candidates.items():
        for candidate in row.get("polymer_class_candidates", []):
            class_id = candidate.get("polymer_class_id")
            if class_id and class_id != "pc_unknown":
                by_class.setdefault(normalize_alias(class_id.removeprefix("pc_")), []).append((repeat_unit_id, row, candidate))
                for name in [candidate.get("canonical_name")] + list(candidate.get("aliases") or []):
                    if name:
                        by_alias.setdefault(normalize_alias(name), []).append((repeat_unit_id, row, candidate))
    links = []
    for document in _validated_documents(validated_dir):
        for sample in document.get("literature_samples", []):
            matches = []
            explicit_smiles = sample.get("repeat_unit_smiles")
            if explicit_smiles and explicit_smiles in by_canonical:
                unit = by_canonical[explicit_smiles]
                matches.append((unit, "canonical_smiles_match", ["canonical_smiles"], 0.98, "deterministic"))
            class_key = normalize_alias(sample.get("polymer_class"))
            for repeat_unit_id, mapping, _ in by_class.get(class_key, []):
                unit = next(item for item in units if item["repeat_unit_id"] == repeat_unit_id)
                confidence = min(0.85, float(mapping.get("confidence", 0)))
                matches.append((unit, "polymer_class_match", ["polymer_name"], confidence, mapping.get("source_type", "mapping")))
            names = [sample.get("polymer_name")] + list(sample.get("aliases") or [])
            for name in filter(None, names):
                for repeat_unit_id, mapping, _ in by_alias.get(normalize_alias(name), []):
                    unit = next(item for item in units if item["repeat_unit_id"] == repeat_unit_id)
                    confidence = min(0.75, float(mapping.get("confidence", 0)))
                    matches.append((unit, "alias_match", ["alias"], confidence, mapping.get("source_type", "mapping")))
            if not matches:
                continue
            seen = set()
            for unit, relation, matched_on, confidence, source_type in matches:
                key = (sample["sample_id"], unit["repeat_unit_id"], relation)
                if key in seen:
                    continue
                seen.add(key)
                links.append({
                    "link_id": stable_id("link", *key, length=20),
                    "literature_sample_id": sample["sample_id"],
                    "target_repeat_unit": {
                        "repeat_unit_id": unit["repeat_unit_id"],
                        "original_smiles": unit["raw_smiles"],
                        "canonical_smiles": unit["canonical_smiles"],
                    },
                    "relation_type": relation,
                    "matched_on": matched_on,
                    "confidence": confidence,
                    "evidence_refs": sample.get("identity_evidence_refs", []),
                    "source_type": source_type,
                })
    write_jsonl(output_path, links)
    return len(links)

