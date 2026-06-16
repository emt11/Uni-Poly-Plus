import copy
import os
from collections import defaultdict

from .common import stable_id
from .schemas import empty_article_document


FACT_ARRAYS = (
    "composition_assertions", "sequence_distribution_assertions",
    "chain_architecture_assertions", "molecular_weight_measurements",
    "polymerization_events",
)


def _sample_key(sample):
    for field in ("sample_label", "polymer_name", "polymer_class"):
        value = str(sample.get(field) or "").strip().lower()
        if value:
            return f"{field}:{value}"
    aliases = sample.get("aliases") or []
    return f"alias:{str(aliases[0]).lower()}" if aliases else None


def aggregate_article(article_id, documents):
    result = empty_article_document(article_id)
    evidence_by_signature = {}
    sample_groups = defaultdict(list)
    for document in documents:
        if document.get("article"):
            for key, value in document["article"].items():
                if value is not None:
                    result["article"][key] = value
        for evidence in document.get("evidence_records", []):
            signature = (evidence.get("chunk_id"), evidence.get("sentence"))
            evidence_by_signature.setdefault(signature, copy.deepcopy(evidence))
        for sample in document.get("literature_samples", []):
            key = _sample_key(sample) or f"unresolved:{len(sample_groups)}"
            sample_groups[key].append(copy.deepcopy(sample))
        result["warnings"].extend(document.get("warnings", []))

    evidence_id_map = {}
    for index, (signature, evidence) in enumerate(sorted(evidence_by_signature.items(), key=str), 1):
        old_id = evidence.get("evidence_id")
        new_id = stable_id("ev", article_id, signature[0], signature[1], length=20)
        evidence["evidence_id"] = new_id
        if old_id:
            evidence_id_map[old_id] = new_id
        result["evidence_records"].append(evidence)

    for sample_index, (_, samples) in enumerate(sorted(sample_groups.items()), 1):
        merged = {
            "sample_id": f"ls_{article_id}_{sample_index:04d}", "sample_label": None,
            "polymer_name": None, "polymer_class": None, "aliases": [],
            "repeat_unit_smiles": None, "identity_evidence_refs": [],
            **{field: [] for field in FACT_ARRAYS},
        }
        for sample in samples:
            for field in ("sample_label", "polymer_name", "polymer_class", "repeat_unit_smiles"):
                if merged[field] is None and sample.get(field) is not None:
                    merged[field] = sample[field]
            merged["aliases"].extend(sample.get("aliases") or [])
            merged["identity_evidence_refs"].extend(sample.get("identity_evidence_refs") or [])
            for field in FACT_ARRAYS:
                merged[field].extend(sample.get(field) or [])
        merged["aliases"] = sorted(set(merged["aliases"]))
        for field in ("identity_evidence_refs",):
            merged[field] = sorted(set(evidence_id_map.get(ref, ref) for ref in merged[field]))
        for field in FACT_ARRAYS:
            id_key = {
                "composition_assertions": "assertion_id", "sequence_distribution_assertions": "assertion_id",
                "chain_architecture_assertions": "assertion_id", "molecular_weight_measurements": "measurement_id",
                "polymerization_events": "event_id",
            }[field]
            prefix = {"composition_assertions": "comp", "sequence_distribution_assertions": "seq", "chain_architecture_assertions": "arch", "molecular_weight_measurements": "mw", "polymerization_events": "poly"}[field]
            for fact_index, fact in enumerate(merged[field], 1):
                fact[id_key] = stable_id(prefix, merged["sample_id"], field, fact_index, str(fact), length=20)
                fact["evidence_refs"] = sorted(set(evidence_id_map.get(ref, ref) for ref in fact.get("evidence_refs", [])))
        result["literature_samples"].append(merged)
    result["warnings"] = sorted(set(str(item) for item in result["warnings"]))
    return result


def load_raw_documents(raw_dir):
    import json
    documents = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, name), "r", encoding="utf-8") as handle:
            documents.append(json.load(handle))
    return documents

