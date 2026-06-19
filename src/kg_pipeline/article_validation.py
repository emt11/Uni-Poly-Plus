"""Schema v2 helpers, raw extraction aggregation, and article validation."""

from .io_utils import read_jsonl, stable_id, write_json

import json
import os


COMPOSITION_TYPES = {
    "homopolymer", "condensation_multi_monomer", "copolymer", "terpolymer",
    "blend", "composite", "mixture", "unknown",
}
RATIO_BASES = {"mol_fraction", "weight_fraction", "feed_ratio", "actual_ratio", "stoichiometric_ratio", None}
COMPONENT_ROLES = {"monomer", "comonomer", "polymer_component", "filler", "additive", "solvent", "unknown", None}
SEQUENCE_TYPES = {"random", "statistical", "alternating", "block", "blocky", "multiblock", "gradient", "graft", "periodic", "not_applicable", "unknown"}
ARCHITECTURE_TYPES = {"linear", "branched", "star", "graft", "comb", "brush", "network", "crosslinked_network", "hyperbranched", "dendrimer", "cyclic", "ladder", "unknown"}
DP_TYPES = {"number_average", "weight_average", "unknown", None}


SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schemas", "schema_v2.json")


def empty_article_document(article_id, title=None):
    return {
        "schema_version": "2.0",
        "article": {"article_id": article_id, "title": title, "doi": None, "year": None, "journal": None},
        "evidence_records": [],
        "literature_samples": [],
        "dataset_links": [],
        "warnings": [],
    }


def _fallback_validation_errors(document):
    errors = []
    required = {"article", "evidence_records", "literature_samples", "dataset_links", "warnings"}
    for field in sorted(required.difference(document)):
        errors.append({"path": field, "message": "required field is missing"})
    if str(document.get("schema_version", "2.0")) != "2.0":
        errors.append({"path": "schema_version", "message": "must be 2.0"})
    article = document.get("article")
    if not isinstance(article, dict):
        errors.append({"path": "article", "message": "must be an object"})
    else:
        for field in ("article_id", "title", "doi", "year", "journal"):
            if field not in article:
                errors.append({"path": f"article.{field}", "message": "required field is missing"})
        if not isinstance(article.get("article_id", ""), str):
            errors.append({"path": "article.article_id", "message": "must be a string"})
    for field in ("evidence_records", "literature_samples", "dataset_links", "warnings"):
        if not isinstance(document.get(field, []), list):
            errors.append({"path": field, "message": "must be an array"})
    for index, evidence in enumerate(document.get("evidence_records", []) if isinstance(document.get("evidence_records", []), list) else []):
        if not isinstance(evidence, dict):
            errors.append({"path": f"evidence_records.{index}", "message": "must be an object"})
            continue
        for field in ("evidence_id", "chunk_id", "sentence"):
            if field not in evidence:
                errors.append({"path": f"evidence_records.{index}.{field}", "message": "required field is missing"})
    sample_required = (
        "sample_id", "aliases", "identity_evidence_refs", "composition_assertions",
        "sequence_distribution_assertions", "chain_architecture_assertions",
        "molecular_weight_measurements", "polymerization_events",
    )
    for index, sample in enumerate(document.get("literature_samples", []) if isinstance(document.get("literature_samples", []), list) else []):
        if not isinstance(sample, dict):
            errors.append({"path": f"literature_samples.{index}", "message": "must be an object"})
            continue
        for field in sample_required:
            if field not in sample:
                errors.append({"path": f"literature_samples.{index}.{field}", "message": "required field is missing"})
            elif field != "sample_id" and not isinstance(sample.get(field), list):
                errors.append({"path": f"literature_samples.{index}.{field}", "message": "must be an array"})
    return errors


def validate_schema_v2_details(document):
    try:
        import jsonschema
        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            schema = json.load(handle)
        validator = jsonschema.Draft202012Validator(schema)
        errors = []
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path)):
            path = ".".join(str(part) for part in error.path) or "$"
            errors.append({"path": path, "message": error.message})
        return errors
    except ImportError:
        return _fallback_validation_errors(document)


def validate_schema_v2(document):
    return [f"schema_error:{item['path']}:{item['message']}" for item in validate_schema_v2_details(document)]


import copy
import os
from collections import defaultdict



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


import json
import math
import os
import re
from collections import defaultdict



def _positive(value):
    if value is None:
        return True
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _valid_fact(field, fact):
    if field == "composition_assertions":
        return (
            fact.get("composition_type") in COMPOSITION_TYPES
            and fact.get("ratio_basis") in RATIO_BASES
            and all(item.get("role") in COMPONENT_ROLES for item in fact.get("components", []))
        )
    if field == "sequence_distribution_assertions":
        return fact.get("distribution_type") in SEQUENCE_TYPES
    if field == "chain_architecture_assertions":
        return fact.get("architecture_type") in ARCHITECTURE_TYPES
    if field == "molecular_weight_measurements":
        mn = (fact.get("Mn") or {}).get("value")
        mw = (fact.get("Mw") or {}).get("value")
        dispersity = (fact.get("dispersity") or {}).get("value")
        dp = fact.get("degree_of_polymerization") or {}
        reported = any(value is not None for value in (mn, mw, dispersity, dp.get("value")))
        return (
            reported and _positive(mn) and _positive(mw)
            and (dispersity is None or (_positive(dispersity) and float(dispersity) >= 1))
            and _positive(dp.get("value")) and dp.get("dp_type") in DP_TYPES
        )
    conditions = fact.get("conditions") or {}
    condition_values = []
    for value in conditions.values():
        if isinstance(value, dict):
            condition_values.extend(value.values())
        else:
            condition_values.append(value)
    return bool(fact.get("method") or any(value is not None for value in condition_values))


def validate_article(document, source_chunks):
    warnings = list(document.get("warnings", []))
    schema_errors = validate_schema_v2_details(document)
    warnings.extend(f"schema_error:{item['path']}:{item['message']}" for item in schema_errors)
    document["_schema_errors"] = schema_errors
    evidence = {item.get("evidence_id"): item for item in document.get("evidence_records", [])}
    valid_evidence = set()
    for evidence_id, item in evidence.items():
        chunk = source_chunks.get(item.get("chunk_id"))
        sentence = str(item.get("sentence") or "").strip()
        if not chunk or not sentence or re.sub(r"\s+", " ", sentence) not in re.sub(r"\s+", " ", chunk.get("text", "")):
            warnings.append(f"evidence_missing_or_unlocatable:{evidence_id}")
        else:
            valid_evidence.add(evidence_id)
    for sample in document.get("literature_samples", []):
        for ref in sample.get("identity_evidence_refs", []):
            if ref not in evidence:
                warnings.append(f"dangling_evidence_ref:{ref}")
        for fact in sample.get("composition_assertions", []):
            if fact.get("composition_type") not in COMPOSITION_TYPES:
                warnings.append(f"unknown_composition_type:{fact.get('composition_type')}")
            if fact.get("ratio_basis") not in RATIO_BASES:
                warnings.append(f"unknown_ratio_basis:{fact.get('ratio_basis')}")
            for component in fact.get("components", []):
                if component.get("role") not in COMPONENT_ROLES:
                    warnings.append(f"unknown_component_role:{component.get('role')}")
        for fact in sample.get("sequence_distribution_assertions", []):
            if fact.get("distribution_type") not in SEQUENCE_TYPES:
                warnings.append(f"unknown_sequence_type:{fact.get('distribution_type')}")
        for fact in sample.get("chain_architecture_assertions", []):
            if fact.get("architecture_type") not in ARCHITECTURE_TYPES:
                warnings.append(f"unknown_architecture_type:{fact.get('architecture_type')}")
        for fact in sample.get("molecular_weight_measurements", []):
            if not _positive((fact.get("Mn") or {}).get("value")) or not _positive((fact.get("Mw") or {}).get("value")):
                warnings.append(f"nonpositive_molecular_weight:{fact.get('measurement_id')}")
            dispersity = (fact.get("dispersity") or {}).get("value")
            if dispersity is not None and (not _positive(dispersity) or float(dispersity) < 1):
                warnings.append(f"invalid_dispersity:{fact.get('measurement_id')}")
            dp = fact.get("degree_of_polymerization") or {}
            if not _positive(dp.get("value")) or dp.get("dp_type") not in DP_TYPES:
                warnings.append(f"invalid_degree_of_polymerization:{fact.get('measurement_id')}")
        for field in ("composition_assertions", "sequence_distribution_assertions", "chain_architecture_assertions", "molecular_weight_measurements", "polymerization_events"):
            accepted = []
            for fact in sample.get(field, []):
                refs = fact.get("evidence_refs", [])
                if not refs:
                    warnings.append(f"fact_without_evidence:{next(iter(fact.values()), field)}")
                for ref in refs:
                    if ref not in evidence:
                        warnings.append(f"dangling_evidence_ref:{ref}")
                if refs and all(ref in valid_evidence for ref in refs) and _valid_fact(field, fact):
                    accepted.append(fact)
                else:
                    warnings.append(f"fact_rejected_from_kg:{next(iter(fact.values()), field)}")
            sample[field] = accepted
    document["warnings"] = sorted(set(warnings))
    return document


def aggregate_and_validate(raw_dir, source_chunks_path, aggregated_dir, validated_dir):
    chunks = {row["chunk_id"]: row for row in read_jsonl(source_chunks_path)}
    grouped = defaultdict(list)
    for document in load_raw_documents(raw_dir):
        grouped[document.get("article", {}).get("article_id", "art_unknown")].append(document)
    os.makedirs(aggregated_dir, exist_ok=True)
    os.makedirs(validated_dir, exist_ok=True)
    review = []
    for article_id, documents in grouped.items():
        aggregated = aggregate_article(article_id, documents)
        write_json(os.path.join(aggregated_dir, f"{article_id}.json"), aggregated)
        validated = validate_article(aggregated, chunks)
        if validated.get("_schema_errors"):
            review.append({
                "article_id": article_id,
                "source_file": f"{article_id}.json",
                "validation_errors": validated["_schema_errors"],
                "warnings": validated["warnings"],
                "rejected_from_kg": True,
            })
            continue
        validated.pop("_schema_errors", None)
        write_json(os.path.join(validated_dir, f"{article_id}.json"), validated)
        if validated["warnings"]:
            review.append({"article_id": article_id, "warnings": validated["warnings"], "rejected_from_kg": False})
    write_json(os.path.join(validated_dir, "review_queue.json"), review)
    return len(grouped)
