import json
import math
import os
import re
from collections import defaultdict

from .aggregation import aggregate_article, load_raw_documents
from .common import read_jsonl, write_json
from .schemas.v2 import (
    ARCHITECTURE_TYPES, COMPOSITION_TYPES, COMPONENT_ROLES, DP_TYPES,
    RATIO_BASES, SEQUENCE_TYPES, validate_schema_v2,
)


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
    warnings.extend(validate_schema_v2(document))
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
        write_json(os.path.join(validated_dir, f"{article_id}.json"), validated)
        if validated["warnings"]:
            review.append({"article_id": article_id, "warnings": validated["warnings"]})
    write_json(os.path.join(validated_dir, "review_queue.json"), review)
    return len(grouped)
