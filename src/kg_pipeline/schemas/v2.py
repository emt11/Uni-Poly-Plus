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


def empty_article_document(article_id, title=None):
    return {
        "schema_version": "2.0",
        "article": {"article_id": article_id, "title": title, "doi": None, "year": None, "journal": None},
        "evidence_records": [],
        "literature_samples": [],
        "dataset_links": [],
        "warnings": [],
    }


def validate_schema_v2(document):
    errors = []
    required = {"article", "evidence_records", "literature_samples", "dataset_links", "warnings"}
    missing = required.difference(document)
    if missing:
        errors.append(f"missing top-level fields: {sorted(missing)}")
    if str(document.get("schema_version", "2.0")) != "2.0":
        errors.append("schema_version must be 2.0")
    for field in ("evidence_records", "literature_samples", "dataset_links", "warnings"):
        if not isinstance(document.get(field, []), list):
            errors.append(f"{field} must be an array")
    return errors

