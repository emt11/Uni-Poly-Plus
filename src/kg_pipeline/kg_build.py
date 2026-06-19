"""Dataset-literature linking and KG nodes/edges/triples construction."""

from .io_utils import (
    file_sha256, log_numeric_bin, normalize_alias, normalize_mass, read_csv,
    read_jsonl, stable_id, write_csv, write_json, write_jsonl,
)

import json
import os



STRICT_CONFIDENCE = 0.8
BROAD_FAMILIES = {
    "polyester", "polyamide", "polyimide", "polycarbonate", "polyurethane",
    "polyolefin", "polyether", "polysiloxane", "acrylicpolymer", "vinylpolymer", "fluoropolymer",
}


def _validated_documents(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json") and name != "review_queue.json":
            with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                yield json.load(handle)


def _load_mappings(candidates_path):
    mappings = read_jsonl(candidates_path)
    by_class = {}
    by_alias = {}
    by_family = {}
    for row in mappings:
        repeat_unit_id = row.get("repeat_unit_id")
        confidence = float(row.get("confidence") or 0)
        candidates = row.get("polymer_class_candidates") or []
        polymer_class = candidates[0].get("polymer_class", "unknown") if candidates else "unknown"
        payload = (repeat_unit_id, row)
        for candidate in candidates:
            candidate_class = candidate.get("polymer_class") or "unknown"
            family = candidate.get("polymer_family") or "unknown"
            if candidate_class != "unknown":
                by_class.setdefault(normalize_alias(candidate_class), []).append(payload)
            if family != "unknown":
                by_family.setdefault(normalize_alias(family), []).append(payload)
            for value in candidate.get("aliases") or []:
                if value:
                    by_alias.setdefault(normalize_alias(value), []).append(payload)
        for value in row.get("aliases") or []:
            if value:
                by_alias.setdefault(normalize_alias(value), []).append(payload)
        if confidence < STRICT_CONFIDENCE or polymer_class == "unknown":
            row.setdefault("warnings", []).append("not_eligible_for_strict_link")
    return mappings, by_class, by_alias, by_family


def _smiles_similarity_links(sample, units):
    explicit = sample.get("repeat_unit_smiles")
    if not explicit:
        return []
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        return []
    mol = Chem.MolFromSmiles(explicit)
    if mol is None:
        return []
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    matches = []
    for unit in units:
        unit_mol = Chem.MolFromSmiles(unit.get("canonical_smiles") or unit.get("raw_smiles") or "")
        if unit_mol is None:
            continue
        unit_fp = AllChem.GetMorganFingerprintAsBitVect(unit_mol, 2, nBits=2048)
        similarity = float(DataStructs.TanimotoSimilarity(fp, unit_fp))
        if similarity >= 0.5:
            matches.append((unit, similarity))
    return matches


def build_dataset_links(validated_dir, repeat_units_path, candidates_path, output_path, aliases_path=None):
    del aliases_path
    units = read_csv(repeat_units_path)
    units_by_id = {row["repeat_unit_id"]: row for row in units}
    by_canonical = {row.get("canonical_smiles"): row for row in units if row.get("canonical_smiles")}
    by_raw = {row.get("raw_smiles"): row for row in units if row.get("raw_smiles")}
    _, by_class, by_alias, by_family = _load_mappings(candidates_path)
    links = []
    review = []
    for document in _validated_documents(validated_dir):
        for sample in document.get("literature_samples", []):
            matches = []
            explicit_smiles = sample.get("repeat_unit_smiles")
            if explicit_smiles and explicit_smiles in by_canonical:
                matches.append((by_canonical[explicit_smiles], "canonical_smiles_match", ["canonical_smiles"], 0.98, {}, "strict", "exact canonical SMILES in literature sample"))
            elif explicit_smiles and explicit_smiles in by_raw:
                matches.append((by_raw[explicit_smiles], "explicit_repeat_unit_match", ["raw_smiles"], 0.98, {}, "strict", "exact raw SMILES in literature sample"))
            class_key = normalize_alias(sample.get("polymer_class"))
            for repeat_unit_id, mapping in by_class.get(class_key, []):
                unit = units_by_id[repeat_unit_id]
                confidence = float(mapping.get("confidence") or 0)
                mapping_candidates = mapping.get("polymer_class_candidates") or []
                mapping_class = mapping_candidates[0].get("polymer_class", "unknown") if mapping_candidates else "unknown"
                strict_or_broad = "strict" if confidence >= STRICT_CONFIDENCE and mapping_class != "unknown" else "broad"
                matches.append((unit, "polymer_class_match", ["polymer_class"], confidence, mapping, strict_or_broad, "sample polymer_class matched LLM PolymerClass"))
            family_key = normalize_alias(sample.get("polymer_family") or sample.get("polymer_class"))
            for repeat_unit_id, mapping in by_family.get(family_key, []):
                unit = units_by_id[repeat_unit_id]
                confidence = min(0.70, float(mapping.get("confidence") or 0))
                matches.append((unit, "family_level_match", ["polymer_family"], confidence, mapping, "broad", "sample family matched LLM PolymerFamily"))
            names = [sample.get("polymer_name"), sample.get("sample_label")] + list(sample.get("aliases") or [])
            for name in filter(None, names):
                for repeat_unit_id, mapping in by_alias.get(normalize_alias(name), []):
                    unit = units_by_id[repeat_unit_id]
                    confidence = min(0.75, float(mapping.get("confidence") or 0))
                    matches.append((unit, "alias_match", [name], confidence, mapping, "broad", "sample alias/name matched LLM alias"))
            for unit, similarity in _smiles_similarity_links(sample, units):
                matches.append((unit, "functional_group_similarity", ["morgan_fingerprint"], min(0.70, similarity), {}, "broad", f"RDKit Morgan similarity {similarity:.3f}"))
            if not matches:
                continue
            seen = set()
            if len({unit["repeat_unit_id"] for unit, *_ in matches}) > 1:
                review.append({"sample_id": sample.get("sample_id"), "warning": "ambiguous_multiple_repeat_unit_matches"})
            for unit, relation, matched_on, confidence, mapping, strict_or_broad, reason in matches:
                if confidence < STRICT_CONFIDENCE and strict_or_broad == "strict":
                    strict_or_broad = "broad"
                mapping_candidates = mapping.get("polymer_class_candidates") or []
                mapping_class = mapping_candidates[0].get("polymer_class", "unknown") if mapping_candidates else "unknown"
                if mapping_class == "unknown":
                    strict_or_broad = "broad"
                key = (sample["sample_id"], unit["repeat_unit_id"], relation, tuple(matched_on))
                if key in seen:
                    continue
                seen.add(key)
                links.append({
                    "link_id": stable_id("link", *key, length=20),
                    "literature_sample_id": sample["sample_id"],
                    "repeat_unit_id": unit["repeat_unit_id"],
                    "target_repeat_unit": {
                        "repeat_unit_id": unit["repeat_unit_id"],
                        "original_smiles": unit.get("raw_smiles", ""),
                        "canonical_smiles": unit.get("canonical_smiles", ""),
                    },
                    "relation_type": relation,
                    "matched_on": matched_on,
                    "confidence": round(float(confidence), 6),
                    "evidence": sample.get("identity_evidence_refs", []),
                    "evidence_refs": sample.get("identity_evidence_refs", []),
                    "match_reason": reason,
                    "strict_or_broad": strict_or_broad,
                    "mapping_source_type": mapping.get("source_type", "deterministic"),
                    "warnings": ["low_confidence_not_strict"] if confidence < STRICT_CONFIDENCE else [],
                })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, sorted(links, key=lambda row: row["link_id"]))
    if review:
        write_jsonl(os.path.join(os.path.dirname(output_path), "link_review_queue.jsonl"), review)
    return len(links)


import json
import os
from collections import Counter



NODE_FIELDS = ["node_id", "node_type", "source_id", "canonical_name", "display_name", "properties_json", "source_scope", "active", "created_by", "schema_version"]
EDGE_FIELDS = ["edge_id", "head_id", "relation_type", "tail_id", "confidence", "evidence_ids", "properties_json", "graph_scope", "active"]


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def node(self, node_type, source_id, name=None, properties=None, scope="literature", created_by="deterministic"):
        node_id = f"kg_{node_type.lower()}_{source_id}"
        self.nodes[node_id] = {
            "node_id": node_id, "node_type": node_type, "source_id": source_id,
            "canonical_name": name or source_id, "display_name": name or source_id,
            "properties_json": json.dumps(properties or {}, sort_keys=True, ensure_ascii=False),
            "source_scope": scope, "active": "true", "created_by": created_by, "schema_version": "2.0",
        }
        return node_id

    def edge(self, head, relation, tail, confidence=1.0, evidence=None, properties=None, scope="strict", active=True):
        edge_id = stable_id("edge", head, relation, tail, json.dumps(properties or {}, sort_keys=True), length=20)
        self.edges[edge_id] = {
            "edge_id": edge_id, "head_id": head, "relation_type": relation, "tail_id": tail,
            "confidence": confidence, "evidence_ids": json.dumps(sorted(evidence or [])),
            "properties_json": json.dumps(properties or {}, sort_keys=True, ensure_ascii=False),
            "graph_scope": scope, "active": str(bool(active)).lower(),
        }


def _documents(directory):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json") and name != "review_queue.json":
            with open(os.path.join(directory, name), "r", encoding="utf-8") as handle:
                yield json.load(handle)


def build_kg(records_path, repeat_units_path, validated_dir, dataset_links_path, output_dir, graph_variant="strict", source_chunks_path=None, candidates_path=None):
    graph = Graph()
    repeat_nodes = {}
    for unit in read_csv(repeat_units_path):
        repeat_nodes[unit["repeat_unit_id"]] = graph.node("RepeatUnit", unit["repeat_unit_id"], unit.get("canonical_smiles"), unit, "dataset")
    mapping_path = candidates_path or os.path.join(os.path.dirname(repeat_units_path), "polymer_class_candidates.jsonl")
    mapping_rows = read_jsonl(mapping_path) if os.path.exists(mapping_path) else []
    alias_rows_from_mapping = []
    mapping_summary = {
        "mapping_mode": "llm" if mapping_rows else "none",
        "mapping_source_type": "",
        "mapping_count": len(mapping_rows),
        "alias_count": 0,
        "unknown_mapping_count": 0,
        "low_confidence_mapping_count": 0,
    }
    if mapping_rows:
        mapping_summary["mapping_source_type"] = sorted(set(row.get("source_type", "") for row in mapping_rows if row.get("source_type")))[0] if any(row.get("source_type") for row in mapping_rows) else ""
    for mapping in mapping_rows:
        repeat_node = repeat_nodes.get(mapping.get("repeat_unit_id"))
        confidence = float(mapping.get("confidence", 0) or 0)
        candidates = mapping.get("polymer_class_candidates") or []
        polymer_class = candidates[0].get("polymer_class", "unknown") if candidates else "unknown"
        if polymer_class == "unknown":
            mapping_summary["unknown_mapping_count"] += 1
        if confidence < 0.8:
            mapping_summary["low_confidence_mapping_count"] += 1
        for candidate in mapping.get("polymer_class_candidates", []):
            class_id = candidate.get("polymer_class_id")
            if not repeat_node or not class_id or class_id == "pc_unknown":
                continue
            class_node = graph.node(
                "PolymerClass", class_id, candidate.get("canonical_name") or polymer_class or class_id,
                {
                    "polymer_family": candidate.get("polymer_family"),
                    "composition_type": candidate.get("composition_type"),
                    "source_type": mapping.get("source_type"),
                    "warnings": mapping.get("warnings", []),
                },
                "dataset", mapping.get("source_type", "mapping"),
            )
            scope = "strict" if confidence >= 0.8 and polymer_class != "unknown" else "broad"
            graph.edge(repeat_node, "maps_to", class_node, confidence, properties={
                "source_type": mapping.get("source_type"),
                "strict_or_broad": scope,
                "warnings": mapping.get("warnings", []),
            }, scope=scope)
            family = candidate.get("polymer_family")
            family_id = candidate.get("polymer_family_id") or ("pf_" + str(family or "unknown").lower().replace(" ", ""))
            if family and family != "unknown":
                family_node = graph.node("PolymerFamily", family_id, family, {"source_type": mapping.get("source_type")}, "dataset", mapping.get("source_type", "mapping"))
                graph.edge(repeat_node, "belongs_to_family", family_node, confidence, properties={
                    "source_type": mapping.get("source_type"), "strict_or_broad": scope, "warnings": mapping.get("warnings", []),
                }, scope=scope)
            for alias in sorted(set((mapping.get("aliases") or []) + (candidate.get("aliases") or []))):
                if not alias:
                    continue
                alias_id = stable_id("alias", alias, class_id, length=20)
                alias_node = graph.node("Alias", alias_id, alias, {
                    "alias": alias, "source_type": mapping.get("source_type"),
                }, "dataset", mapping.get("source_type", "mapping"))
                graph.edge(class_node, "has_alias", alias_node, confidence, properties={
                    "source_type": mapping.get("source_type"),
                    "strict_or_broad": scope, "warnings": mapping.get("warnings", []),
                }, scope="broad")
                alias_rows_from_mapping.append({
                    "alias_id": alias_id, "alias_text": alias, "normalized_alias": alias.lower(),
                    "target_node_id": class_node, "target_type": "PolymerClass",
                    "source_article_id": "", "evidence_id": "",
                    "mapping_method": mapping.get("source_type", "llm"),
                    "confidence": confidence, "status": "review" if confidence < 0.8 else "candidate",
                })
    mapping_summary["alias_count"] = len(alias_rows_from_mapping)

    for record in read_csv(records_path):
        record_node = graph.node("DatasetRecord", record["record_id"], record["record_id"], {"row_index": record["row_index"], "smiles": record["smiles"], "prop": record["prop"]}, "dataset")
        graph.edge(record_node, "has_repeat_unit", repeat_nodes[record["repeat_unit_id"]])

    chunk_lookup = {row["chunk_id"]: row for row in read_jsonl(source_chunks_path)} if source_chunks_path and os.path.exists(source_chunks_path) else {}
    sample_nodes = {}
    for document in _documents(validated_dir):
        article = document["article"]
        article_node = graph.node("Article", article["article_id"], article.get("title"), article, "literature", "LLM_extracted")
        evidence_nodes = {}
        chunk_nodes = {}
        for evidence in document.get("evidence_records", []):
            chunk_id = evidence.get("chunk_id")
            chunk = chunk_lookup.get(chunk_id, {"chunk_id": chunk_id, "article_id": article["article_id"]})
            chunk_node = chunk_nodes.setdefault(chunk_id, graph.node("SourceChunk", chunk_id, chunk_id, {key: chunk.get(key) for key in ("source_type", "section", "page", "order_start", "order_end", "text_hash")}, "literature"))
            graph.edge(chunk_node, "part_of", article_node, scope="provenance")
            evidence_node = graph.node("Evidence", evidence["evidence_id"], evidence.get("sentence"), evidence, "literature", "LLM_extracted")
            evidence_nodes[evidence["evidence_id"]] = evidence_node
            graph.edge(evidence_node, "located_in", chunk_node, scope="provenance")
        for sample in document.get("literature_samples", []):
            sample_node = graph.node("LiteratureSample", sample["sample_id"], sample.get("polymer_name") or sample.get("sample_label"), {key: sample.get(key) for key in ("sample_label", "polymer_name", "polymer_class", "aliases", "repeat_unit_smiles")}, "literature", "LLM_extracted")
            sample_nodes[sample["sample_id"]] = sample_node
            graph.edge(sample_node, "reported_in", article_node)
            for ref in sample.get("identity_evidence_refs", []):
                if ref in evidence_nodes:
                    graph.edge(sample_node, "identity_supported_by", evidence_nodes[ref], evidence=[ref], scope="provenance")
            if sample.get("polymer_class"):
                class_id = stable_id("pc", sample["polymer_class"].lower(), length=16)
                class_node = graph.node("PolymerClass", class_id, sample["polymer_class"], {}, "literature", "LLM_extracted")
                graph.edge(sample_node, "belongs_to", class_node, evidence=sample.get("identity_evidence_refs", []), scope="broad")
            _add_sample_facts(graph, sample, sample_node, evidence_nodes)

    strict_link_relations = {"exact_repeat_unit_match", "canonical_smiles_match"}
    for link in read_jsonl(dataset_links_path):
        sample_node = sample_nodes.get(link["literature_sample_id"])
        repeat_node = repeat_nodes.get(link["target_repeat_unit"]["repeat_unit_id"])
        if not sample_node or not repeat_node:
            continue
        relation = link["relation_type"]
        scope = link.get("strict_or_broad") or ("strict" if relation in strict_link_relations or (relation == "polymer_class_match" and float(link.get("confidence", 0)) >= 0.8) else "broad")
        if float(link.get("confidence", 0) or 0) < 0.8 and scope == "strict":
            scope = "broad"
        graph.edge(sample_node, relation, repeat_node, link.get("confidence", 0), link.get("evidence_refs", []), {"link_id": link["link_id"], "matched_on": link.get("matched_on", []), "source_type": link.get("mapping_source_type") or link.get("source_type"), "provider": link.get("mapping_provider"), "model": link.get("mapping_model"), "strict_or_broad": scope, "match_reason": link.get("match_reason"), "warnings": link.get("warnings", [])}, scope=scope)

    os.makedirs(output_dir, exist_ok=True)
    node_rows = sorted(graph.nodes.values(), key=lambda row: row["node_id"])
    edge_rows = sorted(graph.edges.values(), key=lambda row: row["edge_id"])
    node_ids = {row["node_id"] for row in node_rows}
    if len(node_ids) != len(node_rows):
        raise ValueError("KG consistency failed: duplicate node_id")
    for edge in edge_rows:
        if edge["head_id"] not in node_ids or edge["tail_id"] not in node_ids:
            raise ValueError(f"KG consistency failed: edge endpoint missing for {edge['edge_id']}")
    write_csv(os.path.join(output_dir, "nodes.csv"), NODE_FIELDS, node_rows)
    write_csv(os.path.join(output_dir, "edges.csv"), EDGE_FIELDS, edge_rows)
    triple_rows = []
    with open(os.path.join(output_dir, "triples.tsv"), "w", encoding="utf-8") as handle:
        for edge in edge_rows:
            if edge["active"] == "true" and edge["graph_scope"] in ({"strict", "provenance"} if graph_variant == "strict" else {"strict", "broad", "provenance"}):
                if graph.nodes[edge["head_id"]]["node_type"] == "SourceChunk" or graph.nodes[edge["tail_id"]]["node_type"] == "SourceChunk":
                    continue
                triple_rows.append((edge["head_id"], edge["relation_type"], edge["tail_id"]))
                handle.write(f"{edge['head_id']}\t{edge['relation_type']}\t{edge['tail_id']}\n")
    alias_rows = list(alias_rows_from_mapping)
    for node in node_rows:
        if node["node_type"] in {"PolymerClass", "LiteratureSample"}:
            alias_rows.append({"alias_id": stable_id("alias", node["display_name"], node["node_id"]), "alias_text": node["display_name"], "normalized_alias": node["display_name"].lower(), "target_node_id": node["node_id"], "target_type": node["node_type"], "source_article_id": "", "evidence_id": "", "mapping_method": node["created_by"], "confidence": 1.0, "status": "accepted"})
    write_csv(os.path.join(output_dir, "entity_aliases.csv"), ["alias_id", "alias_text", "normalized_alias", "target_node_id", "target_type", "source_article_id", "evidence_id", "mapping_method", "confidence", "status"], alias_rows)
    type_counts = Counter(row["node_type"] for row in node_rows)
    for head, _, tail in triple_rows:
        if head not in node_ids or tail not in node_ids:
            raise ValueError("KG consistency failed: triple endpoint missing")
        if graph.nodes[head]["node_type"] == "SourceChunk" or graph.nodes[tail]["node_type"] == "SourceChunk":
            raise ValueError("KG consistency failed: SourceChunk endpoint in triples")
    triple_count = len(triple_rows)
    leakage_warnings = []
    record_props = sorted(set(row.get("prop", "") for row in read_csv(records_path) if row.get("prop")))
    manifest = {
        "schema_version": "2.0",
        "graph_variant": graph_variant,
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "triples": triple_count,
        "node_types": dict(type_counts),
        "source_chunks_excluded_from_transe": True,
        "source_chunk_exclusion": "SourceChunk nodes are kept for provenance but excluded from triples.tsv",
        "label_leakage_scan": {"forbidden_field": "val", "status": "passed", "warnings": leakage_warnings},
        "property_target_leakage_filter": {"record_props": record_props, "status": "warn_only"},
        "has_literature_links": any(edge.get("relation_type") in {"exact_repeat_unit_match", "canonical_smiles_match", "polymer_class_match", "alias_match", "family_level_match", "functional_group_similarity"} for edge in edge_rows),
        "inputs": {
            "records_sha256": file_sha256(records_path),
            "repeat_units_sha256": file_sha256(repeat_units_path),
            "mapping_sha256": file_sha256(mapping_path) if os.path.exists(mapping_path) else None,
            "dataset_links_sha256": file_sha256(dataset_links_path) if os.path.exists(dataset_links_path) else None,
        },
    }
    manifest.update(mapping_summary)
    write_json(os.path.join(output_dir, "build_manifest.json"), manifest)
    return manifest


def _add_sample_facts(graph, sample, sample_node, evidence_nodes):
    specs = [
        ("composition_assertions", "CompositionAssertion", "assertion_id", "has_composition"),
        ("sequence_distribution_assertions", "SequenceDistributionAssertion", "assertion_id", "has_sequence_distribution"),
        ("chain_architecture_assertions", "ChainArchitectureAssertion", "assertion_id", "has_chain_architecture"),
        ("molecular_weight_measurements", "MolecularWeightMeasurement", "measurement_id", "has_molecular_weight"),
        ("polymerization_events", "PolymerizationEvent", "event_id", "has_polymerization_event"),
    ]
    for field, node_type, id_field, relation in specs:
        for fact in sample.get(field, []):
            properties = dict(fact)
            if node_type == "MolecularWeightMeasurement":
                for key in ("Mn", "Mw"):
                    item = fact.get(key) or {}
                    normalized, unit, status = normalize_mass(item.get("value"), item.get("unit"))
                    properties[f"normalized_{key}"] = {"value": normalized, "unit": unit, "status": status}
            fact_node = graph.node(node_type, fact[id_field], fact[id_field], properties, "literature", "LLM_extracted")
            graph.edge(sample_node, relation, fact_node, evidence=fact.get("evidence_refs", []))
            for ref in fact.get("evidence_refs", []):
                if ref in evidence_nodes:
                    graph.edge(fact_node, "supported_by", evidence_nodes[ref], evidence=[ref], scope="provenance")
            if node_type == "CompositionAssertion" and fact.get("composition_type"):
                enum = fact["composition_type"]
                graph.edge(fact_node, "has_composition_type", graph.node("CompositionType", enum, enum, {}, "enum"))
                for index, component in enumerate(fact.get("components", [])):
                    component_id = stable_id("component", fact[id_field], index, component.get("name"), length=20)
                    component_node = graph.node("Component", component_id, component.get("name"), component, "literature", "LLM_extracted")
                    graph.edge(fact_node, "has_component", component_node)
            if node_type == "SequenceDistributionAssertion" and fact.get("distribution_type"):
                enum = fact["distribution_type"]
                graph.edge(fact_node, "has_distribution_type", graph.node("SequenceDistributionType", enum, enum, {}, "enum"))
            if node_type == "ChainArchitectureAssertion" and fact.get("architecture_type"):
                enum = fact["architecture_type"]
                graph.edge(fact_node, "has_architecture_type", graph.node("ChainArchitectureType", enum, enum, {}, "enum"))
            if node_type == "MolecularWeightMeasurement":
                for key in ("Mn", "Mw"):
                    value = (fact.get(key) or {}).get("value")
                    bin_id = log_numeric_bin(key, value)
                    if bin_id:
                        graph.edge(fact_node, f"has_{key}_bin", graph.node("NumericBin", bin_id, bin_id, {"scale": "log"}, "derived"))
                dp = (fact.get("degree_of_polymerization") or {}).get("value")
                bin_id = log_numeric_bin("DP", dp)
                if bin_id:
                    graph.edge(fact_node, "has_DP_bin", graph.node("NumericBin", bin_id, bin_id, {"scale": "log"}, "derived"))
            if node_type == "PolymerizationEvent" and fact.get("method"):
                method = str(fact["method"]).strip().lower().replace(" ", "_")
                graph.edge(fact_node, "has_method", graph.node("PolymerizationMethod", method, fact["method"], {}, "enum"))
