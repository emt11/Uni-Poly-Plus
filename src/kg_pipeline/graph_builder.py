import json
import os
from collections import Counter

from .common import file_sha256, read_csv, read_jsonl, stable_id, write_csv, write_json
from .numeric_features import log_numeric_bin, normalize_mass


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
    del candidates_path
    graph = Graph()
    repeat_nodes = {}
    for unit in read_csv(repeat_units_path):
        repeat_nodes[unit["repeat_unit_id"]] = graph.node("RepeatUnit", unit["repeat_unit_id"], unit.get("canonical_smiles"), unit, "dataset")
    mapping_path = os.path.join(os.path.dirname(repeat_units_path), "polymer_class_candidates.jsonl")
    if os.path.exists(mapping_path):
        for mapping in read_jsonl(mapping_path):
            repeat_node = repeat_nodes.get(mapping.get("repeat_unit_id"))
            confidence = float(mapping.get("confidence", 0))
            for candidate in mapping.get("polymer_class_candidates", []):
                class_id = candidate.get("polymer_class_id")
                if not repeat_node or not class_id or class_id == "pc_unknown":
                    continue
                class_node = graph.node(
                    "PolymerClass", class_id, candidate.get("canonical_name") or class_id,
                    {"polymer_family": candidate.get("polymer_family"), "composition_type": candidate.get("composition_type"), "temporary_mapping": mapping.get("source_type") == "llm_temporary_mapping"},
                    "dataset", mapping.get("source_type", "mapping"),
                )
                scope = "strict" if confidence >= 0.8 else "broad"
                graph.edge(repeat_node, "maps_to", class_node, confidence, properties={"mapping_method": mapping.get("mapping_method"), "source_type": mapping.get("source_type")}, scope=scope)

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
        scope = "strict" if relation in strict_link_relations or (relation == "polymer_class_match" and float(link.get("confidence", 0)) >= 0.8) else "broad"
        graph.edge(sample_node, relation, repeat_node, link.get("confidence", 0), link.get("evidence_refs", []), {"link_id": link["link_id"], "matched_on": link.get("matched_on", []), "source_type": link.get("source_type")}, scope=scope)

    os.makedirs(output_dir, exist_ok=True)
    node_rows = sorted(graph.nodes.values(), key=lambda row: row["node_id"])
    edge_rows = sorted(graph.edges.values(), key=lambda row: row["edge_id"])
    write_csv(os.path.join(output_dir, "nodes.csv"), NODE_FIELDS, node_rows)
    write_csv(os.path.join(output_dir, "edges.csv"), EDGE_FIELDS, edge_rows)
    with open(os.path.join(output_dir, "triples.tsv"), "w", encoding="utf-8") as handle:
        for edge in edge_rows:
            if edge["active"] == "true" and edge["graph_scope"] in ({"strict", "provenance"} if graph_variant == "strict" else {"strict", "broad", "provenance"}):
                if graph.nodes[edge["head_id"]]["node_type"] == "SourceChunk" or graph.nodes[edge["tail_id"]]["node_type"] == "SourceChunk":
                    continue
                handle.write(f"{edge['head_id']}\t{edge['relation_type']}\t{edge['tail_id']}\n")
    alias_rows = []
    for node in node_rows:
        if node["node_type"] in {"PolymerClass", "LiteratureSample"}:
            alias_rows.append({"alias_id": stable_id("alias", node["display_name"], node["node_id"]), "alias_text": node["display_name"], "normalized_alias": node["display_name"].lower(), "target_node_id": node["node_id"], "target_type": node["node_type"], "source_article_id": "", "evidence_id": "", "mapping_method": node["created_by"], "confidence": 1.0, "status": "accepted"})
    write_csv(os.path.join(output_dir, "entity_aliases.csv"), ["alias_id", "alias_text", "normalized_alias", "target_node_id", "target_type", "source_article_id", "evidence_id", "mapping_method", "confidence", "status"], alias_rows)
    type_counts = Counter(row["node_type"] for row in node_rows)
    manifest = {"schema_version": "2.0", "graph_variant": graph_variant, "nodes": len(node_rows), "edges": len(edge_rows), "node_types": dict(type_counts), "source_chunks_excluded_from_transe": True, "inputs": {"records_sha256": file_sha256(records_path), "repeat_units_sha256": file_sha256(repeat_units_path)}}
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

