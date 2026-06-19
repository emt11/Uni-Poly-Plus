"""TransE baseline and repeat-unit KG embedding export."""

import json
import os
import random

import numpy as np

from .io_utils import read_csv, stable_id, write_csv, write_json


def _read_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def train_transe(triples_path, nodes_path, edges_path, repeat_units_path, output_dir,
                 embedding_dim=128, epochs=100, seed=13, graph_variant="strict"):
    edge_rows = read_csv(edges_path)
    literature_relations = {"exact_repeat_unit_match", "canonical_smiles_match", "polymer_class_match", "alias_match", "functional_group_similarity", "family_level_match"}
    literature_repeat_units = {
        edge["tail_id"] for edge in edge_rows
        if edge.get("relation_type") in literature_relations and edge.get("graph_scope") in {"strict", "broad"}
    }
    random.seed(seed)
    np.random.seed(seed)
    triples = _read_triples(triples_path)
    nodes = read_csv(nodes_path)
    build_manifest_path = os.path.join(os.path.dirname(nodes_path), "build_manifest.json")
    build_manifest = {}
    if os.path.exists(build_manifest_path):
        with open(build_manifest_path, "r", encoding="utf-8") as handle:
            build_manifest = json.load(handle)
    node_types = {row["node_id"]: row["node_type"] for row in nodes}
    node_sources = {row["node_id"]: row["source_id"] for row in nodes}
    allowed_scopes = {"strict"} if graph_variant == "strict" else {"strict", "broad"}
    repeat_to_class = {
        edge["head_id"]: edge["tail_id"] for edge in edge_rows
        if edge.get("relation_type") == "maps_to" and edge.get("graph_scope") in allowed_scopes
    }
    repeat_to_family = {
        edge["head_id"]: edge["tail_id"] for edge in edge_rows
        if edge.get("relation_type") == "belongs_to_family" and edge.get("graph_scope") in allowed_scopes
    }
    eligible = sorted({entity for triple in triples for entity in (triple[0], triple[2]) if node_types.get(entity) != "SourceChunk"})
    test_only = not triples or not literature_repeat_units
    if not eligible:
        eligible = sorted(row["node_id"] for row in nodes if row["node_type"] in {"RepeatUnit", "PolymerClass"})
    entity_to_index = {entity: index for index, entity in enumerate(eligible)}
    relations = sorted({relation for _, relation, _ in triples})
    relation_to_index = {relation: index for index, relation in enumerate(relations)}
    entity_embeddings = np.random.default_rng(seed).normal(0, 0.05, (max(1, len(eligible)), embedding_dim)).astype(np.float32)
    if triples:
        try:
            import torch
            import torch.nn.functional as F

            torch.manual_seed(seed)
            entity = torch.nn.Embedding(len(eligible), embedding_dim)
            relation = torch.nn.Embedding(len(relations), embedding_dim)
            torch.nn.init.uniform_(entity.weight, -6 / embedding_dim ** 0.5, 6 / embedding_dim ** 0.5)
            torch.nn.init.uniform_(relation.weight, -6 / embedding_dim ** 0.5, 6 / embedding_dim ** 0.5)
            optimizer = torch.optim.Adam([entity.weight, relation.weight], lr=0.01)
            indexed = [(entity_to_index[h], relation_to_index[r], entity_to_index[t]) for h, r, t in triples if h in entity_to_index and t in entity_to_index]
            for _ in range(epochs):
                random.shuffle(indexed)
                for start in range(0, len(indexed), 256):
                    batch = torch.tensor(indexed[start:start + 256], dtype=torch.long)
                    if batch.numel() == 0:
                        continue
                    negative_tail = torch.randint(0, len(eligible), (len(batch),))
                    positive = torch.linalg.vector_norm(entity(batch[:, 0]) + relation(batch[:, 1]) - entity(batch[:, 2]), ord=1, dim=1)
                    negative = torch.linalg.vector_norm(entity(batch[:, 0]) + relation(batch[:, 1]) - entity(negative_tail), ord=1, dim=1)
                    loss = F.relu(1.0 + positive - negative).mean()
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    with torch.no_grad():
                        entity.weight[:] = F.normalize(entity.weight, p=2, dim=1)
            entity_embeddings = entity.weight.detach().cpu().numpy().astype(np.float32)
        except ImportError:
            test_only = True

    repeat_units = read_csv(repeat_units_path)
    mapping_rows = []
    output_vectors = []
    unknown_vector = np.random.default_rng(seed + 1).normal(0, 0.05, embedding_dim).astype(np.float32)
    fallback_count = 0
    for unit in repeat_units:
        kg_node_id = f"kg_repeatunit_{unit['repeat_unit_id']}"
        has_entity = kg_node_id in entity_to_index
        class_node_id = repeat_to_class.get(kg_node_id)
        family_node_id = repeat_to_family.get(kg_node_id)
        if has_entity:
            vector = entity_embeddings[entity_to_index[kg_node_id]]
        elif class_node_id in entity_to_index:
            vector = entity_embeddings[entity_to_index[class_node_id]]
            fallback_count += 1
        elif family_node_id in entity_to_index:
            vector = entity_embeddings[entity_to_index[family_node_id]]
            fallback_count += 1
        else:
            vector = unknown_vector.copy()
            fallback_count += 1
        embedding_index = len(output_vectors)
        output_vectors.append(vector)
        has_link = kg_node_id in literature_repeat_units
        mapping_rows.append({
            "embedding_index": embedding_index, "kg_node_id": kg_node_id,
            "entity_type": "RepeatUnit", "repeat_unit_id": unit["repeat_unit_id"],
            "canonical_smiles": unit.get("canonical_smiles", ""), "polymer_class_id": node_sources.get(class_node_id, ""),
            "graph_variant": graph_variant, "has_literature_link": str(has_link).lower(),
            "embedding_version": "transe_mvp_v1",
        })
    os.makedirs(output_dir, exist_ok=True)
    matrix = np.stack(output_vectors).astype(np.float32) if output_vectors else np.empty((0, embedding_dim), dtype=np.float32)
    np.save(os.path.join(output_dir, "kg_embedding.npy"), matrix)
    fields = ["embedding_index", "kg_node_id", "entity_type", "repeat_unit_id", "canonical_smiles", "polymer_class_id", "graph_variant", "has_literature_link", "embedding_version"]
    write_csv(os.path.join(output_dir, "kg_entity_mapping.csv"), fields, mapping_rows)
    manifest = {
        "model": "TransE",
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "seed": seed,
        "graph_variant": graph_variant,
        "triple_count": len(triples),
        "entity_count": len(eligible),
        "mapped_repeat_unit_count": len(mapping_rows),
        "output_shape": list(matrix.shape),
        "dtype": "float32",
        "fallback_count": fallback_count,
        "source_chunks_excluded": True,
        "has_literature_links": bool(literature_repeat_units),
        "test_only": test_only,
        "warnings": ["no_validated_literature_links_embedding_is_test_only"] if test_only else [],
        "fallback": "repeat_unit_then_polymer_class_then_family_then_seeded_unknown",
        "mapping_mode": build_manifest.get("mapping_mode", ""),
        "mapping_source_type": build_manifest.get("mapping_source_type", ""),
        "mapping_provider": build_manifest.get("mapping_provider", ""),
        "mapping_model": build_manifest.get("mapping_model", ""),
    }
    write_json(os.path.join(output_dir, "embedding_manifest.json"), manifest)
    return manifest
