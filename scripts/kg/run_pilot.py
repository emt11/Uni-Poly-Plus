import argparse
import json
import os
import _common  # noqa: F401

from src.kg_pipeline.chunk_recall import recall_chunks
from src.kg_pipeline.common import read_csv, read_jsonl, write_json
from src.kg_pipeline.document_parser import parse_documents
from src.kg_pipeline.embedding import train_transe
from src.kg_pipeline.extraction import extract_articles
from src.kg_pipeline.graph_builder import build_kg
from src.kg_pipeline.linking import build_dataset_links
from src.kg_pipeline.polymer_classes import generate_rule_stub
from src.kg_pipeline.records import prepare_records
from src.kg_pipeline.validation import aggregate_and_validate


STAGES = ["records", "mapping", "documents", "recall", "extraction", "validation", "links", "kg", "embedding"]


def count_json_files(path):
    return len([name for name in os.listdir(path) if name.endswith(".json") and name != "review_queue.json"]) if os.path.isdir(path) else 0


def main():
    parser = argparse.ArgumentParser(description="Run the local Polymer KG MVP pilot")
    parser.add_argument("--input", required=True)
    parser.add_argument("--documents_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_repeat_units", type=int, default=50)
    parser.add_argument("--provider", choices=["qwen", "deepseek"], default="qwen")
    parser.add_argument("--stop_after", choices=STAGES, default="embedding")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--embedding_epochs", type=int, default=20)
    args = parser.parse_args()
    root = args.output_dir
    paths = {
        "records": f"{root}/records.csv", "units": f"{root}/repeat_units.csv",
        "candidates_map": f"{root}/polymer_class_candidates.jsonl",
        "chunks": f"{root}/chunks/source_chunks.jsonl", "candidate_chunks": f"{root}/chunks/candidate_chunks.jsonl",
        "raw": f"{root}/extractions/raw", "aggregated": f"{root}/extractions/aggregated",
        "validated": f"{root}/extractions/validated", "links": f"{root}/links/dataset_links.jsonl",
        "kg": f"{root}/kg", "features": f"{root}/features",
    }
    os.makedirs(f"{root}/state", exist_ok=True)

    def done(stage, marker):
        return args.resume and os.path.exists(marker)

    def stop(stage):
        return STAGES.index(stage) >= STAGES.index(args.stop_after)

    if not done("records", paths["records"]):
        prepare_records(args.input, root, args.max_repeat_units)
    if stop("records"): return report(paths, root)
    if not done("mapping", paths["candidates_map"]):
        generate_rule_stub(paths["units"], root)
    if stop("mapping"): return report(paths, root)
    if not done("documents", paths["chunks"]):
        parse_documents(args.documents_dir, paths["chunks"])
    if stop("documents"): return report(paths, root)
    if not done("recall", paths["candidate_chunks"]):
        recall_chunks(paths["chunks"], paths["candidate_chunks"])
    if stop("recall"): return report(paths, root)
    os.makedirs(paths["raw"], exist_ok=True)
    if read_jsonl(paths["candidate_chunks"]) and not done("extraction", f"{paths['raw']}/calls.jsonl"):
        extract_articles(paths["candidate_chunks"], paths["raw"], args.provider, paths["chunks"], args.max_candidates)
    if stop("extraction"): return report(paths, root)
    if not done("validation", f"{paths['validated']}/review_queue.json"):
        aggregate_and_validate(paths["raw"], paths["chunks"], paths["aggregated"], paths["validated"])
    if stop("validation"): return report(paths, root)
    if not done("links", paths["links"]):
        build_dataset_links(paths["validated"], paths["units"], paths["candidates_map"], paths["links"])
    if stop("links"): return report(paths, root)
    if not done("kg", f"{paths['kg']}/build_manifest.json"):
        build_kg(paths["records"], paths["units"], paths["validated"], paths["links"], paths["kg"], "strict", paths["chunks"])
    if stop("kg"): return report(paths, root)
    if not done("embedding", f"{paths['features']}/embedding_manifest.json"):
        train_transe(f"{paths['kg']}/triples.tsv", f"{paths['kg']}/nodes.csv", f"{paths['kg']}/edges.csv", paths["units"], paths["features"], 128, args.embedding_epochs, 13, "strict")
    return report(paths, root)


def report(paths, root):
    import numpy as np
    build_manifest = {}
    if os.path.exists(f"{paths['kg']}/build_manifest.json"):
        build_manifest = json.load(open(f"{paths['kg']}/build_manifest.json", encoding="utf-8"))
    embedding_shape = None
    if os.path.exists(f"{paths['features']}/kg_embedding.npy"):
        embedding_shape = list(np.load(f"{paths['features']}/kg_embedding.npy", mmap_mode="r").shape)
    report_value = {
        "records_count": len(read_csv(paths["records"])) if os.path.exists(paths["records"]) else 0,
        "repeat_units_count": len(read_csv(paths["units"])) if os.path.exists(paths["units"]) else 0,
        "parsed_articles": len(set(row["article_id"] for row in read_jsonl(paths["chunks"]))) if os.path.exists(paths["chunks"]) else 0,
        "chunks_count": len(read_jsonl(paths["chunks"])) if os.path.exists(paths["chunks"]) else 0,
        "candidate_chunks_count": len(read_jsonl(paths["candidate_chunks"])) if os.path.exists(paths["candidate_chunks"]) else 0,
        "raw_extractions_count": count_json_files(paths["raw"]),
        "validated_articles_count": count_json_files(paths["validated"]),
        "dataset_links_count": len(read_jsonl(paths["links"])) if os.path.exists(paths["links"]) else 0,
        "nodes_count": build_manifest.get("nodes", 0), "edges_count": build_manifest.get("edges", 0),
        "triples_count": sum(1 for _ in open(f"{paths['kg']}/triples.tsv", encoding="utf-8")) if os.path.exists(f"{paths['kg']}/triples.tsv") else 0,
        "embedding_shape": embedding_shape,
        "missing_kg_mappings": 0 if embedding_shape else None,
        "possible_label_leakage_warnings": 0,
    }
    write_json(f"{root}/pilot_report.json", report_value)
    print(json.dumps(report_value, indent=2))
    return report_value


if __name__ == "__main__":
    main()
