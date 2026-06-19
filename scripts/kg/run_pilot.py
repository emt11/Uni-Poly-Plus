import argparse
import json
import os
import _common  # noqa: F401

from src.kg_pipeline.literature import recall_chunks
from src.kg_pipeline.io_utils import read_csv, read_jsonl, write_json
from src.kg_pipeline.literature import parse_documents
from src.kg_pipeline.kg_embedding import train_transe
from src.kg_pipeline.llm_extract import extract_articles, generate_llm_polymer_mapping, provider_key_env, provider_model
from src.kg_pipeline.kg_build import build_kg
from src.kg_pipeline.kg_build import build_dataset_links
from src.kg_pipeline.dataset import prepare_records
from src.kg_pipeline.literature import retrieve_literature
from src.kg_pipeline.article_validation import aggregate_and_validate


STAGES = ["records", "mapping", "retrieval", "documents", "recall", "extraction", "validation", "links", "kg", "embedding"]


def count_json_files(path):
    return len([name for name in os.listdir(path) if name.endswith(".json") and name != "review_queue.json"]) if os.path.isdir(path) else 0


def _is_unknown_mapping(row):
    return not row.get("aliases")


def _mapping_stats(paths):
    rows = read_jsonl(paths["candidates_map"]) if os.path.exists(paths["candidates_map"]) else []
    return {
        "mapping_mode": "llm" if rows else "none",
        "polymer_class_mapping_count": len(rows),
        "alias_count": sum(len(row.get("aliases") or []) for row in rows),
        "unknown_mapping_count": sum(1 for row in rows if _is_unknown_mapping(row)),
        "low_confidence_mapping_count": sum(1 for row in rows if float(row.get("confidence") or 0) < 0.8),
    }



def parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")

def main():
    parser = argparse.ArgumentParser(description="Run the local Polymer KG pilot pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--documents_dir", help="Local/online document directory. Defaults to <output_dir>/documents")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_repeat_units", type=int, default=50)
    parser.add_argument("--provider", choices=["qwen", "deepseek"], default="qwen")
    parser.add_argument("--model", help="Provider model override. If omitted, falls back to QWEN_MODEL/DEEPSEEK_MODEL/default")
    parser.add_argument("--mapping_mode", choices=["llm"], default="llm", help="Formal pipeline uses LLM mapping only")
    parser.add_argument("--mock_llm", action="store_true", help="Use mock LLM mapping and extraction; no API key required")
    parser.add_argument("--literature_mode", choices=["local", "online", "both"], default="local")
    parser.add_argument("--sources", default="crossref,openalex,semantic_scholar,europe_pmc,arxiv")
    parser.add_argument("--max_queries_per_repeat_unit", type=int, default=5)
    parser.add_argument("--max_results_per_query", type=int, default=10)
    parser.add_argument("--max_articles_per_repeat_unit", type=int, default=5)
    parser.add_argument("--max_downloads_per_repeat_unit", type=int, default=3)
    parser.add_argument("--max_total_downloads", type=int, default=100)
    parser.add_argument("--min_article_score", type=float, default=0.4)
    parser.add_argument("--require_oa_for_download", type=parse_bool, default=True)
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--stop_after", choices=STAGES, default="embedding")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_candidates", type=int, default=20)
    parser.add_argument("--embedding_epochs", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--overwrite_mapping", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not args.mock_llm and STAGES.index(args.stop_after) >= STAGES.index("mapping"):
        key_env = provider_key_env(args.provider)
        if not os.environ.get(key_env):
            raise SystemExit(f"ERROR: Missing API key: set {key_env} in the environment")
    root = args.output_dir
    documents_dir = args.documents_dir or f"{root}/documents"
    paths = {
        "records": f"{root}/records.csv", "units": f"{root}/repeat_units.csv",
        "candidates_map": f"{root}/polymer_class_candidates.jsonl",
        "chunks": f"{root}/chunks/source_chunks.jsonl", "candidate_chunks": f"{root}/chunks/candidate_chunks.jsonl",
        "raw": f"{root}/extractions/raw", "aggregated": f"{root}/extractions/aggregated",
        "validated": f"{root}/extractions/validated", "links": f"{root}/links/dataset_links.jsonl",
        "kg": f"{root}/kg", "features": f"{root}/features", "retrieval": f"{root}/retrieval", "documents_dir": documents_dir,
    }
    os.makedirs(f"{root}/state", exist_ok=True)

    def done(marker):
        return args.resume and os.path.exists(marker)

    def stop(stage):
        return STAGES.index(stage) >= STAGES.index(args.stop_after)

    def log_stage(stage, event, detail=""):
        suffix = f" - {detail}" if detail else ""
        print(f"[{stage}] {event}{suffix}", flush=True)

    def run_stage(stage, marker, action, enabled=True):
        if not enabled:
            log_stage(stage, "skipped", "disabled")
        elif done(marker):
            log_stage(stage, "skipped", f"resume marker exists: {marker}")
        else:
            log_stage(stage, "starting")
            action()
            log_stage(stage, "ending")
        if stop(stage):
            log_stage(stage, "stop_after reached")
            return report(paths, root, args.literature_mode)
        return None

    result = run_stage(
        "records",
        paths["records"],
        lambda: prepare_records(args.input, root, args.max_repeat_units),
    )
    if result is not None:
        return result

    result = run_stage(
        "mapping",
        paths["candidates_map"],
        lambda: generate_llm_polymer_mapping(
            paths["units"], root, args.provider, args.max_repeat_units,
            mock_response=args.mock_llm, overwrite=args.overwrite_mapping or not os.path.exists(paths["candidates_map"]),
            sleep_seconds=args.sleep_seconds, timeout=args.timeout, max_retries=args.max_retries,
            model=args.model,
        ),
    )
    if result is not None:
        return result

    result = run_stage(
        "retrieval",
        f"{paths['retrieval']}/retrieval_report.json",
        lambda: retrieve_literature(
            paths["units"], paths["candidates_map"], None, paths["retrieval"], paths["documents_dir"],
            sources=args.sources,
            max_repeat_units=args.max_repeat_units,
            max_queries_per_repeat_unit=args.max_queries_per_repeat_unit,
            max_results_per_query=args.max_results_per_query,
            max_articles_per_repeat_unit=args.max_articles_per_repeat_unit,
            max_downloads_per_repeat_unit=args.max_downloads_per_repeat_unit,
            max_total_downloads=args.max_total_downloads,
            min_article_score=args.min_article_score,
            require_oa_for_download=args.require_oa_for_download,
            metadata_only=args.metadata_only,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        ),
        enabled=args.literature_mode in {"online", "both"},
    )
    if result is not None:
        return result

    result = run_stage(
        "documents",
        paths["chunks"],
        lambda: parse_documents(paths["documents_dir"], paths["chunks"]),
    )
    if result is not None:
        return result

    result = run_stage(
        "recall",
        paths["candidate_chunks"],
        lambda: recall_chunks(
            paths["chunks"], paths["candidate_chunks"],
            mapping_path=paths["candidates_map"],
            retrieval_dir=paths["retrieval"] if args.literature_mode in {"online", "both"} else None,
        ),
    )
    if result is not None:
        return result

    os.makedirs(paths["raw"], exist_ok=True)
    result = run_stage(
        "extraction",
        f"{paths['raw']}/calls.jsonl",
        lambda: extract_articles(
            paths["candidate_chunks"], paths["raw"], args.provider, paths["chunks"], args.max_candidates,
            mock_response=args.mock_llm, timeout=args.timeout, max_retries=args.max_retries, sleep_seconds=args.sleep_seconds,
            model=args.model,
        ),
        enabled=bool(read_jsonl(paths["candidate_chunks"])),
    )
    if result is not None:
        return result

    result = run_stage(
        "validation",
        f"{paths['validated']}/review_queue.json",
        lambda: aggregate_and_validate(paths["raw"], paths["chunks"], paths["aggregated"], paths["validated"]),
    )
    if result is not None:
        return result

    result = run_stage(
        "links",
        paths["links"],
        lambda: build_dataset_links(paths["validated"], paths["units"], paths["candidates_map"], paths["links"]),
    )
    if result is not None:
        return result

    result = run_stage(
        "kg",
        f"{paths['kg']}/build_manifest.json",
        lambda: build_kg(paths["records"], paths["units"], paths["validated"], paths["links"], paths["kg"], "strict", paths["chunks"], paths["candidates_map"]),
    )
    if result is not None:
        return result

    result = run_stage(
        "embedding",
        f"{paths['features']}/embedding_manifest.json",
        lambda: train_transe(f"{paths['kg']}/triples.tsv", f"{paths['kg']}/nodes.csv", f"{paths['kg']}/edges.csv", paths["units"], paths["features"], args.embedding_dim, args.embedding_epochs, 13, "strict"),
    )
    if result is not None:
        return result
    return report(paths, root, args.literature_mode)


def report(paths, root, literature_mode="local"):
    import numpy as np
    build_manifest = {}
    embedding_manifest = {}
    if os.path.exists(f"{paths['kg']}/build_manifest.json"):
        build_manifest = json.load(open(f"{paths['kg']}/build_manifest.json", encoding="utf-8"))
    if os.path.exists(f"{paths['features']}/embedding_manifest.json"):
        embedding_manifest = json.load(open(f"{paths['features']}/embedding_manifest.json", encoding="utf-8"))
    retrieval_report = {}
    if os.path.exists(f"{paths.get('retrieval', root + '/retrieval')}/retrieval_report.json"):
        retrieval_report = json.load(open(f"{paths.get('retrieval', root + '/retrieval')}/retrieval_report.json", encoding="utf-8"))
    embedding_shape = None
    if os.path.exists(f"{paths['features']}/kg_embedding.npy"):
        embedding_shape = list(np.load(f"{paths['features']}/kg_embedding.npy", mmap_mode="r").shape)
    mapping_stats = _mapping_stats(paths)
    warnings = []
    if os.path.exists(paths["chunks"]) and not read_jsonl(paths["chunks"]):
        warnings.append("no_local_document_chunks")
    if os.path.exists(paths["candidate_chunks"]) and not read_jsonl(paths["candidate_chunks"]):
        warnings.append("no_candidate_chunks")
    if embedding_manifest.get("test_only"):
        warnings.append("embedding_manifest_test_only_true")
    report_value = {
        "records_count": len(read_csv(paths["records"])) if os.path.exists(paths["records"]) else 0,
        "repeat_units_count": len(read_csv(paths["units"])) if os.path.exists(paths["units"]) else 0,
        **mapping_stats,
        "literature_mode": literature_mode,
        "retrieval_enabled": literature_mode in {"online", "both"},
        "retrieved_article_count": retrieval_report.get("deduplicated_article_count", 0),
        "selected_article_count": retrieval_report.get("selected_article_count", 0),
        "downloaded_full_text_count": retrieval_report.get("downloaded_full_text_count", 0),
        "metadata_only_count": retrieval_report.get("metadata_only_count", 0),
        "parsed_articles": len(set(row["article_id"] for row in read_jsonl(paths["chunks"]))) if os.path.exists(paths["chunks"]) else 0,
        "source_chunks_count": len(read_jsonl(paths["chunks"])) if os.path.exists(paths["chunks"]) else 0,
        "candidate_chunks_count": len(read_jsonl(paths["candidate_chunks"])) if os.path.exists(paths["candidate_chunks"]) else 0,
        "raw_extractions_count": count_json_files(paths["raw"]),
        "validated_articles_count": count_json_files(paths["validated"]),
        "dataset_links_count": len(read_jsonl(paths["links"])) if os.path.exists(paths["links"]) else 0,
        "nodes_count": build_manifest.get("nodes", 0),
        "edges_count": build_manifest.get("edges", 0),
        "triples_count": build_manifest.get("triples", 0),
        "embedding_shape": embedding_shape,
        "test_only": embedding_manifest.get("test_only"),
        "warnings": warnings,
    }
    write_json(f"{root}/pilot_report.json", report_value)
    print(json.dumps(report_value, indent=2))
    return report_value


if __name__ == "__main__":
    main()
