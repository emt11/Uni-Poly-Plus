import argparse
import _common  # noqa: F401
from src.kg_pipeline.literature import retrieve_literature


def parse_bool(value):
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main():
    parser = argparse.ArgumentParser(description="Retrieve online literature metadata and selected OA full text for Polymer KG")
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--mapping_candidates", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--documents_dir", required=True)
    parser.add_argument("--sources", default="crossref,openalex,semantic_scholar,europe_pmc,arxiv")
    parser.add_argument("--max_repeat_units", type=int)
    parser.add_argument("--max_queries_per_repeat_unit", type=int, default=5)
    parser.add_argument("--max_results_per_query", type=int, default=10)
    parser.add_argument("--max_articles_per_repeat_unit", type=int, default=5)
    parser.add_argument("--max_downloads_per_repeat_unit", type=int, default=3)
    parser.add_argument("--max_total_downloads", type=int, default=100)
    parser.add_argument("--min_article_score", type=float, default=0.4)
    parser.add_argument("--require_oa_for_download", type=parse_bool, default=True)
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep_seconds", type=float, default=0.2)
    args = parser.parse_args()
    print(retrieve_literature(
        args.repeat_units,
        args.mapping_candidates,
        None,
        args.output_dir,
        args.documents_dir,
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
    ))


if __name__ == "__main__":
    main()
