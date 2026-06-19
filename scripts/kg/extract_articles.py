import argparse
import _common  # noqa: F401
from src.kg_pipeline.llm_extract import extract_articles


def main():
    parser = argparse.ArgumentParser(description="Extract schema v2.0 facts using Qwen or DeepSeek")
    parser.add_argument("--candidate_chunks", required=True)
    parser.add_argument("--source_chunks")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--model", help="Provider model override. If omitted, falls back to QWEN_MODEL/DEEPSEEK_MODEL/default")
    parser.add_argument("--max_candidates", type=int)
    parser.add_argument("--mock_response", action="store_true", help="Write mock extraction JSON without API calls")
    parser.add_argument("--dry_run", action="store_true", help="Only audit calls; do not write extraction JSON")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    args = parser.parse_args()
    try:
        successful = extract_articles(
            args.candidate_chunks,
            args.output_dir,
            args.provider,
            args.source_chunks,
            args.max_candidates,
            mock_response=args.mock_response,
            dry_run=args.dry_run,
            timeout=args.timeout,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
            model=args.model,
        )
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print({"successful_extractions": successful})


if __name__ == "__main__":
    main()
