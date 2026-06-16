import argparse
import _common  # noqa: F401
from src.kg_pipeline.extraction import extract_articles


def main():
    parser = argparse.ArgumentParser(description="Extract schema v2.0 facts using Qwen or DeepSeek")
    parser.add_argument("--candidate_chunks", required=True)
    parser.add_argument("--source_chunks")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--max_candidates", type=int)
    args = parser.parse_args()
    print({"successful_extractions": extract_articles(args.candidate_chunks, args.output_dir, args.provider, args.source_chunks, args.max_candidates)})


if __name__ == "__main__":
    main()

