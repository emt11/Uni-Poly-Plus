import argparse
import os
import _common  # noqa: F401
from src.kg_pipeline.article_validation import aggregate_and_validate


def main():
    parser = argparse.ArgumentParser(description="Aggregate and validate article-level schema v2.0 JSON")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--source_chunks", required=True)
    parser.add_argument("--aggregated_dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    aggregated = args.aggregated_dir or os.path.join(os.path.dirname(args.output_dir), "aggregated")
    print({"validated_articles": aggregate_and_validate(args.raw_dir, args.source_chunks, aggregated, args.output_dir)})


if __name__ == "__main__":
    main()

