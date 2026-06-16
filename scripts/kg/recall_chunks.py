import argparse
import _common  # noqa: F401
from src.kg_pipeline.chunk_recall import recall_chunks


def main():
    parser = argparse.ArgumentParser(description="Recall field-specific candidate chunks")
    parser.add_argument("--source_chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bm25_top_k", type=int, default=10)
    parser.add_argument("--dense_top_k", type=int, default=10)
    parser.add_argument("--merged_top_k_per_fact_type", type=int, default=8)
    parser.add_argument("--sample_conditioned_top_k_per_fact_type", type=int, default=5)
    parser.add_argument("--neighbor_window", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=2)
    args = parser.parse_args()
    print({"candidates": recall_chunks(args.source_chunks, args.output, args.bm25_top_k, args.dense_top_k, args.merged_top_k_per_fact_type, args.neighbor_window, args.max_iterations)})


if __name__ == "__main__":
    main()

