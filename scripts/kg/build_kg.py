import argparse
import _common  # noqa: F401
from src.kg_pipeline.graph_builder import build_kg


def main():
    parser = argparse.ArgumentParser(description="Build Polymer KG nodes, edges, triples and manifest")
    parser.add_argument("--records", required=True)
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--validated_dir", required=True)
    parser.add_argument("--dataset_links", required=True)
    parser.add_argument("--source_chunks")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--graph", choices=["strict", "broad"], default="strict")
    args = parser.parse_args()
    print(build_kg(args.records, args.repeat_units, args.validated_dir, args.dataset_links, args.output_dir, args.graph, args.source_chunks))


if __name__ == "__main__":
    main()

