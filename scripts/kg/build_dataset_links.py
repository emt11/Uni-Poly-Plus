import argparse
import _common  # noqa: F401
from src.kg_pipeline.linking import build_dataset_links


def main():
    parser = argparse.ArgumentParser(description="Build typed LiteratureSample to RepeatUnit links")
    parser.add_argument("--validated_dir", required=True)
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--polymer_class_candidates", required=True)
    parser.add_argument("--entity_aliases")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print({"dataset_links": build_dataset_links(args.validated_dir, args.repeat_units, args.polymer_class_candidates, args.output, args.entity_aliases)})


if __name__ == "__main__":
    main()

