import argparse
import _common  # noqa: F401
from src.kg_pipeline.extraction import generate_llm_polymer_mapping


def main():
    parser = argparse.ArgumentParser(description="Generate unverified temporary LLM PolymerClass mappings")
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--provider", required=True, choices=["qwen", "deepseek"])
    parser.add_argument("--max_units", type=int)
    args = parser.parse_args()
    print({"mapped_repeat_units": generate_llm_polymer_mapping(args.repeat_units, args.output_dir, args.provider, args.max_units), "test_mapping": True})


if __name__ == "__main__":
    main()

