import argparse
import _common  # noqa: F401
from src.kg_pipeline.polymer_classes import generate_rule_stub


def main():
    parser = argparse.ArgumentParser(description="Generate temporary PolymerClass mapping candidates")
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["rule_stub"], default="rule_stub")
    args = parser.parse_args()
    print({"mapped_repeat_units": generate_rule_stub(args.repeat_units, args.output_dir), "mode": args.mode})


if __name__ == "__main__":
    main()

