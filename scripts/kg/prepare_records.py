import argparse
import _common  # noqa: F401
from src.kg_pipeline.records import prepare_records


def main():
    parser = argparse.ArgumentParser(description="Generate label-free records.csv and repeat_units.csv")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_repeat_units", type=int)
    args = parser.parse_args()
    print(prepare_records(args.input, args.output_dir, args.max_repeat_units))


if __name__ == "__main__":
    main()

