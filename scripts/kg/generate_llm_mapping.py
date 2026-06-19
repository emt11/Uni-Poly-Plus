import argparse
import _common  # noqa: F401
from src.kg_pipeline.llm_extract import generate_llm_polymer_mapping


def main():
    parser = argparse.ArgumentParser(description="Generate LLM PolymerClass and alias mappings for RepeatUnits")
    parser.add_argument("--repeat_units", required=True, help="Path to repeat_units.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory, usually kg_work")
    parser.add_argument("--provider", required=True, choices=["qwen", "deepseek"], help="LLM provider")
    parser.add_argument("--model", help="Provider model override. If omitted, falls back to QWEN_MODEL/DEEPSEEK_MODEL/default")
    parser.add_argument("--max_repeat_units", "--max_units", dest="max_repeat_units", type=int, help="Limit repeat units for smoke tests")
    parser.add_argument("--dry_run", action="store_true", help="Build prompts and audit logs without real API calls")
    parser.add_argument("--mock_response", action="store_true", help="Use deterministic mock mapping; no API key required")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mapping outputs")
    parser.add_argument("--sleep_seconds", type=float, default=1.0, help="Sleep between real API calls")
    parser.add_argument("--timeout", type=int, default=120, help="Provider request timeout seconds")
    parser.add_argument("--max_retries", type=int, default=2, help="Provider retry count with exponential backoff")
    args = parser.parse_args()
    try:
        mapped = generate_llm_polymer_mapping(
            args.repeat_units,
            args.output_dir,
            args.provider,
            max_units=args.max_repeat_units,
            mock_response=args.mock_response,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            sleep_seconds=args.sleep_seconds,
            timeout=args.timeout,
            max_retries=args.max_retries,
            model=args.model,
        )
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print({"mapped_repeat_units": mapped, "mapping_mode": "llm", "mock_response": args.mock_response, "dry_run": args.dry_run})


if __name__ == "__main__":
    main()
