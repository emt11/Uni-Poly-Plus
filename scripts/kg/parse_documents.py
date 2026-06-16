import argparse
import _common  # noqa: F401
from src.kg_pipeline.document_parser import parse_documents


def main():
    parser = argparse.ArgumentParser(description="Parse local literature documents into SourceChunks")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_tokens", type=int, default=450)
    parser.add_argument("--overlap_tokens", type=int, default=75)
    args = parser.parse_args()
    print({"chunks": parse_documents(args.input_dir, args.output, args.target_tokens, args.overlap_tokens)})


if __name__ == "__main__":
    main()

