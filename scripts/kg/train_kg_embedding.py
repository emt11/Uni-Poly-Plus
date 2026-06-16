import argparse
import _common  # noqa: F401
from src.kg_pipeline.embedding import train_transe


def main():
    parser = argparse.ArgumentParser(description="Train TransE and export RepeatUnit KG embeddings")
    parser.add_argument("--triples", required=True)
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--repeat_units", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--model", choices=["transe"], default="transe")
    parser.add_argument("--graph", choices=["strict", "broad"], default="strict")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    print(train_transe(args.triples, args.nodes, args.edges, args.repeat_units, args.output_dir, args.embedding_dim, args.epochs, args.seed, args.graph))


if __name__ == "__main__":
    main()
