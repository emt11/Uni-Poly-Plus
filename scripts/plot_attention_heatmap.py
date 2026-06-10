import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_modalities(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]

    return None


def parse_attention(value, modalities=None):
    if isinstance(value, (dict, list, tuple)):
        parsed_value = value
    else:
        if pd.isna(value):
            raise ValueError("Missing attention value.")
        parsed_value = None

    text = str(value).strip()
    if not text:
        raise ValueError("Empty attention value.")

    if parsed_value is not None:
        if isinstance(parsed_value, dict):
            return {str(key): float(value) for key, value in parsed_value.items()}
        weights = np.asarray(parsed_value, dtype=float)
        if weights.ndim == 2:
            weights = weights.mean(axis=0)
        if weights.ndim != 1:
            raise ValueError(f"Unsupported attention array shape: {weights.shape}")
        if modalities is None:
            modalities = [f"modality_{idx}" for idx in range(len(weights))]
        if len(modalities) != len(weights):
            raise ValueError(
                f"Modalities length ({len(modalities)}) does not match "
                f"attention length ({len(weights)})."
            )
        return {modality: float(weight) for modality, weight in zip(modalities, weights)}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

        if isinstance(parsed, dict):
            return {str(key): float(value) for key, value in parsed.items()}

        if isinstance(parsed, (list, tuple)):
            weights = np.asarray(parsed, dtype=float)
            if weights.ndim == 2:
                weights = weights.mean(axis=0)
            if weights.ndim != 1:
                raise ValueError(f"Unsupported attention array shape: {weights.shape}")
            if modalities is None:
                modalities = [f"modality_{idx}" for idx in range(len(weights))]
            if len(modalities) != len(weights):
                raise ValueError(
                    f"Modalities length ({len(modalities)}) does not match "
                    f"attention length ({len(weights)})."
                )
            return {modality: float(weight) for modality, weight in zip(modalities, weights)}

    if ":" in text:
        parsed = {}
        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue
            name, raw_weight = item.split(":", 1)
            parsed[name.strip()] = float(raw_weight.strip())
        return parsed

    raise ValueError(f"Could not parse attention value: {text}")


def build_attention_matrix(results_df):
    attention_column = "attention" if "attention" in results_df.columns else "attention_weights"
    if attention_column not in results_df.columns:
        raise ValueError("results.csv must contain an 'attention' or 'attention_weights' column.")
    if "task" not in results_df.columns:
        raise ValueError("results.csv must contain a 'task' column.")

    rows = []
    modality_order = []

    for _, row in results_df.iterrows():
        modalities = parse_modalities(row.get("model_modality_list"))
        attention = parse_attention(row[attention_column], modalities=modalities)
        rows.append((row["task"], attention))
        for modality in attention:
            if modality not in modality_order:
                modality_order.append(modality)

    matrix = pd.DataFrame(
        [
            [attention.get(modality, np.nan) for modality in modality_order]
            for _, attention in rows
        ],
        index=[task for task, _ in rows],
        columns=modality_order,
    )
    matrix.index.name = "task"
    return matrix


def plot_heatmap(matrix, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(4.0, 0.45 * len(matrix.index) + 1.5)
    width = max(5.0, 0.9 * len(matrix.columns) + 2.0)

    plt.figure(figsize=(width, height))
    sns.set_theme(style="white", font_scale=0.95)
    ax = sns.heatmap(
        matrix,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "5-fold mean attention"},
    )
    ax.set_xlabel("modalities")
    ax.set_ylabel("task")
    ax.set_title("Attention Pooling Weights")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot attention pooling heatmap from results.csv.")
    parser.add_argument("--results_csv", default="./results/results.csv", help="Path to results.csv.")
    parser.add_argument(
        "--output",
        default="./results/attention_heatmap.png",
        help="Output path for the heatmap PNG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_df = pd.read_csv(args.results_csv)
    matrix = build_attention_matrix(results_df)
    plot_heatmap(matrix, Path(args.output))
    print(f"Saved attention heatmap to {args.output}")


if __name__ == "__main__":
    main()
