import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SUPPORTED_MODALITIES = ("smiles", "graph", "fp", "geom")


def validate_modalities(modalities):
    if modalities is None:
        return
    unsupported = [modality for modality in modalities if modality not in SUPPORTED_MODALITIES]
    if unsupported:
        raise ValueError(
            f"Unsupported modality: {unsupported[0]}. Current supported modalities are: "
            f"{', '.join(SUPPORTED_MODALITIES)}."
        )


def parse_modalities(value):
    if isinstance(value, (list, tuple)):
        modalities = [str(item) for item in value]
        validate_modalities(modalities)
        return modalities
    if pd.isna(value):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw_value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (list, tuple)):
            modalities = [str(item) for item in parsed]
            validate_modalities(modalities)
            return modalities

    return None


def parse_attention(value, modalities=None):
    if isinstance(value, (dict, list, tuple)):
        parsed_value = value
    else:
        if pd.isna(value):
            raise ValueError("Missing attention value.")
        parsed_value = None

    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Empty attention value.")

    if parsed_value is not None:
        if isinstance(parsed_value, dict):
            result = {str(key): float(value) for key, value in parsed_value.items()}
            validate_modalities(list(result))
            return result
        weights = np.asarray(parsed_value, dtype=float)
        if weights.ndim == 2:
            weights = weights.mean(axis=0)
        if weights.ndim != 1:
            raise ValueError(f"Unsupported attention array shape: {weights.shape}")
        if modalities is None:
            if len(weights) > len(SUPPORTED_MODALITIES):
                raise ValueError("Attention contains more entries than supported modalities.")
            modalities = list(SUPPORTED_MODALITIES[:len(weights)])
        validate_modalities(modalities)
        if len(modalities) != len(weights):
            raise ValueError(
                f"Modalities length ({len(modalities)}) does not match "
                f"attention length ({len(weights)})."
            )
        return {modality: float(weight) for modality, weight in zip(modalities, weights)}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw_value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue

        if isinstance(parsed, dict):
            result = {str(key): float(value) for key, value in parsed.items()}
            validate_modalities(list(result))
            return result

        if isinstance(parsed, (list, tuple)):
            weights = np.asarray(parsed, dtype=float)
            if weights.ndim == 2:
                weights = weights.mean(axis=0)
            if weights.ndim != 1:
                raise ValueError(f"Unsupported attention array shape: {weights.shape}")
            if modalities is None:
                if len(weights) > len(SUPPORTED_MODALITIES):
                    raise ValueError("Attention contains more entries than supported modalities.")
                modalities = list(SUPPORTED_MODALITIES[:len(weights)])
            validate_modalities(modalities)
            if len(modalities) != len(weights):
                raise ValueError(
                    f"Modalities length ({len(modalities)}) does not match "
                    f"attention length ({len(weights)})."
                )
            return {modality: float(weight) for modality, weight in zip(modalities, weights)}

    if ":" in raw_value:
        parsed = {}
        for item in raw_value.split(";"):
            item = item.strip()
            if not item:
                continue
            name, raw_weight = item.split(":", 1)
            parsed[name.strip()] = float(raw_weight.strip())
        validate_modalities(list(parsed))
        return parsed

    raise ValueError(f"Could not parse attention value: {raw_value}")


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
        validate_modalities(modalities)
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
